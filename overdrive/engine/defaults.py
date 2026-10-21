import argparse
import copy
import os
import sys
from contextlib import nullcontext

import torch

from overdrive.checkpoint import Checkpointer
from overdrive.config import LazyConfig
from overdrive.config import instantiate
from overdrive.config import try_get_key
from overdrive.data import transforms as T
from overdrive.data import utils
from overdrive.utils import comm
from overdrive.utils.collect_env import collect_env_info
from overdrive.utils.env import seed_all_rng
from overdrive.utils.file_io import PathManager
from overdrive.utils.logger import setup_logger

__all__ = [
    "DefaultPredictor",
    "default_argument_parser",
    "default_setup",
]


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by OverDrive users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.py

Change some config options:
    $ {sys.argv[0]} --config-file cfg.py a=b c.d.e=f

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--resume", action="store_true", help="Whether to attempt to resume from the checkpoint directory.")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)")

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default=f"tcp://127.0.0.1:{port}",
        help="initialization URL for pytorch distributed backend. See https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help='Modify config options at the end of the command. For python-based LazyConfig, use "path.key=value".',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def _highlight(code, filename):
    try:
        import pygments
    except ImportError:
        return code

    from pygments.formatters import Terminal256Formatter
    from pygments.lexers.data import YamlLexer
    from pygments.lexers.python import Python3Lexer

    lexer = Python3Lexer() if filename.endswith(".py") else YamlLexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))
    return code


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the OverDrive logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = try_get_key(cfg, "train.output_dir", "output_dir", default=None)
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info(f"Rank of current process: {rank}. World size: {comm.get_world_size()}")
    logger.info(f"Environment info:\n{collect_env_info()}")

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(f"Contents of args.config_file={args.config_file}:\n{_highlight(PathManager.open(args.config_file, 'r').read(), args.config_file)}")

    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        LazyConfig.save(cfg, path)
        logger.info(f"Full config saved to {path}")

    # make sure each worker has a different, yet deterministic seed if specified
    seed = try_get_key(cfg, "train.seed", "seed", default=-1)
    seed_all_rng(None if seed < 0 else seed + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = try_get_key(cfg, "train.cudnn_benchmark", "cudnn_benchmark", default=False)


class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.train.init_checkpoint`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.dataloader.test.mapper.image_format`.
    3. Apply augmentations defined by `cfg.dataloader.test.mapper.augmentations`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:

        >>> pred = DefaultPredictor(cfg)
        >>> inputs = cv2.imread("input.jpg")
        >>> outputs = pred(inputs)
    """

    def __init__(self, cfg, precision=torch.float32):
        self.cfg = copy.deepcopy(cfg)
        self.device = torch.device(self.cfg.train.device)
        self.precision = self._normalize_inference_precision(precision)
        self.autocast_enabled = self.precision != torch.float32
        if self.precision == torch.float16 and self.device.type != "cuda":
            raise ValueError("DefaultPredictor precision=torch.float16 requires a CUDA device.")
        if self.autocast_enabled and self.device.type not in {"cuda", "cpu"}:
            raise ValueError(f"DefaultPredictor precision='{self.precision}' is not supported on device '{self.device}'.")

        self.model = instantiate(self.cfg.model)
        self.model.eval()
        self.model.to(self.device)

        checkpointer = Checkpointer(self.model)
        checkpointer.load(cfg.train.init_checkpoint)

        self.augmentations = T.AugmentationList(instantiate(cfg.dataloader.test.mapper.augmentations))
        self.input_format = cfg.dataloader.test.mapper.image_format
        if self.input_format not in ["RGB", "BGR"]:
            raise ValueError(self.input_format)

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
        """
        with torch.no_grad():
            height, width = original_image.shape[:2]
            aug_input = T.AugInput(image=original_image)
            transforms = self.augmentations(aug_input)
            image = utils.convert_bgr_to_image_format(aug_input.image, self.input_format)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image = image.to(self.device)

            inputs = {"image": image, "height": height, "width": width, "transforms": transforms}

            with self._autocast_context():
                predictions = self.model([inputs])[0]
            return predictions

    def _autocast_context(self):
        if not self.autocast_enabled:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.precision, enabled=True)

    @staticmethod
    def _normalize_inference_precision(precision):
        if precision is None:
            precision = torch.float32
        if precision not in {torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError("DefaultPredictor precision must be one of: torch.float32, torch.float16, torch.bfloat16.")
        return precision
