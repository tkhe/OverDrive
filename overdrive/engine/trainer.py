import copy
import logging
import os
import weakref
from collections import OrderedDict
from contextlib import nullcontext

import torch

from overdrive.checkpoint import Checkpointer
from overdrive.config import instantiate
from overdrive.config import try_get_key
from overdrive.evaluation import DatasetEvaluator
from overdrive.evaluation import inference_on_dataset
from overdrive.evaluation import print_csv_format
from overdrive.utils import comm
from overdrive.utils.file_io import PathManager
from overdrive.utils.logger import setup_logger

from . import backend
from . import hook
from . import loop

__all__ = [
    "SingleTeacherDistillationTrainer",
    "Trainer",
]

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        logger = logging.getLogger("overdrive")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for overdrive
            setup_logger()

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        dataloader = self.build_train_dataloader(cfg)

        self.backend = self.build_backend(cfg, model, optimizer, dataloader)
        self.loop = self.build_loop(cfg, self.backend)
        scheduler = self.build_lr_scheduler(cfg)

        self.checkpointer = Checkpointer(self.backend.model, cfg.train.output_dir, loop=weakref.proxy(self.loop), backend=weakref.proxy(self.backend))
        self.loop.checkpointer = weakref.proxy(self.checkpointer)
        self.loop.register_hooks(self.build_default_hooks(scheduler))
        self.loop.register_hooks(self.build_custom_hooks())

    def resume_or_load(self, resume: bool = True):
        self.checkpointer.resume_or_load(self.cfg.train.init_checkpoint, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            if self.loop.mode == "epoch":
                self.loop.start_epoch = self.loop.epoch + 1
            elif self.loop.mode == "iteration":
                self.loop.start_iter = self.loop.iter + 1
            else:
                raise ValueError(f"Unsupported training mode: {self.loop.mode}")

    def train(self):
        if self.loop.mode == "epoch":
            max_epoch = self.cfg.train.max_epoch
            start_epoch = self.loop.start_epoch
            self.loop.run(start_epoch, max_epoch)
        elif self.loop.mode == "iteration":
            max_iter = self.cfg.train.max_iter
            start_iter = self.loop.start_iter
            self.loop.run(start_iter, max_iter)
        else:
            raise ValueError(f"Unsupported training mode: {self.loop.mode}")

    def build_default_hooks(self, lr_scheduler):
        cfg = copy.deepcopy(self.cfg)

        ret = [hook.IterationTimeHook()]
        if lr_scheduler is not None:
            ret.append(hook.LRSchedulerHook(scheduler=lr_scheduler))

        if comm.is_main_process():
            ret.append(hook.CheckpointHook(self.checkpointer, **cfg.train.checkpointer))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hook.EvalHook(self.cfg.train.eval_period, test_and_save_results))

        if comm.is_main_process():
            ret.append(hook.CommonMetricPrinterHook(try_get_key(cfg, "train.log_period", default=10)))
            ret.append(hook.TensorboardXHook(try_get_key(cfg, "train.log_period", default=10), cfg.train.output_dir))

        return ret

    def build_custom_hooks(self):
        cfg = copy.deepcopy(self.cfg)

        hooks = []
        for hook_cfg in try_get_key(cfg, "train.custom_hooks", default=[]):
            hooks.append(instantiate(hook_cfg))
        return hooks

    @classmethod
    def build_backend(cls, cfg, model, optimizer, dataloader):
        backend_cfg = try_get_key(cfg, "train.backend", default="native")
        backend_kwargs = dict(
            ddp=try_get_key(cfg, "train.ddp", default={}),
            amp=try_get_key(cfg, "train.amp", default={}),
            gradient_clipping=try_get_key(cfg, "train.gradient_clipping", default={}),
            gradient_accumulation_steps=try_get_key(cfg, "train.gradient_accumulation_steps", default=1),
        )
        if backend_cfg == "native":
            return backend.NativeBackend(model, optimizer, dataloader, **backend_kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend_cfg}")

    @classmethod
    def build_loop(cls, cfg, backend):
        mode = try_get_key(cfg, "train.mode", default="epoch")
        if mode == "epoch":
            return loop.EpochLoop(backend)
        elif mode == "iteration":
            return loop.IterationLoop(backend)
        else:
            raise ValueError(f"Unsupported training mode: {mode}")

    @classmethod
    def build_model(cls, cfg):
        model = instantiate(cfg.model)
        model.to(torch.device(cfg.train.device))
        logger = logging.getLogger(__name__)
        logger.info(f"Model:\n{model}")
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        cfg = copy.deepcopy(cfg)
        cfg.optimizer.params.model = model
        optimizer = instantiate(cfg.optimizer)
        return optimizer

    @classmethod
    def build_lr_scheduler(cls, cfg):
        return instantiate(cfg.lr_multiplier)

    @classmethod
    def build_train_dataloader(cls, cfg):
        dataloader = instantiate(cfg.dataloader.train)
        return dataloader

    @classmethod
    def build_test_dataloader(cls, cfg, dataset_name):
        origin_dataset_names = cfg.dataloader.test.dataset.names
        cfg.dataloader.test.dataset.names = dataset_name
        dataloader = instantiate(cfg.dataloader.test)
        cfg.dataloader.test.dataset.names = origin_dataset_names
        return dataloader

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        logger = logging.getLogger(__name__)

        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]

        dataset_names = cfg.dataloader.test.dataset.names
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        if evaluators is not None:
            if not (len(dataset_names) == len(evaluators)):
                raise ValueError(f"{len(dataset_names)} != {len(evaluators)}")

        results = OrderedDict()
        for idx, dataset_name in enumerate(dataset_names):
            data_loader = cls.build_test_dataloader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warning("No evaluator found. Use `Trainer.test(evaluators=)`, or implement its `build_evaluator` method.")
                    results[dataset_name] = {}
                    continue

            with cls._build_eval_autocast_context(cfg):
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i

            if comm.is_main_process():
                if not (isinstance(results_i, dict)):
                    raise TypeError(f"Evaluator must return a dict on the main process. Got {results_i} instead.")

                logger.info(f"Evaluation results for {dataset_name} in csv format:")
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @staticmethod
    def _build_eval_autocast_context(cfg):
        amp = try_get_key(cfg, "train.amp", default={})
        if not amp.get("enable", False):
            return nullcontext()

        mixed_precision = amp.get("mixed_precision", "fp16")
        if mixed_precision not in {"fp16", "bf16"}:
            raise ValueError(f"Unsupported mixed_precision for evaluation: {mixed_precision}")

        amp_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[mixed_precision]
        return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Returns:
            DatasetEvaluator or None

        It is not implemented by default.
        """
        raise NotImplementedError(
            "If you want Trainer to automatically run evaluation, please implement `build_evaluator()` in subclasses (see train_net.py for example). Alternatively, you can call evaluation functions yourself."
        )

    @property
    def model(self):
        return self.backend.model

    @model.setter
    def model(self, model):
        self.backend.model = model

    @property
    def optimizer(self):
        return self.backend.optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.backend.optimizer = optimizer

    @property
    def dataloader(self):
        return self.backend.dataloader

    @dataloader.setter
    def dataloader(self, dataloader):
        self.backend.dataloader = dataloader

    def reset_dataloader(self, dataloader_builder):
        self.backend.reset_dataloader(dataloader_builder)


class SingleTeacherDistillationTrainer(Trainer):
    def resume_or_load(self, resume=True):
        cfg = self.cfg
        checkpoint = None
        if resume:
            marker = os.path.join(cfg.train.output_dir, "last_checkpoint")
            if PathManager.isfile(marker):
                with PathManager.open(marker) as handle:
                    name = handle.read().strip()
                if name:
                    latest = os.path.join(cfg.train.output_dir, name)
                    if PathManager.isfile(latest):
                        checkpoint = latest
            if not checkpoint:
                logger.warning(f"No latest checkpoint found in {cfg.train.output_dir}; trying train.init_checkpoint.resume")

        if not checkpoint:
            checkpoint = cfg.train.init_checkpoint.get("resume")
            if resume and (not checkpoint or not PathManager.isfile(checkpoint)):
                logger.warning(f"No checkpoint found at train.init_checkpoint.resume={checkpoint!r}; initializing student and teacher")
                checkpoint = None

        if checkpoint:
            self.checkpointer.load(checkpoint, checkpointables=None if resume else [])
        else:
            model = self.checkpointer.model
            Checkpointer(model.student).load(cfg.train.init_checkpoint.get("student"))
            Checkpointer(model.teacher).load(cfg.train.init_checkpoint.get("teacher"))
