import importlib.resources
import os

import torch
from overdrive.checkpoint import Checkpointer
from overdrive.config import LazyConfig
from overdrive.config import instantiate

__all__ = [
    "get",
    "get_checkpoint_url",
    "get_config",
    "get_config_file",
]


class _ModelZooUrls:
    """
    Mapping from names to released pre-trained models.
    """

    CONFIG_PATH_TO_URL_SUFFIX = {}

    @staticmethod
    def query(config_path: str) -> str | None:
        """
        Args:
            config_path: relative config filename
        """
        name = config_path.replace(".py", "")
        if name in _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX:
            return _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX[name]
        return None


def get_checkpoint_url(config_path):
    """
    Returns the URL to the model trained using the given config

    Args:
        config_path (str): config file name relative to centernet's "configs/" directory

    Returns:
        str: a URL to the model
    """
    url = _ModelZooUrls.query(config_path)
    if url is None:
        raise RuntimeError(f"Pretrained model for {config_path} is not available!")
    return url


def get_config_file(config_path):
    """
    Returns path to a builtin config file.

    Args:
        config_path (str): config file name relative to centernet's "configs/" directory

    Returns:
        str: the real path to the config file.
    """
    cfg_file = str(importlib.resources.files("centernet.model_zoo").joinpath(os.path.join("configs", config_path)))
    if not os.path.exists(cfg_file):
        raise RuntimeError(f"{config_path} not available in Model Zoo!")
    return cfg_file


def get_config(config_path, trained: bool = False):
    """
    Returns a config object for a model in model zoo.

    Args:
        config_path (str): config file name relative to centernet's "configs/" directory
        trained (bool): If True, will set `train.init_checkpoint` to trained model zoo weights.
            If False, the checkpoint specified in the config file's `train.init_checkpoint` is used
            instead; this will typically (though not always) initialize a subset of weights using
            an ImageNet pre-trained model, while randomly initializing the other weights.

    Returns:
        omegaconf.DictConfig: a config object
    """
    cfg_file = get_config_file(config_path)

    if not cfg_file.endswith(".py"):
        raise ValueError(f"Expected a Python config file, got {cfg_file}")

    cfg = LazyConfig.load(cfg_file)
    if trained:
        url = get_checkpoint_url(config_path)
        if "train" in cfg and "init_checkpoint" in cfg.train:
            cfg.train.init_checkpoint = url
        else:
            raise NotImplementedError
    return cfg


def get(config_path, trained: bool = False, device: str | None = None):
    """
    Get a model specified by relative path under centernet's official ``configs/`` directory.

    Args:
        config_path (str): config file name relative to centernet's "configs/" directory
        trained (bool): see :func:`get_config`.
        device (str or None): overwrite the device in config, if given.

    Returns:
        nn.Module: a model. Will be in training mode.

    """
    cfg = get_config(config_path, trained)
    if device is None and not torch.cuda.is_available():
        device = "cpu"

    model = instantiate(cfg.model)
    if device is not None:
        model = model.to(device)
    if "train" in cfg and "init_checkpoint" in cfg.train:
        Checkpointer(model).load(cfg.train.init_checkpoint)
    return model
