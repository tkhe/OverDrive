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
    Map published DN-DETR configurations to model checkpoints.
    """

    CONFIG_PATH_TO_URL_SUFFIX = {}

    @staticmethod
    def query(config_path: str) -> str | None:
        name = config_path.replace(".py", "")
        return _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX.get(name)


def get_checkpoint_url(config_path: str) -> str:
    """
    Return a published checkpoint URL for a configuration.
    """
    url = _ModelZooUrls.query(config_path)
    if url is None:
        raise RuntimeError(f"Pretrained model for {config_path} is not available!")
    return url


def get_config_file(config_path: str) -> str:
    """
    Return the path to a DN-DETR configuration file.
    """
    config_file = str(importlib.resources.files("dn_detr.model_zoo").joinpath(os.path.join("configs", config_path)))
    if not os.path.exists(config_file):
        raise RuntimeError(f"{config_path} not available in Model Zoo!")
    return config_file


def get_config(config_path: str, trained: bool = False):
    """
    Load a DN-DETR configuration by its project-relative path.
    """
    config_file = get_config_file(config_path)
    if not config_file.endswith(".py"):
        raise ValueError(f"Expected a Python config file, got {config_file}")

    config = LazyConfig.load(config_file)
    if trained:
        if "train" not in config or "init_checkpoint" not in config.train:
            raise NotImplementedError
        config.train.init_checkpoint = get_checkpoint_url(config_path)
    return config


def get(config_path: str, trained: bool = False, device: str | None = None):
    """
    Instantiate a DN-DETR model and load its configured checkpoint.
    """
    config = get_config(config_path, trained)
    if device is None and not torch.cuda.is_available():
        device = "cpu"

    model = instantiate(config.model)
    if device is not None:
        model = model.to(device)
    if "train" in config and "init_checkpoint" in config.train:
        Checkpointer(model).load(config.train.init_checkpoint)
    return model
