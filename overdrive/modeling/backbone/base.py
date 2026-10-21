from abc import ABC
from abc import abstractmethod
from functools import wraps

import torch
import torch.nn as nn

from overdrive.layers import ShapeSpec

__all__ = ["Backbone"]


class Backbone(nn.Module, ABC):
    """
    Abstract base class for backbones.
    """

    def __init__(self):
        super().__init__()

        self._out_features: list[str] = []
        self._out_feature_channels: dict[str, int] = {}
        self._out_feature_strides: dict[str, int] = {}

    @abstractmethod
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Subclasses must override this method, but adhere to the same return type.

        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "res2") to tensor
        """
        raise NotImplementedError

    @property
    def padding_constraints(self) -> dict[str, int | tuple[int, int]]:
        return {
            "size_divisibility": 0,
            "pad_size": 0,
        }

    def output_shape(self) -> dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at: int = 0):
        raise NotImplementedError


def validate_factory_kwargs(*predefined_keys: str):
    """
    Reject keyword arguments that override predefined factory parameters.
    """
    if not all(isinstance(key, str) for key in predefined_keys):
        raise TypeError("Predefined parameter names must be strings.")

    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            conflicting_keys = [key for key in predefined_keys if key in kwargs]
            if conflicting_keys:
                formatted_keys = ", ".join(f"`{key}`" for key in conflicting_keys)
                raise ValueError(f"Predefined parameters for `{func.__name__}` cannot be passed through `kwargs`: {formatted_keys}.")
            return func(*args, **kwargs)

        return wrapped

    return decorator
