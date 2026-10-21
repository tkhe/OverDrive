from abc import ABC
from abc import abstractmethod

import torch
import torch.nn as nn

from overdrive.layers import FrozenBatchNorm2d
from overdrive.layers import ShapeSpec

__all__ = ["Neck"]


class Neck(nn.Module, ABC):
    """
    Abstract base class for neck.
    """

    def __init__(self):
        super().__init__()

        self._out_features: list[str] = []
        self._out_feature_channels: dict[str, int] = {}
        self._out_feature_strides: dict[str, int] = {}

    @abstractmethod
    def forward(self, bottom_up_features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Subclasses must override this method, but adhere to the same return type.

        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "p5") to tensor
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

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batch_norm(self)
        return self
