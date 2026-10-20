from abc import ABC
from abc import abstractmethod

import torch.nn as nn

from .misc import FreezableMixin

__all__ = ["CNNBlock"]


class CNNBlock(nn.Module, FreezableMixin, ABC):
    """
    A CNN block is assumed to have input channels, output channels and a stride.
    The input and output of `forward()` method must be NCHW tensors.
    The method can perform arbitrary computation but must match the given
    channels and stride specification.

    Attribute:
        in_channels (int):
        out_channels (int):
        stride (int):
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError
