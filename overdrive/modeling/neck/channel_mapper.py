import math

import torch
import torch.nn as nn

from overdrive.layers import Conv2d
from overdrive.layers import ShapeSpec
from overdrive.layers import weight_init

from .base import Neck
from .fpn import FPNTopBlock

__all__ = ["ChannelMapper"]


class LastLevelConvs(FPNTopBlock):
    """
    Stacked 3x3 stride-2 convs to generate extra levels after the last level.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_levels: int = 1,
        norm: str | dict = "",
        activation: str | dict = "",
        in_feature: str = "p5",
    ):
        super().__init__()

        if num_levels <= 0:
            raise ValueError(f"num_levels must be positive, got {num_levels}")

        self._num_levels = num_levels
        self._in_feature = in_feature

        self.convs = nn.ModuleList()
        for idx in range(num_levels):
            conv = Conv2d(
                in_channels if idx == 0 else out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                norm=norm,
                activation=activation,
            )
            weight_init.xavier_uniform_fill(conv)
            self.convs.append(conv)

    def forward(self, x):
        results = []
        for conv in self.convs:
            x = conv(x)
            results.append(x)
        return results


class ChannelMapper(Neck):
    """
    Channel Mapper for reduce/increase channels of backbone features.
    """

    def __init__(
        self,
        input_shape: dict[str, ShapeSpec],
        in_features: list[str],
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
        top_block: FPNTopBlock | None = None,
    ):
        super().__init__()

        self.in_features = in_features

        self.convs = nn.ModuleList()
        for in_channels in [input_shape[f].channels for f in in_features]:
            conv = Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                norm=norm,
                activation=activation,
            )
            weight_init.xavier_uniform_fill(conv)
            self.convs.append(conv)

        self.top_block = top_block

        strides = [input_shape[f].stride for f in in_features]
        self._out_feature_strides = {f"p{int(math.log2(s))}": s for s in strides}
        if self.top_block is not None:
            stage = int(math.log2(strides[-1]))
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides[f"p{s + 1}"] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {f: out_channels for f in self._out_features}

    def forward(self, bottom_up_features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        results = []

        for idx, f in enumerate(self.in_features):
            results.append(self.convs[idx](bottom_up_features[f]))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))

        return {f: res for f, res in zip(self._out_features, results, strict=True)}
