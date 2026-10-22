import math

import torch
import torch.nn as nn

from overdrive.layers import Conv2d
from overdrive.layers import ShapeSpec
from overdrive.layers import get_activation
from overdrive.layers import get_norm

from .base import Neck
from .fpn import FPNTopBlock
from .fpn import _assert_strides_are_log2_contiguous

__all__ = ["SimpleFPN"]


class SimpleFPN(Neck):
    """
    Create a simple feature pyramid from one precomputed backbone feature map.
    """

    def __init__(
        self,
        input_shape: dict[str, ShapeSpec],
        in_features: list[str],
        out_channels: int,
        scale_factors: list[float],
        top_block: FPNTopBlock | None = None,
        norm: str | dict = {"name": "LN2d", "eps": 1e-6},
        activation: str | dict | nn.Module | None = "GELU",
    ):
        """
        Args:
            input_shape: Shapes of the precomputed backbone features.
            in_features: List containing exactly one feature name used by all pyramid branches.
            out_channels: Number of channels in each output feature map.
            scale_factors: Scaling factors, chosen from 4.0, 2.0, 1.0, and 0.5.
                The resulting strides must be consecutive powers of two in increasing order.
            top_block: Optional block that adds coarser pyramid levels.
            norm: Normalization specification. Defaults to channel normalization
                for NCHW tensors with eps=1e-6.
            activation: Activation between the two transpose convolutions in the
                4.0 branch. Falsy values disable it. Defaults to GELU.
        """
        super().__init__()

        if len(in_features) != 1:
            raise ValueError(f"Expected exactly one input feature, got {len(in_features)}")

        if not scale_factors:
            raise ValueError("Expected at least one scale factor")

        for scale in scale_factors:
            if scale not in (4.0, 2.0, 1.0, 0.5):
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

        dim = input_shape[in_features[0]].channels
        input_stride = input_shape[in_features[0]].stride
        if input_stride is None or input_stride <= 0:
            raise ValueError(f"Expected a positive input stride, got {input_stride}")

        strides = []
        for scale in scale_factors:
            stride = input_stride / scale
            if stride < 1 or stride != int(stride):
                raise ValueError(f"Expected a positive integer pyramid stride, got {stride}")
            stride = int(stride)
            if stride & (stride - 1):
                raise ValueError(f"Expected a power-of-two pyramid stride, got {stride}")
            strides.append(stride)
        _assert_strides_are_log2_contiguous(strides)

        self.scale_factors = scale_factors
        self.stages: list[str] = []
        use_bias = not norm

        for scale, stride in zip(scale_factors, strides, strict=True):
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    get_activation(activation),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                    Conv2d(dim // 4, out_channels, kernel_size=1, bias=use_bias, norm=norm),
                    Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias, norm=norm),
                ]
            elif scale == 2.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    Conv2d(dim // 2, out_channels, kernel_size=1, bias=use_bias, norm=norm),
                    Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias, norm=norm),
                ]
            elif scale == 1.0:
                layers = [
                    Conv2d(dim, out_channels, kernel_size=1, bias=use_bias, norm=norm),
                    Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias, norm=norm),
                ]
            else:
                layers = [
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    Conv2d(dim, out_channels, kernel_size=1, bias=use_bias, norm=norm),
                    Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias, norm=norm),
                ]

            layers = [layer for layer in layers if layer is not None]
            stage = int(math.log2(stride))
            name = f"simfpn_{stage}"
            self.add_module(name, nn.Sequential(*layers))
            self.stages.append(name)

        self.in_features = tuple(in_features)
        self.top_block = top_block
        self._out_feature_strides = {f"p{int(math.log2(s))}": s for s in strides}
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides[f"p{s + 1}"] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {name: out_channels for name in self._out_features}
        self._size_divisibility = strides[-1]

    def forward(self, bottom_up_features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Return pyramid features ordered from highest to lowest resolution.
        """
        features = bottom_up_features[self.in_features[0]]
        results = []

        for stage in self.stages:
            results.append(getattr(self, stage)(features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))

        if len(self._out_features) != len(results):
            raise ValueError(f"Expected len(self._out_features) == len(results), but got len(self._out_features)={len(self._out_features)} and len(results)={len(results)}")

        return {feature: result for feature, result in zip(self._out_features, results, strict=True)}

    @property
    def padding_constraints(self) -> dict[str, int]:
        return {"size_divisibility": self._size_divisibility}
