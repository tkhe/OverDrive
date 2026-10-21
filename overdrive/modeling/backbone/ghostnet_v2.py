import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from overdrive.layers import CNNBlock
from overdrive.layers import Conv2d
from overdrive.layers import DepthwiseSeparableConv2d
from overdrive.layers import ShapeSpec
from overdrive.layers import get_activation
from overdrive.layers import make_divisible
from overdrive.layers import weight_init

from .base import Backbone
from .base import validate_factory_kwargs

__all__ = ["GhostNetV2"]

# Default GhostNet block configurations:
# [kernel_size, expansion_channels, out_channels, use_se, stride]
GHOSTNET_ARCH = [
    [[3, 16, 16, False, 1], [3, 48, 24, False, 2], [3, 72, 24, False, 1]],
    [[5, 72, 40, True, 2], [5, 120, 40, True, 1]],
    [[3, 240, 80, False, 2], [3, 200, 80, False, 1], [3, 184, 80, False, 1], [3, 184, 80, False, 1], [3, 480, 112, True, 1], [3, 672, 112, True, 1]],
    [[5, 672, 160, True, 2], [5, 960, 160, False, 1], [5, 960, 160, True, 1], [5, 960, 160, False, 1], [5, 960, 160, True, 1]],
]


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_channels: int,
        divisor: int = 4,
        activation: str | dict = "ReLU",
        gate_activation: str | dict = "HardSigmoid",
    ):
        super().__init__()

        reduced_channels = make_divisible(in_channels * 0.25, divisor)
        self.conv_reduce = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=True)
        self.activation = get_activation(activation)
        self.conv_expand = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=True)
        self.gate = get_activation(gate_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.activation(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class Stem(CNNBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
    ):
        super().__init__(in_channels, out_channels, 2)

        self.conv = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, norm=norm, activation=activation)

        weight_init.c2_msra_fill(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GhostModule(CNNBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        ratio: int = 2,
        dw_kernel_size: int = 3,
        stride: int = 1,
        norm: str | dict = "BN",
        activation: str | dict | None = "ReLU",
    ):
        super().__init__(in_channels, out_channels, stride)

        primary_channels = math.ceil(out_channels / ratio)
        cheap_channels = primary_channels * (ratio - 1)

        self.primary_conv = Conv2d(in_channels, primary_channels, kernel_size=kernel_size, stride=stride, norm=norm, activation=activation)
        self.cheap_operation = Conv2d(
            primary_channels,
            cheap_channels,
            kernel_size=dw_kernel_size,
            groups=primary_channels,
            norm=norm,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        primary = self.primary_conv(x)
        cheap = self.cheap_operation(primary)
        out = torch.cat([primary, cheap], dim=1)
        return out[:, : self.out_channels, :, :]


class GhostModuleV2(CNNBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        ratio: int = 2,
        dw_kernel_size: int = 3,
        stride: int = 1,
        norm: str | dict = "BN",
        activation: str | dict | None = "ReLU",
    ):
        super().__init__(in_channels, out_channels, stride)

        primary_channels = math.ceil(out_channels / ratio)
        cheap_channels = primary_channels * (ratio - 1)

        self.primary_conv = Conv2d(in_channels, primary_channels, kernel_size=kernel_size, stride=stride, norm=norm, activation=activation)
        self.cheap_operation = Conv2d(
            primary_channels,
            cheap_channels,
            kernel_size=dw_kernel_size,
            groups=primary_channels,
            norm=norm,
            activation=activation,
        )
        self.short_conv = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False, norm=norm),
            Conv2d(out_channels, out_channels, kernel_size=(1, 5), groups=out_channels, bias=False, norm=norm),
            Conv2d(out_channels, out_channels, kernel_size=(5, 1), groups=out_channels, bias=False, norm=norm),
        )
        self.gate = get_activation({"name": "Sigmoid", "inplace": False})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
        primary = self.primary_conv(x)
        cheap = self.cheap_operation(primary)
        out = torch.cat([primary, cheap], dim=1)
        return out[:, : self.out_channels, :, :] * F.interpolate(self.gate(residual), size=out.shape[-2:], mode="nearest")


class GhostBottleneck(CNNBlock):
    """
    GhostNet bottleneck with optional downsampling, squeeze-excitation and attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_se: bool = False,
        use_attention: bool = False,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
    ):
        super().__init__(in_channels, out_channels, stride)

        ghost_module = GhostModuleV2 if use_attention else GhostModule
        self.ghost1 = ghost_module(in_channels, mid_channels, norm=norm, activation=activation)
        self.conv_dw = Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, groups=mid_channels, norm=norm) if stride > 1 else None
        self.se = SqueezeExcite(mid_channels, activation=activation) if use_se else None
        self.ghost2 = GhostModule(mid_channels, out_channels, norm=norm, activation=None)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, norm=norm)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ghost1(x)
        if self.conv_dw is not None:
            x = self.conv_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        return x + residual


class GhostNetV2(Backbone):
    """
    GhostNet v2 (https://arxiv.org/abs/2211.12905).
    """

    def __init__(
        self,
        arch: list[list] = GHOSTNET_ARCH,
        width_multi: float = 1.0,
        input_shape: ShapeSpec | None = None,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
        out_features: list[str] | None = None,
        freeze_at: int = 0,
        num_classes: int | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        if input_shape is None:
            input_shape = ShapeSpec(channels=3)

        self.num_classes = num_classes

        if out_features is not None:
            num_stages = max([{"stage2": 1, "stage3": 2, "stage4": 3, "stage5": 4, "conv5": 4}.get(f, 0) for f in out_features])
            arch = arch[:num_stages]

        stem_channels = make_divisible(16 * width_multi, 4)
        self.stem = Stem(input_shape.channels, stem_channels, norm, activation)
        current_stride = self.stem.stride
        self._out_feature_channels["stem"] = stem_channels
        self._out_feature_strides["stem"] = current_stride
        in_channels = stem_channels

        self.stages = []
        layer_idx = 0
        for idx, stage_cfg in enumerate(arch):
            name = f"stage{idx + 2}"
            blocks = []
            for kernel_size, mid_channels, out_channels, use_se, stride in stage_cfg:
                mid_channels = make_divisible(mid_channels * width_multi, 4)
                out_channels = make_divisible(out_channels * width_multi, 4)
                blocks.append(
                    GhostBottleneck(
                        in_channels,
                        out_channels,
                        mid_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        use_se=use_se,
                        use_attention=layer_idx > 1,
                        norm=norm,
                        activation=activation,
                    )
                )
                current_stride *= stride
                in_channels = out_channels
                layer_idx += 1

            self.add_module(name, nn.Sequential(*blocks))
            self.stages.append(name)
            self._out_feature_channels[name] = in_channels
            self._out_feature_strides[name] = current_stride

        if out_features is not None and "conv5" in out_features:
            conv5_channels = make_divisible(arch[-1][-1][1] * width_multi, 4)
            self.conv5 = Conv2d(in_channels, conv5_channels, kernel_size=1, norm=norm, activation=activation)
            self.stages.append("conv5")
            self._out_feature_channels["conv5"] = conv5_channels
            self._out_feature_strides["conv5"] = current_stride
            in_channels = conv5_channels

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.conv_head = Conv2d(in_channels, 1280, kernel_size=1, bias=True, activation=activation)
            self.linear = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(1280, num_classes),
            )
            weight_init.normal_fill(self.linear[1], std=0.01)

        if out_features is None:
            out_features = [self.stages[-1]]

        self._out_features = out_features
        if len(self._out_features) == 0:
            raise ValueError("out_features must contain at least one feature name.")

        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            if out_feature not in children:
                raise ValueError(f"Available children: {', '.join(children)}")

        self.freeze(freeze_at)

    @staticmethod
    def build(name, **kwargs):
        predefined = {
            "GhostNet-V2-x0.5": ghostnet_v2_x0_5,
            "GhostNet-V2-x1.0": ghostnet_v2_x1_0,
            "GhostNet-V2-x1.3": ghostnet_v2_x1_3,
            "GhostNet-V2-x1.5": ghostnet_v2_x1_5,
            "GhostNet-V2-x1.6": ghostnet_v2_x1_6,
            "GhostNet-V2-x2.0": ghostnet_v2_x2_0,
        }

        if name not in predefined:
            raise ValueError(f"`{name}` is not predefined for GhostNetV2. Available names: {', '.join(predefined)}.")

        return predefined[name](**kwargs)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}

        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x

        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if stage_name in self._out_features:
                outputs[stage_name] = x

        if self.num_classes is not None:
            x = self.avgpool(x)
            x = self.conv_head(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            outputs["linear"] = x

        return outputs

    def freeze(self, freeze_at: int = 0):
        if freeze_at >= 1:
            self.stem.freeze()

        for idx, stage_name in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                module = getattr(self, stage_name)
                if stage_name == "conv5":
                    module.freeze()
                else:
                    for block in module.children():
                        block.freeze()


@validate_factory_kwargs("arch", "width_multi")
def ghostnet_v2_x0_5(**kwargs):
    return GhostNetV2(arch=GHOSTNET_ARCH, width_multi=0.5, **kwargs)


@validate_factory_kwargs("arch", "width_multi")
def ghostnet_v2_x1_0(**kwargs):
    return GhostNetV2(arch=GHOSTNET_ARCH, width_multi=1.0, **kwargs)


@validate_factory_kwargs("arch", "width_multi")
def ghostnet_v2_x1_3(**kwargs):
    return GhostNetV2(arch=GHOSTNET_ARCH, width_multi=1.3, **kwargs)


@validate_factory_kwargs("arch", "width_multi")
def ghostnet_v2_x1_5(**kwargs):
    return GhostNetV2(arch=GHOSTNET_ARCH, width_multi=1.5, **kwargs)


@validate_factory_kwargs("arch", "width_multi")
def ghostnet_v2_x1_6(**kwargs):
    return GhostNetV2(arch=GHOSTNET_ARCH, width_multi=1.6, **kwargs)


@validate_factory_kwargs("arch", "width_multi")
def ghostnet_v2_x2_0(**kwargs):
    return GhostNetV2(arch=GHOSTNET_ARCH, width_multi=2.0, **kwargs)
