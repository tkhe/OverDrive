from collections.abc import Sequence

import torch
import torch.nn as nn

from overdrive.layers import CNNBlock
from overdrive.layers import Conv2d
from overdrive.layers import DepthwiseSeparableConv2d
from overdrive.layers import ShapeSpec
from overdrive.layers import make_divisible
from overdrive.layers import weight_init

from .base import Backbone
from .base import validate_factory_kwargs

__all__ = ["MobileNetV1"]


class Stem(CNNBlock):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU6",
    ):
        super().__init__(in_channels, out_channels, 2)

        self.conv = Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, norm=norm, activation=activation)
        self.dw = DepthwiseSeparableConv2d(mid_channels, out_channels, kernel_size=3, stride=1, norm=norm, activation=activation)

        weight_init.c2_msra_fill(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.dw(x)
        return x


class MobileNetV1(Backbone):
    """
    support MobileNet v1 (https://arxiv.org/abs/1704.04861).
    """

    def __init__(
        self,
        num_blocks: list[int],
        out_channels_per_stage: list[int],
        strides: Sequence[int] = (2, 2, 2, 2),
        width_multi: float = 1.0,
        input_shape: ShapeSpec | None = None,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU6",
        out_features: list[str] | None = None,
        freeze_at: int = 0,
        num_classes: int | None = None,
    ):
        super().__init__()

        if input_shape is None:
            input_shape = ShapeSpec(channels=3)

        self.num_classes = num_classes

        if out_features is not None:
            num_stages = max([{"stage2": 1, "stage3": 2, "stage4": 3, "stage5": 4}.get(f, 0) for f in out_features])
            num_blocks = num_blocks[:num_stages]
            out_channels_per_stage = out_channels_per_stage[:num_stages]
            strides = strides[:num_stages]

        # Build stem
        stem_mid_channels = make_divisible(32 * width_multi, 8)
        stem_out_channels = make_divisible(64 * width_multi, 8)
        self.stem = Stem(input_shape.channels, stem_mid_channels, stem_out_channels, norm, activation)
        current_stride = self.stem.stride
        self._out_feature_strides["stem"] = current_stride
        self._out_feature_channels["stem"] = stem_out_channels
        in_channels = stem_out_channels

        # Build stages
        self.stages = []
        for idx, (n, c, s) in enumerate(zip(num_blocks, out_channels_per_stage, strides, strict=True)):
            name = f"stage{idx + 2}"
            out_channels = make_divisible(c * width_multi, 8)
            blocks = [DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, stride=s, norm=norm, activation=activation)]
            for _ in range(1, n):
                blocks.append(DepthwiseSeparableConv2d(out_channels, out_channels, kernel_size=3, stride=1, norm=norm, activation=activation))
            current_stride *= s

            self.add_module(name, nn.Sequential(*blocks))
            self.stages.append(name)
            self._out_feature_channels[name] = out_channels
            self._out_feature_strides[name] = current_stride
            in_channels = out_channels

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(in_channels, num_classes)
            weight_init.normal_fill(self.linear, std=0.01)

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
            "MobileNet-V1-x0.25": mobilenet_v1_x0_25,
            "MobileNet-V1-x0.5": mobilenet_v1_x0_5,
            "MobileNet-V1-x0.75": mobilenet_v1_x0_75,
            "MobileNet-V1-x1.0": mobilenet_v1_x1_0,
        }

        if name not in predefined:
            raise ValueError(f"`{name}` is not predefined for MobileNet. Available names: {', '.join(predefined)}.")

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
            x = torch.flatten(x, 1)
            x = self.linear(x)
            outputs["linear"] = x

        return outputs

    def freeze(self, freeze_at: int = 0):
        if freeze_at >= 1:
            self.stem.freeze()

        for idx, stage_name in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                stage = getattr(self, stage_name)
                for block in stage.children():
                    block.freeze()


@validate_factory_kwargs("num_blocks", "out_channels_per_stage", "width_multi")
def mobilenet_v1_x0_25(**kwargs):
    return MobileNetV1(
        num_blocks=[2, 2, 6, 2],
        out_channels_per_stage=[128, 256, 512, 1024],
        width_multi=0.25,
        **kwargs,
    )


@validate_factory_kwargs("num_blocks", "out_channels_per_stage", "width_multi")
def mobilenet_v1_x0_5(**kwargs):
    return MobileNetV1(
        num_blocks=[2, 2, 6, 2],
        out_channels_per_stage=[128, 256, 512, 1024],
        width_multi=0.5,
        **kwargs,
    )


@validate_factory_kwargs("num_blocks", "out_channels_per_stage", "width_multi")
def mobilenet_v1_x0_75(**kwargs):
    return MobileNetV1(
        num_blocks=[2, 2, 6, 2],
        out_channels_per_stage=[128, 256, 512, 1024],
        width_multi=0.75,
        **kwargs,
    )


@validate_factory_kwargs("num_blocks", "out_channels_per_stage", "width_multi")
def mobilenet_v1_x1_0(**kwargs):
    return MobileNetV1(
        num_blocks=[2, 2, 6, 2],
        out_channels_per_stage=[128, 256, 512, 1024],
        width_multi=1.0,
        **kwargs,
    )
