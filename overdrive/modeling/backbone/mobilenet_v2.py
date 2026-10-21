import torch
import torch.nn as nn
import torch.nn.functional as F

from overdrive.layers import CNNBlock
from overdrive.layers import Conv2d
from overdrive.layers import ShapeSpec
from overdrive.layers import make_divisible
from overdrive.layers import weight_init

from .base import Backbone
from .base import validate_factory_kwargs

__all__ = ["MobileNetV2"]

# Default MobileNet v2 block configurations: [expansion, channels, num_blocks, stride]
MOBILENET_V2_ARCH = [
    [[1, 16, 1, 1], [6, 24, 2, 2]],
    [[6, 32, 3, 2]],
    [[6, 64, 4, 2], [6, 96, 3, 1]],
    [[6, 160, 3, 2], [6, 320, 1, 1]],
]


class Stem(CNNBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU6",
    ):
        super().__init__(in_channels, out_channels, 2)

        self.conv = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, norm=norm, activation=activation)

        weight_init.c2_msra_fill(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class InvertedResidual(CNNBlock):
    """
    MobileNet v2 inverted residual block with linear bottleneck.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 1,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU6",
    ):
        super().__init__(in_channels, out_channels, stride)

        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = make_divisible(in_channels * expand_ratio, 8)

        layers = []
        # Expansion phase (pointwise conv)
        if expand_ratio != 1:
            layers.append(Conv2d(in_channels, hidden_dim, kernel_size=1, norm=norm, activation=activation))
        # Depthwise conv
        layers.append(Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim, norm=norm, activation=activation))
        # Projection phase (pointwise linear)
        layers.append(Conv2d(hidden_dim, out_channels, kernel_size=1, norm=norm))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(Backbone):
    """
    support MobileNet v2 (https://arxiv.org/abs/1801.04381).
    """

    def __init__(
        self,
        arch: list[list] = MOBILENET_V2_ARCH,
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
            num_stages = max([{"stage2": 1, "stage3": 2, "stage4": 3, "stage5": 4, "conv5": 4}.get(f, 0) for f in out_features])
            arch = arch[:num_stages]

        # Build stem
        stem_channels = make_divisible(32 * width_multi, 8)
        self.stem = Stem(input_shape.channels, stem_channels, norm, activation)
        current_stride = self.stem.stride
        self._out_feature_strides["stem"] = current_stride
        self._out_feature_channels["stem"] = stem_channels
        in_channels = stem_channels

        # Build stages
        self.stages = []
        for idx, stage_cfg in enumerate(arch):
            name = f"stage{idx + 2}"
            blocks = []
            for t, c, n, s in stage_cfg:
                out_channels = make_divisible(c * width_multi, 8)
                for i in range(n):
                    stride = s if i == 0 else 1
                    blocks.append(InvertedResidual(in_channels, out_channels, stride=stride, expand_ratio=t, norm=norm, activation=activation))
                    if stride > 1:
                        current_stride *= stride
                    in_channels = out_channels

            self.add_module(name, nn.Sequential(*blocks))
            self.stages.append(name)
            self._out_feature_channels[name] = in_channels
            self._out_feature_strides[name] = current_stride

        if out_features is not None and "conv5" in out_features:
            conv5_channels = make_divisible(1280 * max(1.0, width_multi), 8)
            conv5 = Conv2d(in_channels, conv5_channels, kernel_size=1, norm=norm, activation=activation)
            self.add_module("conv5", conv5)
            self.stages.append("conv5")
            self._out_feature_channels["conv5"] = conv5_channels
            self._out_feature_strides["conv5"] = current_stride
            in_channels = conv5_channels
            name = "conv5"

        if num_classes is not None:
            self.linear = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_channels, num_classes),
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
            "MobileNet-V2-x0.25": mobilenet_v2_x0_25,
            "MobileNet-V2-x0.5": mobilenet_v2_x0_5,
            "MobileNet-V2-x0.75": mobilenet_v2_x0_75,
            "MobileNet-V2-x1.0": mobilenet_v2_x1_0,
        }

        if name not in predefined:
            raise ValueError(f"`{name}` is not predefined for MobileNetV2. Available names: {', '.join(predefined)}.")

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
            x = F.adaptive_avg_pool2d(x, (1, 1))
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
def mobilenet_v2_x0_25(**kwargs):
    return MobileNetV2(arch=MOBILENET_V2_ARCH, width_multi=0.25, **kwargs)


@validate_factory_kwargs("arch", "width_multi")
def mobilenet_v2_x0_5(**kwargs):
    return MobileNetV2(arch=MOBILENET_V2_ARCH, width_multi=0.5, **kwargs)


@validate_factory_kwargs("arch", "width_multi")
def mobilenet_v2_x0_75(**kwargs):
    return MobileNetV2(arch=MOBILENET_V2_ARCH, width_multi=0.75, **kwargs)


@validate_factory_kwargs("arch", "width_multi")
def mobilenet_v2_x1_0(**kwargs):
    return MobileNetV2(arch=MOBILENET_V2_ARCH, width_multi=1.0, **kwargs)
