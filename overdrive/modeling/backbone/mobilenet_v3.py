import torch
import torch.nn as nn

from overdrive.layers import CNNBlock
from overdrive.layers import Conv2d
from overdrive.layers import ShapeSpec
from overdrive.layers import get_activation
from overdrive.layers import make_divisible
from overdrive.layers import weight_init

from .base import Backbone
from .base import validate_factory_kwargs

__all__ = ["MobileNetV3"]

# Default MobileNet v3 Large block configurations:
# [kernel_size, expansion, channels, use_se, activation, stride]
MOBILENET_V3_LARGE_ARCH = [
    [[3, 1, 16, False, "ReLU", 1], [3, 4, 24, False, "ReLU", 2], [3, 3, 24, False, "ReLU", 1]],
    [[5, 3, 40, True, "ReLU", 2], [5, 3, 40, True, "ReLU", 1], [5, 3, 40, True, "ReLU", 1]],
    [[3, 6, 80, False, "HardSwish", 2], [3, 2.5, 80, False, "HardSwish", 1], [3, 2.3, 80, False, "HardSwish", 1], [3, 2.3, 80, False, "HardSwish", 1], [3, 6, 112, True, "HardSwish", 1], [3, 6, 112, True, "HardSwish", 1]],
    [[5, 6, 160, True, "HardSwish", 2], [5, 6, 160, True, "HardSwish", 1], [5, 6, 160, True, "HardSwish", 1]],
]

# Default MobileNet v3 Small block configurations:
# [kernel_size, expansion, channels, use_se, activation, stride]
MOBILENET_V3_SMALL_ARCH = [
    [[3, 1, 16, True, "ReLU", 2]],
    [[3, 4.5, 24, False, "ReLU", 2], [3, 3.67, 24, False, "ReLU", 1]],
    [[5, 4, 40, True, "HardSwish", 2], [5, 6, 40, True, "HardSwish", 1], [5, 6, 40, True, "HardSwish", 1], [5, 3, 48, True, "HardSwish", 1], [5, 3, 48, True, "HardSwish", 1]],
    [[5, 6, 96, True, "HardSwish", 2], [5, 6, 96, True, "HardSwish", 1], [5, 6, 96, True, "HardSwish", 1]],
]


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_channels: int,
        divisor: int = 8,
        activation: str | dict = "ReLU",
        gate_activation: str | dict = "HardSigmoid",
    ):
        super().__init__()

        reduced_channels = make_divisible(in_channels // 4, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_channels, reduced_channels, 1, bias=True)
        self.activation = get_activation(activation)
        self.conv_expand = nn.Conv2d(reduced_channels, in_channels, 1, bias=True)
        self.gate_activation = get_activation(gate_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.activation(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate_activation(x_se)


class Stem(CNNBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str | dict = "BN",
        activation: str | dict = "HardSwish",
    ):
        super().__init__(in_channels, out_channels, 2)

        self.conv = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, norm=norm, activation=activation)

        weight_init.c2_msra_fill(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class InvertedResidual(CNNBlock):
    """
    MobileNet v3 inverted residual block with optional SE module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_se: bool = False,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
    ):
        super().__init__(in_channels, out_channels, stride)

        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        # Expansion phase (pointwise conv)
        if mid_channels != in_channels:
            layers.append(Conv2d(in_channels, mid_channels, kernel_size=1, norm=norm, activation=activation))
        # Depthwise conv
        layers.append(Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, groups=mid_channels, norm=norm, activation=activation))
        self.conv = nn.Sequential(*layers)

        self.se = SqueezeExcite(mid_channels) if use_se else None

        # Projection phase (pointwise linear)
        self.project = Conv2d(mid_channels, out_channels, kernel_size=1, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv(x)
        if self.se is not None:
            out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = out + residual
        return out


class MobileNetV3(Backbone):
    """
    support MobileNet v3 (https://arxiv.org/abs/1905.02244).
    """

    def __init__(
        self,
        arch: list[list] = MOBILENET_V3_LARGE_ARCH,
        width_multi: float = 1.0,
        input_shape: ShapeSpec | None = None,
        norm: str | dict = {"name": "BN", "eps": 0.001, "momentum": 0.01},
        out_features: list[str] | None = None,
        freeze_at: int = 0,
        num_classes: int | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        if input_shape is None:
            input_shape = ShapeSpec(channels=3)

        self.num_classes = num_classes
        last_channel = 1024 if arch == MOBILENET_V3_SMALL_ARCH else 1280

        if out_features is not None:
            num_stages = max([{"stage2": 1, "stage3": 2, "stage4": 3, "stage5": 4, "conv5": 4}.get(f, 0) for f in out_features])
            arch = arch[:num_stages]

        stem_channels = make_divisible(16 * width_multi, 8)
        self.stem = Stem(input_shape.channels, stem_channels, norm)
        current_stride = self.stem.stride
        self._out_feature_strides["stem"] = current_stride
        self._out_feature_channels["stem"] = stem_channels
        in_channels = stem_channels

        self.stages = []
        for idx, stage_cfg in enumerate(arch):
            name = f"stage{idx + 2}"
            blocks = []
            for k, t, c, use_se, act, s in stage_cfg:
                out_channels = make_divisible(c * width_multi, 8)
                mid_channels = make_divisible(in_channels * t, 8)
                blocks.append(InvertedResidual(in_channels, out_channels, mid_channels, kernel_size=k, stride=s, use_se=use_se, norm=norm, activation=act))
                current_stride *= s
                in_channels = out_channels

            self.add_module(name, nn.Sequential(*blocks))
            self.stages.append(name)
            self._out_feature_channels[name] = in_channels
            self._out_feature_strides[name] = current_stride

        if out_features is not None and "conv5" in out_features:
            conv5_channels = make_divisible(in_channels * 6, 8)
            conv5 = Conv2d(in_channels, conv5_channels, kernel_size=1, norm=norm, activation="HardSwish")
            self.add_module("conv5", conv5)
            self.stages.append("conv5")
            self._out_feature_channels["conv5"] = conv5_channels
            self._out_feature_strides["conv5"] = current_stride
            in_channels = conv5_channels
            name = "conv5"

        if num_classes is not None:
            classifier_channels = make_divisible(last_channel * width_multi, 8)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.linear = nn.Sequential(
                nn.Linear(in_channels, classifier_channels),
                get_activation("HardSwish"),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(classifier_channels, num_classes),
            )
            weight_init.normal_fill(self.linear[0], std=0.01)
            weight_init.normal_fill(self.linear[3], std=0.01)

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
            "MobileNet-V3-Large-x0.25": mobilenet_v3_large_x0_25,
            "MobileNet-V3-Large-x0.5": mobilenet_v3_large_x0_5,
            "MobileNet-V3-Large-x0.75": mobilenet_v3_large_x0_75,
            "MobileNet-V3-Large-x1.0": mobilenet_v3_large_x1_0,
            "MobileNet-V3-Small-x0.25": mobilenet_v3_small_x0_25,
            "MobileNet-V3-Small-x0.5": mobilenet_v3_small_x0_5,
            "MobileNet-V3-Small-x0.75": mobilenet_v3_small_x0_75,
            "MobileNet-V3-Small-x1.0": mobilenet_v3_small_x1_0,
        }

        if name not in predefined:
            raise ValueError(f"`{name}` is not predefined for MobileNetV3. Available names: {', '.join(predefined)}.")

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
                module = getattr(self, stage_name)
                if stage_name == "conv5":
                    module.freeze()
                else:
                    for block in module.children():
                        block.freeze()


@validate_factory_kwargs("arch", "width_multi")
def mobilenet_v3_large_x0_25(**kwargs):
    return MobileNetV3(arch=MOBILENET_V3_LARGE_ARCH, width_multi=0.25, **kwargs)


@validate_factory_kwargs("arch", "width_multi")
def mobilenet_v3_large_x1_0(**kwargs):
    return MobileNetV3(arch=MOBILENET_V3_LARGE_ARCH, width_multi=1.0, **kwargs)


@validate_factory_kwargs("arch", "width_multi")
def mobilenet_v3_large_x0_75(**kwargs):
    return MobileNetV3(arch=MOBILENET_V3_LARGE_ARCH, width_multi=0.75, **kwargs)


@validate_factory_kwargs("arch", "width_multi")
def mobilenet_v3_large_x0_5(**kwargs):
    return MobileNetV3(arch=MOBILENET_V3_LARGE_ARCH, width_multi=0.5, **kwargs)


@validate_factory_kwargs("arch", "width_multi")
def mobilenet_v3_small_x1_0(**kwargs):
    return MobileNetV3(arch=MOBILENET_V3_SMALL_ARCH, width_multi=1.0, **kwargs)


@validate_factory_kwargs("arch", "width_multi")
def mobilenet_v3_small_x0_75(**kwargs):
    return MobileNetV3(arch=MOBILENET_V3_SMALL_ARCH, width_multi=0.75, **kwargs)


@validate_factory_kwargs("arch", "width_multi")
def mobilenet_v3_small_x0_5(**kwargs):
    return MobileNetV3(arch=MOBILENET_V3_SMALL_ARCH, width_multi=0.5, **kwargs)


@validate_factory_kwargs("arch", "width_multi")
def mobilenet_v3_small_x0_25(**kwargs):
    return MobileNetV3(arch=MOBILENET_V3_SMALL_ARCH, width_multi=0.25, **kwargs)
