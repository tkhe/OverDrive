import torch
import torch.nn as nn
from torch.utils import checkpoint

from overdrive.layers import CNNBlock
from overdrive.layers import Conv2d
from overdrive.layers import ShapeSpec
from overdrive.layers import get_activation
from overdrive.layers import weight_init

from .base import Backbone
from .base import validate_factory_kwargs

__all__ = ["RegNet"]


class Stem(CNNBlock):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 32,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
    ):
        super().__init__(in_channels, out_channels, 2)

        self.conv = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, norm=norm, activation=activation)

        weight_init.c2_msra_fill(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, input_width: int, squeeze_width: int):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_width, squeeze_width, 1)
        self.fc2 = nn.Conv2d(squeeze_width, input_width, 1)
        self.activation = get_activation("ReLU")
        self.scale_activation = get_activation("Sigmoid")

        weight_init.c2_msra_fill(self.fc1)
        weight_init.c2_msra_fill(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return scale * x


class Bottleneck(CNNBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        group_width: int = 1,
        bottleneck_multiplier: float = 1.0,
        use_se: bool = False,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
        use_checkpoint: bool = False,
    ):
        super().__init__(in_channels, out_channels, stride)

        bottleneck_channels = int(round(out_channels * bottleneck_multiplier))
        if bottleneck_channels % group_width != 0:
            raise ValueError("RegNet bottleneck channels must be divisible by group width")
        groups = bottleneck_channels // group_width

        self.conv1 = Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, norm=norm, activation=activation)
        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            groups=groups,
            norm=norm,
            activation=activation,
        )
        self.se = SqueezeExcitation(bottleneck_channels, int(round(0.25 * in_channels))) if use_se else None
        self.conv3 = Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, norm=norm)
        self.proj = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, norm=norm) if stride != 1 or in_channels != out_channels else None
        self.activation = get_activation(activation)

        self.use_checkpoint = use_checkpoint

        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)
        if self.proj is not None:
            weight_init.c2_msra_fill(self.proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _inner_forward(x):
            out = self.conv1(x)
            out = self.conv2(out)
            if self.se is not None:
                out = self.se(out)
            out = self.conv3(out)

            if self.proj is not None:
                shortcut = self.proj(x)
            else:
                shortcut = x

            out += shortcut

            if self.activation is not None:
                out = self.activation(out)

            return out

        if self.use_checkpoint:
            out = checkpoint.checkpoint(_inner_forward, x, use_reentrant=False)
        else:
            out = _inner_forward(x)

        return out


class RegNet(Backbone):
    """
    Support RegNet (https://arxiv.org/abs/2003.13678).
    """

    def __init__(
        self,
        depths: list[int],
        widths: list[int],
        group_widths: list[int],
        bottleneck_multipliers: list[float],
        strides: list[int],
        use_se: bool = False,
        input_shape: ShapeSpec | None = None,
        stem_width: int = 32,
        use_checkpoint: bool = False,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
        out_features: list[str] | None = None,
        freeze_at: int = 0,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()

        if input_shape is None:
            input_shape = ShapeSpec(channels=3)

        self.num_classes = num_classes

        if not len(depths) == len(widths) == len(group_widths) == len(bottleneck_multipliers) == len(strides):
            raise ValueError("RegNet stage parameter sequences must have the same length")

        if out_features is not None:
            num_stages = max([{"stage2": 1, "stage3": 2, "stage4": 3, "stage5": 4}.get(feature, 0) for feature in out_features])
            depths = depths[:num_stages]
            widths = widths[:num_stages]
            group_widths = group_widths[:num_stages]
            bottleneck_multipliers = bottleneck_multipliers[:num_stages]
            strides = strides[:num_stages]

        self.stem = Stem(input_shape.channels, stem_width, norm=norm, activation=activation)
        cur_stride = self.stem.stride
        self._out_feature_strides["stem"] = cur_stride
        self._out_feature_channels["stem"] = self.stem.out_channels

        in_channels = self.stem.out_channels
        self.stages = []
        for idx, (num_blocks, out_channels, stride, stage_group_width, stage_bottleneck_multiplier) in enumerate(
            zip(depths, widths, strides, group_widths, bottleneck_multipliers, strict=True),
            start=2,
        ):
            name = f"stage{idx}"
            stage = RegNet.make_stage(
                num_blocks,
                in_channels,
                out_channels,
                stride,
                stage_group_width,
                stage_bottleneck_multiplier,
                use_se,
                norm,
                activation,
                use_checkpoint,
            )
            self.add_module(name, stage)
            self.stages.append(name)
            self._out_feature_channels[name] = out_channels
            cur_stride *= stride
            self._out_feature_strides[name] = cur_stride
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
            "RegNet-X-400MF": regnet_x_400mf,
            "RegNet-X-800MF": regnet_x_800mf,
            "RegNet-X-1.6GF": regnet_x_1_6gf,
            "RegNet-X-3.2GF": regnet_x_3_2gf,
            "RegNet-X-8GF": regnet_x_8gf,
            "RegNet-X-16GF": regnet_x_16gf,
            "RegNet-X-32GF": regnet_x_32gf,
            "RegNet-Y-400MF": regnet_y_400mf,
            "RegNet-Y-800MF": regnet_y_800mf,
            "RegNet-Y-1.6GF": regnet_y_1_6gf,
            "RegNet-Y-3.2GF": regnet_y_3_2gf,
            "RegNet-Y-8GF": regnet_y_8gf,
            "RegNet-Y-16GF": regnet_y_16gf,
            "RegNet-Y-32GF": regnet_y_32gf,
        }

        if name not in predefined:
            raise ValueError(f"`{name}` is not predefined for RegNet. Available names: {', '.join(predefined)}.")

        return predefined[name](**kwargs)

    @staticmethod
    def make_stage(
        num_blocks: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        group_width: int,
        bottleneck_multiplier: float,
        use_se: bool,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
        use_checkpoint: bool = False,
    ) -> nn.Sequential:
        layers = [
            Bottleneck(
                in_channels,
                out_channels,
                stride=stride,
                group_width=group_width,
                bottleneck_multiplier=bottleneck_multiplier,
                use_se=use_se,
                norm=norm,
                activation=activation,
                use_checkpoint=use_checkpoint,
            )
        ]
        for _ in range(1, num_blocks):
            layers.append(
                Bottleneck(
                    out_channels,
                    out_channels,
                    stride=1,
                    group_width=group_width,
                    bottleneck_multiplier=bottleneck_multiplier,
                    use_se=use_se,
                    norm=norm,
                    activation=activation,
                    use_checkpoint=use_checkpoint,
                )
            )
        return nn.Sequential(*layers)

    def freeze(self, freeze_at: int = 0):
        if freeze_at >= 1:
            self.stem.freeze()

        for idx, stage_name in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                stage = getattr(self, stage_name)
                for block in stage.children():
                    block.freeze()

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


_REGNET_FACTORY_KEYS = (
    "depths",
    "widths",
    "group_widths",
    "bottleneck_multipliers",
    "strides",
    "use_se",
    "stem_width",
)


@validate_factory_kwargs(*_REGNET_FACTORY_KEYS)
def regnet_x_400mf(**kwargs):
    return RegNet(
        depths=[1, 2, 7, 12],
        widths=[32, 64, 160, 400],
        group_widths=[16, 16, 16, 16],
        bottleneck_multipliers=[1.0, 1.0, 1.0, 1.0],
        strides=[2, 2, 2, 2],
        use_se=False,
        stem_width=32,
        **kwargs,
    )


@validate_factory_kwargs(*_REGNET_FACTORY_KEYS)
def regnet_x_800mf(**kwargs):
    return RegNet(
        depths=[1, 3, 7, 5],
        widths=[64, 128, 288, 672],
        group_widths=[16, 16, 16, 16],
        bottleneck_multipliers=[1.0, 1.0, 1.0, 1.0],
        strides=[2, 2, 2, 2],
        use_se=False,
        stem_width=32,
        **kwargs,
    )


@validate_factory_kwargs(*_REGNET_FACTORY_KEYS)
def regnet_x_1_6gf(**kwargs):
    return RegNet(
        depths=[2, 4, 10, 2],
        widths=[72, 168, 408, 912],
        group_widths=[24, 24, 24, 24],
        bottleneck_multipliers=[1.0, 1.0, 1.0, 1.0],
        strides=[2, 2, 2, 2],
        use_se=False,
        stem_width=32,
        **kwargs,
    )


@validate_factory_kwargs(*_REGNET_FACTORY_KEYS)
def regnet_x_3_2gf(**kwargs):
    return RegNet(
        depths=[2, 6, 15, 2],
        widths=[96, 192, 432, 1008],
        group_widths=[48, 48, 48, 48],
        bottleneck_multipliers=[1.0, 1.0, 1.0, 1.0],
        strides=[2, 2, 2, 2],
        use_se=False,
        stem_width=32,
        **kwargs,
    )


@validate_factory_kwargs(*_REGNET_FACTORY_KEYS)
def regnet_x_8gf(**kwargs):
    return RegNet(
        depths=[2, 5, 15, 1],
        widths=[80, 240, 720, 1920],
        group_widths=[80, 120, 120, 120],
        bottleneck_multipliers=[1.0, 1.0, 1.0, 1.0],
        strides=[2, 2, 2, 2],
        use_se=False,
        stem_width=32,
        **kwargs,
    )


@validate_factory_kwargs(*_REGNET_FACTORY_KEYS)
def regnet_x_16gf(**kwargs):
    return RegNet(
        depths=[2, 6, 13, 1],
        widths=[256, 512, 896, 2048],
        group_widths=[128, 128, 128, 128],
        bottleneck_multipliers=[1.0, 1.0, 1.0, 1.0],
        strides=[2, 2, 2, 2],
        use_se=False,
        stem_width=32,
        **kwargs,
    )


@validate_factory_kwargs(*_REGNET_FACTORY_KEYS)
def regnet_x_32gf(**kwargs):
    return RegNet(
        depths=[2, 7, 13, 1],
        widths=[336, 672, 1344, 2520],
        group_widths=[168, 168, 168, 168],
        bottleneck_multipliers=[1.0, 1.0, 1.0, 1.0],
        strides=[2, 2, 2, 2],
        use_se=False,
        stem_width=32,
        **kwargs,
    )


@validate_factory_kwargs(*_REGNET_FACTORY_KEYS)
def regnet_y_400mf(**kwargs):
    return RegNet(
        depths=[1, 3, 6, 6],
        widths=[48, 104, 208, 440],
        group_widths=[8, 8, 8, 8],
        bottleneck_multipliers=[1.0, 1.0, 1.0, 1.0],
        strides=[2, 2, 2, 2],
        use_se=True,
        stem_width=32,
        **kwargs,
    )


@validate_factory_kwargs(*_REGNET_FACTORY_KEYS)
def regnet_y_800mf(**kwargs):
    return RegNet(
        depths=[1, 3, 8, 2],
        widths=[64, 144, 320, 784],
        group_widths=[16, 16, 16, 16],
        bottleneck_multipliers=[1.0, 1.0, 1.0, 1.0],
        strides=[2, 2, 2, 2],
        use_se=True,
        stem_width=32,
        **kwargs,
    )


@validate_factory_kwargs(*_REGNET_FACTORY_KEYS)
def regnet_y_1_6gf(**kwargs):
    return RegNet(
        depths=[2, 6, 17, 2],
        widths=[48, 120, 336, 888],
        group_widths=[24, 24, 24, 24],
        bottleneck_multipliers=[1.0, 1.0, 1.0, 1.0],
        strides=[2, 2, 2, 2],
        use_se=True,
        stem_width=32,
        **kwargs,
    )


@validate_factory_kwargs(*_REGNET_FACTORY_KEYS)
def regnet_y_3_2gf(**kwargs):
    return RegNet(
        depths=[2, 5, 13, 1],
        widths=[72, 216, 576, 1512],
        group_widths=[24, 24, 24, 24],
        bottleneck_multipliers=[1.0, 1.0, 1.0, 1.0],
        strides=[2, 2, 2, 2],
        use_se=True,
        stem_width=32,
        **kwargs,
    )


@validate_factory_kwargs(*_REGNET_FACTORY_KEYS)
def regnet_y_8gf(**kwargs):
    return RegNet(
        depths=[2, 4, 10, 1],
        widths=[224, 448, 896, 2016],
        group_widths=[56, 56, 56, 56],
        bottleneck_multipliers=[1.0, 1.0, 1.0, 1.0],
        strides=[2, 2, 2, 2],
        use_se=True,
        stem_width=32,
        **kwargs,
    )


@validate_factory_kwargs(*_REGNET_FACTORY_KEYS)
def regnet_y_16gf(**kwargs):
    return RegNet(
        depths=[2, 4, 11, 1],
        widths=[224, 448, 1232, 3024],
        group_widths=[112, 112, 112, 112],
        bottleneck_multipliers=[1.0, 1.0, 1.0, 1.0],
        strides=[2, 2, 2, 2],
        use_se=True,
        stem_width=32,
        **kwargs,
    )


@validate_factory_kwargs(*_REGNET_FACTORY_KEYS)
def regnet_y_32gf(**kwargs):
    return RegNet(
        depths=[2, 5, 12, 1],
        widths=[232, 696, 1392, 3712],
        group_widths=[232, 232, 232, 232],
        bottleneck_multipliers=[1.0, 1.0, 1.0, 1.0],
        strides=[2, 2, 2, 2],
        use_se=True,
        stem_width=32,
        **kwargs,
    )
