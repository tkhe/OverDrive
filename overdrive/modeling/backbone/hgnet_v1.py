import torch
import torch.nn as nn

from overdrive.layers import Conv2d
from overdrive.layers import DropPath
from overdrive.layers import FreezableMixin
from overdrive.layers import ShapeSpec
from overdrive.layers import get_activation

from .base import Backbone
from .base import validate_factory_kwargs

__all__ = ["HGNetV1"]


class ESEModule(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv = Conv2d(channels, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return identity * x


class Stem(nn.Module, FreezableMixin):
    def __init__(
        self,
        in_channels: int,
        stem_channels: tuple[int, int, int],
        norm: str | dict | nn.Module = "BN",
        activation: str | dict | nn.Module = "ReLU",
    ):
        super().__init__()

        channels = (in_channels, *stem_channels)
        stem = [Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=2 if i == 0 else 1, norm=norm, activation=activation) for i in range(len(channels) - 1)]
        self.stem = nn.Sequential(*stem)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.pool(x)
        return x


class HGNetV1Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        num_layers: int,
        residual: bool = False,
        kernel_size: int = 3,
        norm: str | dict | nn.Module = "BN",
        activation: str | dict | nn.Module = "ReLU",
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.residual = residual
        layers = [Conv2d(in_channels if i == 0 else mid_channels, mid_channels, kernel_size=kernel_size, norm=norm, activation=activation) for i in range(num_layers)]
        self.layers = nn.ModuleList(layers)
        self.aggregation = nn.Sequential(
            Conv2d(in_channels + num_layers * mid_channels, out_channels, kernel_size=1, norm=norm, activation=activation),
            ESEModule(out_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        outputs = [x]
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        x = torch.cat(outputs, dim=1)
        x = self.aggregation(x)
        if self.residual:
            if self.drop_path is not None:
                x = self.drop_path(x)
            x = x + identity
        return x


class HGNetV1Stage(nn.Module, FreezableMixin):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        num_blocks: int,
        num_layers: int,
        downsample: bool = True,
        stride: int = 2,
        kernel_size: int = 3,
        norm: str | dict | nn.Module = "BN",
        activation: str | dict | nn.Module = "ReLU",
        drop_path: list[float] | float = 0.0,
    ):
        super().__init__()

        if downsample:
            self.downsample = Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                groups=in_channels,
                norm=norm,
            )
        else:
            self.downsample = None

        blocks = []
        for i in range(num_blocks):
            block = HGNetV1Block(
                in_channels if i == 0 else out_channels,
                mid_channels,
                out_channels,
                num_layers=num_layers,
                residual=i > 0,
                kernel_size=kernel_size,
                norm=norm,
                activation=activation,
                drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
            )
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        x = self.blocks(x)
        return x


class HGNetV1(Backbone):
    """
    support PP-HGNet V1.
    """

    def __init__(
        self,
        stem_channels: tuple[int, int, int],
        mid_channels: tuple[int, ...],
        out_channels: tuple[int, ...],
        num_blocks: tuple[int, ...],
        num_layers: tuple[int, ...],
        downsample: tuple[bool, ...] = (False, True, True, True),
        kernel_sizes: tuple[int, ...] = (3, 3, 3, 3),
        input_shape: ShapeSpec | None = None,
        norm: str | dict | nn.Module = "BN",
        activation: str | dict | nn.Module = "ReLU",
        drop_path_rate: float = 0.0,
        drop_rate: float = 0.0,
        head_hidden_size: int = 2048,
        out_features: list[str] | None = None,
        freeze_at: int = 0,
        num_classes: int | None = None,
    ):
        super().__init__()

        if input_shape is None:
            input_shape = ShapeSpec(channels=3)

        num_stages = len(out_channels)
        drop_path_rates = [x.tolist() for x in torch.linspace(0.0, drop_path_rate, sum(num_blocks)).split(num_blocks)]
        if out_features is not None:
            num_stages = max([{"stage2": 1, "stage3": 2, "stage4": 3, "stage5": 4}.get(f, 0) for f in out_features], default=0)
            mid_channels = mid_channels[:num_stages]
            out_channels = out_channels[:num_stages]
            num_blocks = num_blocks[:num_stages]
            num_layers = num_layers[:num_stages]
            downsample = downsample[:num_stages]
            kernel_sizes = kernel_sizes[:num_stages]
            drop_path_rates = drop_path_rates[:num_stages]

        self.num_classes = num_classes
        self.stem = Stem(
            input_shape.channels,
            stem_channels=stem_channels,
            norm=norm,
            activation=activation,
        )
        self._out_feature_channels["stem"] = stem_channels[-1]
        self._out_feature_strides["stem"] = 4

        current_stride = 4
        in_channels = stem_channels[-1]
        self.stages: list[str] = []
        for index in range(num_stages):
            stage_name = f"stage{index + 2}"
            stage = HGNetV1Stage(
                in_channels=in_channels,
                mid_channels=mid_channels[index],
                out_channels=out_channels[index],
                num_blocks=num_blocks[index],
                num_layers=num_layers[index],
                downsample=downsample[index],
                kernel_size=kernel_sizes[index],
                norm=norm,
                activation=activation,
                drop_path=drop_path_rates[index],
            )
            self.add_module(stage_name, stage)
            self.stages.append(stage_name)
            self._out_feature_channels[stage_name] = out_channels[index]
            if downsample[index]:
                current_stride *= 2
            self._out_feature_strides[stage_name] = current_stride
            in_channels = out_channels[index]

        if num_classes is not None:
            head = [
                nn.AdaptiveAvgPool2d(1),
                Conv2d(in_channels, head_hidden_size, kernel_size=1, bias=False),
            ]
            head_activation = get_activation(activation)
            if head_activation is not None:
                head.append(head_activation)
            head.extend([nn.Dropout(drop_rate), nn.Flatten(1), nn.Linear(head_hidden_size, num_classes)])
            self.head = nn.Sequential(*head)

        if out_features is None:
            out_features = [self.stages[-1]]

        self._out_features = out_features
        if len(self._out_features) == 0:
            raise ValueError("out_features must contain at least one feature name.")

        children = [name for name, _ in self.named_children()]
        for out_feature in self._out_features:
            if out_feature not in children:
                raise ValueError(f"Available children: {', '.join(children)}")

        self.freeze(freeze_at)

    @staticmethod
    def build(name, **kwargs):
        predefined = {
            "HGNet-V1-Tiny": hgnetv1_tiny,
            "HGNet-V1-Small": hgnetv1_small,
            "HGNet-V1-Base": hgnetv1_base,
        }

        if name not in predefined:
            raise ValueError(f"`{name}` is not predefined for HGNetV1. Available names: {', '.join(predefined)}.")
        return predefined[name](**kwargs)

    def freeze(self, freeze_at: int = 0):
        if freeze_at >= 1:
            self.stem.freeze()

        for index, stage_name in enumerate(self.stages, start=2):
            if freeze_at >= index:
                getattr(self, stage_name).freeze()

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
            outputs["linear"] = self.head(x)

        return outputs


@validate_factory_kwargs("stem_channels", "mid_channels", "out_channels", "num_blocks", "num_layers")
def hgnetv1_tiny(**kwargs):
    return HGNetV1(
        stem_channels=(48, 48, 96),
        mid_channels=(96, 128, 160, 192),
        out_channels=(224, 448, 512, 768),
        num_blocks=(1, 1, 2, 1),
        num_layers=(5, 5, 5, 5),
        **kwargs,
    )


@validate_factory_kwargs("stem_channels", "mid_channels", "out_channels", "num_blocks", "num_layers")
def hgnetv1_small(**kwargs):
    return HGNetV1(
        stem_channels=(64, 64, 128),
        mid_channels=(128, 160, 192, 224),
        out_channels=(256, 512, 768, 1024),
        num_blocks=(1, 1, 2, 1),
        num_layers=(6, 6, 6, 6),
        **kwargs,
    )


@validate_factory_kwargs("stem_channels", "mid_channels", "out_channels", "num_blocks", "num_layers")
def hgnetv1_base(**kwargs):
    return HGNetV1(
        stem_channels=(96, 96, 160),
        mid_channels=(192, 224, 256, 288),
        out_channels=(320, 640, 960, 1280),
        num_blocks=(1, 2, 3, 2),
        num_layers=(7, 7, 7, 7),
        **kwargs,
    )
