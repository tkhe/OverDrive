import torch
import torch.nn as nn
import torch.nn.functional as F

from overdrive.layers import Conv2d
from overdrive.layers import DropPath
from overdrive.layers import FreezableMixin
from overdrive.layers import ShapeSpec

from .base import Backbone
from .base import validate_factory_kwargs

__all__ = ["HGNetV2"]


class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value: float = 1.0, bias_value: float = 0.0):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor([scale_value]))
        self.bias = nn.Parameter(torch.tensor([bias_value]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.bias


class HGNetV2LightBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_lab: bool = False,
        norm: str | dict | nn.Module | None = "BN",
        activation: str | dict | nn.Module | None = "ReLU",
    ):
        super().__init__()

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=norm,
            activation=None,
        )
        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            bias=False,
            norm=norm,
            activation=activation,
        )
        self.lab = LearnableAffineBlock() if use_lab else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        if self.lab is not None:
            x = self.lab(x)
        return x


class StemV2(nn.Module, FreezableMixin):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        use_lab: bool = False,
        norm: str | dict | nn.Module | None = "BN",
        activation: str | dict | nn.Module | None = "ReLU",
    ):
        super().__init__()

        self.stem1 = Conv2d(
            in_channels,
            mid_channels,
            kernel_size=3,
            stride=2,
            bias=False,
            norm=norm,
            activation=activation,
        )
        self.stem1_lab = LearnableAffineBlock() if use_lab else None
        self.stem2a = Conv2d(
            mid_channels,
            mid_channels // 2,
            kernel_size=2,
            padding=0,
            bias=False,
            norm=norm,
            activation=activation,
        )
        self.stem2a_lab = LearnableAffineBlock() if use_lab else None
        self.stem2b = Conv2d(
            mid_channels // 2,
            mid_channels,
            kernel_size=2,
            padding=0,
            bias=False,
            norm=norm,
            activation=activation,
        )
        self.stem2b_lab = LearnableAffineBlock() if use_lab else None
        self.stem3 = Conv2d(
            mid_channels * 2,
            mid_channels,
            kernel_size=3,
            stride=2,
            bias=False,
            norm=norm,
            activation=activation,
        )
        self.stem3_lab = LearnableAffineBlock() if use_lab else None
        self.stem4 = Conv2d(
            mid_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=norm,
            activation=activation,
        )
        self.stem4_lab = LearnableAffineBlock() if use_lab else None
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem1(x)
        if self.stem1_lab is not None:
            x = self.stem1_lab(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        if self.stem2a_lab is not None:
            x2 = self.stem2a_lab(x2)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        if self.stem2b_lab is not None:
            x2 = self.stem2b_lab(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        if self.stem3_lab is not None:
            x = self.stem3_lab(x)
        x = self.stem4(x)
        if self.stem4_lab is not None:
            x = self.stem4_lab(x)
        return x


class HGNetV2Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        num_layers: int,
        kernel_size: int = 3,
        residual: bool = False,
        light_block: bool = False,
        use_lab: bool = False,
        drop_path: float = 0.0,
        norm: str | dict | nn.Module | None = "BN",
        activation: str | dict | nn.Module | None = "ReLU",
    ):
        super().__init__()

        self.residual = residual
        layers = []
        labs = []
        for index in range(num_layers):
            layer_in_channels = in_channels if index == 0 else mid_channels
            if light_block:
                layer = HGNetV2LightBlock(
                    layer_in_channels,
                    mid_channels,
                    kernel_size=kernel_size,
                    use_lab=use_lab,
                    norm=norm,
                    activation=activation,
                )
                layers.append(layer)
            else:
                layer = Conv2d(
                    layer_in_channels,
                    mid_channels,
                    kernel_size=kernel_size,
                    bias=False,
                    norm=norm,
                    activation=activation,
                )
                layers.append(layer)
                if use_lab:
                    labs.append(LearnableAffineBlock())
        self.layers = nn.ModuleList(layers)
        self.labs = nn.ModuleList(labs) if labs else None

        total_channels = in_channels + num_layers * mid_channels
        self.aggregation = nn.Sequential(
            Conv2d(total_channels, out_channels // 2, kernel_size=1, bias=False, norm=norm, activation=activation),
            Conv2d(out_channels // 2, out_channels, kernel_size=1, bias=False, norm=norm, activation=activation),
        )
        self.aggregation_labs = nn.ModuleList([LearnableAffineBlock(), LearnableAffineBlock()]) if use_lab else None
        self.drop_path = DropPath(drop_path) if drop_path else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        output = [x]
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if self.labs is not None:
                x = self.labs[index](x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation[0](x)
        if self.aggregation_labs is not None:
            x = self.aggregation_labs[0](x)
        x = self.aggregation[1](x)
        if self.aggregation_labs is not None:
            x = self.aggregation_labs[1](x)
        if self.residual:
            if self.drop_path is not None:
                x = self.drop_path(x)
            x = x + identity
        return x


class HGNetV2Stage(nn.Module, FreezableMixin):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        num_blocks: int,
        num_layers: int,
        downsample: bool = True,
        stride: int = 2,
        light_block: bool = False,
        kernel_size: int = 3,
        use_lab: bool = False,
        drop_path: list[float] | float = 0.0,
        norm: str | dict | nn.Module | None = "BN",
        activation: str | dict | nn.Module | None = "ReLU",
    ):
        super().__init__()

        self.downsample = None
        if downsample:
            self.downsample = Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                groups=in_channels,
                bias=False,
                norm=norm,
            )

        blocks = []
        for index in range(num_blocks):
            rate = drop_path[index] if isinstance(drop_path, (list, tuple)) else drop_path
            block = HGNetV2Block(
                in_channels if index == 0 else out_channels,
                mid_channels,
                out_channels,
                num_layers,
                residual=index > 0,
                light_block=light_block,
                kernel_size=kernel_size,
                use_lab=use_lab,
                drop_path=rate,
                norm=norm,
                activation=activation,
            )
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        return self.blocks(x)


class HGNetV2(Backbone):
    """
    support PP-HGNet V2.
    """

    def __init__(
        self,
        stem_channels: tuple[int, int],
        stage_configs: tuple[tuple[int, int, int, int, bool, bool, int, int], ...],
        input_shape: ShapeSpec | None = None,
        norm: str | dict | nn.Module | None = "BN",
        activation: str | dict | nn.Module | None = "ReLU",
        use_lab: bool = False,
        head_hidden_size: int = 2048,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        out_features: list[str] | None = None,
        freeze_at: int = 0,
        num_classes: int | None = None,
    ):
        super().__init__()

        if input_shape is None:
            input_shape = ShapeSpec(channels=3)

        block_depths = [stage_config[3] for stage_config in stage_configs]
        drop_path_rates = [rates.tolist() for rates in torch.linspace(0.0, drop_path_rate, sum(block_depths)).split(block_depths)]
        num_stages = len(stage_configs)
        if out_features is not None:
            num_stages = max([{"stem": 0, "stage2": 1, "stage3": 2, "stage4": 3, "stage5": 4}.get(name, 0) for name in out_features], default=0)
            stage_configs = stage_configs[:num_stages]
            drop_path_rates = drop_path_rates[:num_stages]

        self.num_classes = num_classes
        self.use_lab = use_lab
        self.stem = StemV2(
            input_shape.channels,
            stem_channels[0],
            stem_channels[1],
            use_lab=use_lab,
            norm=norm,
            activation=activation,
        )
        self._out_feature_channels["stem"] = stem_channels[1]
        self._out_feature_strides["stem"] = 4

        current_stride = 4
        in_channels = stem_channels[1]
        self.stages: list[str] = []
        for index, stage_config in enumerate(stage_configs, start=2):
            stage_in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, num_layers = stage_config
            stage_name = f"stage{index}"
            stage = HGNetV2Stage(
                in_channels=stage_in_channels,
                mid_channels=mid_channels,
                out_channels=out_channels,
                num_blocks=num_blocks,
                num_layers=num_layers,
                downsample=downsample,
                light_block=light_block,
                kernel_size=kernel_size,
                use_lab=use_lab,
                drop_path=drop_path_rates[index - 2],
                norm=norm,
                activation=activation,
            )
            self.add_module(stage_name, stage)
            self.stages.append(stage_name)
            self._out_feature_channels[stage_name] = out_channels
            if downsample:
                current_stride *= 2
            self._out_feature_strides[stage_name] = current_stride
            in_channels = out_channels

        last_channels = in_channels
        if num_classes is not None:
            head = [
                nn.AdaptiveAvgPool2d(1),
                Conv2d(last_channels, head_hidden_size, kernel_size=1, bias=False, activation=activation),
            ]
            if use_lab:
                head.append(LearnableAffineBlock())
            head.append(nn.Dropout(drop_rate))
            head.append(nn.Flatten(1))
            head.append(nn.Linear(head_hidden_size, num_classes))
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
    def build(name: str, **kwargs):
        predefined = {
            "HGNet-V2-B0": hgnetv2_b0,
            "HGNet-V2-B1": hgnetv2_b1,
            "HGNet-V2-B2": hgnetv2_b2,
            "HGNet-V2-B3": hgnetv2_b3,
            "HGNet-V2-B4": hgnetv2_b4,
            "HGNet-V2-B5": hgnetv2_b5,
            "HGNet-V2-B6": hgnetv2_b6,
        }

        if name not in predefined:
            raise ValueError(f"`{name}` is not predefined for HGNetV2. Available names: {', '.join(predefined)}.")
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


@validate_factory_kwargs("stem_channels", "stage_configs", "use_lab")
def hgnetv2_b0(**kwargs):
    return HGNetV2(
        stem_channels=(16, 16),
        stage_configs=(
            (16, 16, 64, 1, False, False, 3, 3),
            (64, 32, 256, 1, True, False, 3, 3),
            (256, 64, 512, 2, True, True, 5, 3),
            (512, 128, 1024, 1, True, True, 5, 3),
        ),
        use_lab=True,
        **kwargs,
    )


@validate_factory_kwargs("stem_channels", "stage_configs", "use_lab")
def hgnetv2_b1(**kwargs):
    return HGNetV2(
        stem_channels=(24, 32),
        stage_configs=(
            (32, 32, 64, 1, False, False, 3, 3),
            (64, 48, 256, 1, True, False, 3, 3),
            (256, 96, 512, 2, True, True, 5, 3),
            (512, 192, 1024, 1, True, True, 5, 3),
        ),
        use_lab=True,
        **kwargs,
    )


@validate_factory_kwargs("stem_channels", "stage_configs", "use_lab")
def hgnetv2_b2(**kwargs):
    return HGNetV2(
        stem_channels=(24, 32),
        stage_configs=(
            (32, 32, 96, 1, False, False, 3, 4),
            (96, 64, 384, 1, True, False, 3, 4),
            (384, 128, 768, 3, True, True, 5, 4),
            (768, 256, 1536, 1, True, True, 5, 4),
        ),
        use_lab=True,
        **kwargs,
    )


@validate_factory_kwargs("stem_channels", "stage_configs", "use_lab")
def hgnetv2_b3(**kwargs):
    return HGNetV2(
        stem_channels=(24, 32),
        stage_configs=(
            (32, 32, 128, 1, False, False, 3, 5),
            (128, 64, 512, 1, True, False, 3, 5),
            (512, 128, 1024, 3, True, True, 5, 5),
            (1024, 256, 2048, 1, True, True, 5, 5),
        ),
        use_lab=True,
        **kwargs,
    )


@validate_factory_kwargs("stem_channels", "stage_configs")
def hgnetv2_b4(**kwargs):
    return HGNetV2(
        stem_channels=(32, 48),
        stage_configs=(
            (48, 48, 128, 1, False, False, 3, 6),
            (128, 96, 512, 1, True, False, 3, 6),
            (512, 192, 1024, 3, True, True, 5, 6),
            (1024, 384, 2048, 1, True, True, 5, 6),
        ),
        **kwargs,
    )


@validate_factory_kwargs("stem_channels", "stage_configs")
def hgnetv2_b5(**kwargs):
    return HGNetV2(
        stem_channels=(32, 64),
        stage_configs=(
            (64, 64, 128, 1, False, False, 3, 6),
            (128, 128, 512, 2, True, False, 3, 6),
            (512, 256, 1024, 5, True, True, 5, 6),
            (1024, 512, 2048, 2, True, True, 5, 6),
        ),
        **kwargs,
    )


@validate_factory_kwargs("stem_channels", "stage_configs")
def hgnetv2_b6(**kwargs):
    return HGNetV2(
        stem_channels=(48, 96),
        stage_configs=(
            (96, 96, 192, 2, False, False, 3, 6),
            (192, 192, 512, 3, True, False, 3, 6),
            (512, 384, 1024, 6, True, True, 5, 6),
            (1024, 768, 2048, 3, True, True, 5, 6),
        ),
        **kwargs,
    )
