import torch
import torch.nn as nn
from torch.utils import checkpoint

from overdrive.layers import CNNBlock
from overdrive.layers import DropPath
from overdrive.layers import ShapeSpec
from overdrive.layers import get_activation
from overdrive.layers import get_norm
from overdrive.layers import weight_init

from .base import Backbone
from .base import validate_factory_kwargs

__all__ = ["ConvNeXt"]


class Stem(CNNBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str | dict | nn.Module = {"name": "LN2d", "eps": 1e-6},
    ):
        super().__init__(in_channels, out_channels, 4)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=4,
            bias=True,
        )
        self.norm = get_norm(norm, out_channels)

        weight_init.c2_msra_fill(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x, channel_first=True)
        return x


class ConvNeXtBlock(CNNBlock):
    def __init__(
        self,
        embed_dim: int,
        norm: str | dict | nn.Module = {"name": "LN2d", "eps": 1e-6},
        activation: str | dict | nn.Module = "GELU",
        mlp_ratio: float = 4.0,
        layer_scale_init_value: float = 1e-6,
        drop_path_rate: float = 0.0,
    ):
        super().__init__(embed_dim, embed_dim, 1)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.dwconv = nn.Conv2d(embed_dim, embed_dim, kernel_size=7, stride=1, padding=3, groups=embed_dim)
        self.norm = get_norm(norm, embed_dim)
        self.pwconv1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = get_activation(activation)
        self.pwconv2 = nn.Linear(hidden_dim, embed_dim)
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(embed_dim, 1, 1), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dwconv(x)
        # [N, C, H, W] -> [N, H, W, C]
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out, channel_first=False)
        out = self.pwconv1(out)
        out = self.activation(out)
        out = self.pwconv2(out)
        # [N, H, W, C] -> [N, C, H, W]
        out = out.permute(0, 3, 1, 2)
        if self.layer_scale is not None:
            out = self.layer_scale * out
        out = x + self.drop_path(out)
        return out


class ConvNeXtStage(CNNBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        depth: int,
        norm: str | dict | nn.Module = {"name": "LN2d", "eps": 1e-6},
        activation: str | dict | nn.Module = "GELU",
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: float = 4.0,
        drop_path_rates: list[float] | None = None,
        use_checkpoint: bool = False,
    ):
        super().__init__(in_channels, out_channels, stride)

        self.use_checkpoint = use_checkpoint

        if stride != 1:
            self.downsample = nn.Sequential(
                get_norm(norm, in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=True),
            )
        else:
            self.downsample = None

        blocks = []
        drop_path_rates = drop_path_rates or [0.0] * depth
        for i in range(depth):
            block = ConvNeXtBlock(
                embed_dim=out_channels,
                norm=norm,
                activation=activation,
                layer_scale_init_value=layer_scale_init_value,
                drop_path_rate=drop_path_rates[i],
                mlp_ratio=mlp_ratio,
            )
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)

        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class ConvNeXt(Backbone):
    """
    support ConvNeXt (https://arxiv.org/abs/2201.03545).
    """

    def __init__(
        self,
        embed_dims: list[int],
        depths: list[int],
        input_shape: ShapeSpec | None = None,
        drop_path_rate: float = 0.0,
        strides: list[int] = [1, 2, 2, 2],
        mlp_ratio: float = 4.0,
        layer_scale_init_value: float = 1e-6,
        norm: str | dict | nn.Module = {"name": "LN2d", "eps": 1e-6},
        activation: str | dict | nn.Module = "GELU",
        use_checkpoint: bool = False,
        out_features: list[str] | None = None,
        out_feature_norm: bool = True,
        freeze_at: int = 0,
        num_classes: int | None = None,
    ):
        super().__init__()

        if input_shape is None:
            input_shape = ShapeSpec(channels=3)

        self.num_classes = num_classes
        self.out_feature_norm = out_feature_norm

        if out_features is not None:
            num_stages = max([{"stage2": 1, "stage3": 2, "stage4": 3, "stage5": 4}.get(f, 0) for f in out_features])
            embed_dims = embed_dims[:num_stages]
            depths = depths[:num_stages]
            strides = strides[:num_stages]

        self.stem = Stem(input_shape.channels, embed_dims[0], norm)
        self._out_feature_channels["stem"] = embed_dims[0]
        cur_stride = self.stem.stride
        self._out_feature_strides["stem"] = cur_stride

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        self.stages = []
        for i in range(len(depths)):
            name = f"stage{i + 2}"
            drop_path_rates = []
            for _ in range(depths[i]):
                drop_path_rate_i = drop_path_rate * stage_block_id / (total_stage_blocks - 1.0)
                drop_path_rates.append(drop_path_rate_i)
                stage_block_id += 1
            stage = ConvNeXtStage(
                in_channels=embed_dims[i - 1] if i > 0 else embed_dims[0],
                out_channels=embed_dims[i],
                stride=strides[i],
                depth=depths[i],
                norm=norm,
                activation=activation,
                mlp_ratio=mlp_ratio,
                layer_scale_init_value=layer_scale_init_value,
                drop_path_rates=drop_path_rates,
                use_checkpoint=use_checkpoint,
            )
            self.add_module(name, stage)
            if out_feature_norm and name in out_features:
                self.add_module(f"norm{i + 2}", get_norm(norm, embed_dims[i]))
            self.stages.append(name)
            self._out_feature_channels[name] = embed_dims[i]
            cur_stride *= strides[i]
            self._out_feature_strides[name] = cur_stride

        if num_classes is not None:
            in_channels = embed_dims[-1]
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.linear = nn.Sequential(
                get_norm(norm, in_channels),
                nn.Flatten(1),
                nn.Linear(in_channels, num_classes),
            )
            weight_init.trunc_normal_fill(self.linear[2], std=0.02)

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
            "ConvNeXt-Tiny": convnext_tiny,
            "ConvNeXt-Small": convnext_small,
            "ConvNeXt-Base": convnext_base,
            "ConvNeXt-Large": convnext_large,
        }

        if name not in predefined:
            raise ValueError(f"`{name}` is not predefined for ConvNeXt. Available names: {', '.join(predefined)}.")

        return predefined[name](**kwargs)

    def freeze(self, freeze_at: int = 0):
        if freeze_at >= 1:
            self.stem.freeze()

        for idx, stage_name in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                stage = getattr(self, stage_name)
                stage.freeze()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}

        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x

        for idx, stage_name in enumerate(self.stages, start=2):
            stage = getattr(self, stage_name)
            x = stage(x)
            if stage_name in self._out_features:
                outputs[stage_name] = getattr(self, f"norm{idx}")(x, channel_first=True) if self.out_feature_norm else x

        if self.num_classes is not None:
            x = self.avgpool(x)
            x = self.linear(x)
            outputs["linear"] = x

        return outputs


@validate_factory_kwargs("embed_dims", "depths")
def convnext_tiny(**kwargs):
    return ConvNeXt(
        embed_dims=[96, 192, 384, 768],
        depths=[3, 3, 9, 3],
        **kwargs,
    )


@validate_factory_kwargs("embed_dims", "depths")
def convnext_small(**kwargs):
    return ConvNeXt(
        embed_dims=[96, 192, 384, 768],
        depths=[3, 3, 27, 3],
        **kwargs,
    )


@validate_factory_kwargs("embed_dims", "depths")
def convnext_base(**kwargs):
    return ConvNeXt(
        embed_dims=[128, 256, 512, 1024],
        depths=[3, 3, 27, 3],
        **kwargs,
    )


@validate_factory_kwargs("embed_dims", "depths")
def convnext_large(**kwargs):
    return ConvNeXt(
        embed_dims=[192, 384, 768, 1536],
        depths=[3, 3, 27, 3],
        **kwargs,
    )
