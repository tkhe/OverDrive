import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint

from overdrive.layers import Conv2d
from overdrive.layers import DropPath
from overdrive.layers import FreezableMixin
from overdrive.layers import ShapeSpec
from overdrive.layers import get_activation
from overdrive.layers import get_norm

from .base import Backbone
from .base import validate_factory_kwargs

__all__ = ["EdgeNeXt"]


class PositionalEncodingFourier(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        dim: int = 768,
        temperature: float = 10000.0,
    ):
        super().__init__()

        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, shape: tuple[int, int, int]) -> torch.Tensor:
        device = self.token_projection.weight.device
        dtype = self.token_projection.weight.dtype

        inv_mask = ~torch.zeros(shape).to(device=device, dtype=torch.bool)
        y_embed = inv_mask.cumsum(1, dtype=torch.float32)
        x_embed = inv_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.int64, device=device).to(torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return self.token_projection(pos.to(dtype))


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        activation: str | dict | nn.Module = "GELU",
        drop: float = 0.0,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = get_activation(activation)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ConvBlock(nn.Module):
    """
    Depthwise-convolutional EdgeNeXt block.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        kernel_size: int = 7,
        stride: int = 1,
        conv_bias: bool = True,
        expand_ratio: float = 4,
        ls_init_value: float = 1e-6,
        norm: str | dict | nn.Module = {"name": "LN", "eps": 1e-6},
        activation: str | dict | nn.Module = "GELU",
        drop_path: float = 0.0,
    ):
        super().__init__()

        dim_out = dim_out or dim
        self.shortcut_after_dw = stride > 1 or dim != dim_out

        self.conv_dw = nn.Conv2d(dim, dim_out, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=dim, bias=conv_bias)
        self.norm = get_norm(norm, dim_out)
        self.mlp = MLP(dim_out, int(expand_ratio * dim_out), activation=activation)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim_out)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv_dw(x)
        if self.shortcut_after_dw:
            shortcut = x

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        if self.drop_path is not None:
            x = self.drop_path(x)
        return shortcut + x


class CrossCovarianceAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, -1).permute(2, 0, 3, 4, 1)
        q, k, v = qkv.unbind(0)

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.permute(0, 3, 1, 2).reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class SplitTransposeBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_scales: int = 1,
        num_heads: int = 8,
        expand_ratio: float = 4,
        use_pos_emb: bool = True,
        conv_bias: bool = True,
        qkv_bias: bool = True,
        ls_init_value: float = 1e-6,
        norm: str | dict | nn.Module = {"name": "LN", "eps": 1e-6},
        activation: str | dict | nn.Module = "GELU",
        drop_path: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        width = max(int(math.ceil(dim / num_scales)), int(math.floor(dim // num_scales)))
        self.width = width
        self.num_scales = max(1, num_scales - 1)

        convs = [nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, groups=width, bias=conv_bias) for _ in range(self.num_scales)]
        self.convs = nn.ModuleList(convs)

        self.pos_embd = PositionalEncodingFourier(dim=dim) if use_pos_emb else None
        self.norm_xca = get_norm(norm, dim)
        self.gamma_xca = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.xca = CrossCovarianceAttn(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)

        self.norm = get_norm(norm, dim)
        self.mlp = MLP(dim, int(expand_ratio * dim), activation=activation)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        split_x = x.split(self.width, dim=1)
        split_out = []
        split = split_x[0]
        for i, conv in enumerate(self.convs):
            if i > 0:
                split = split + split_x[i]
            split = conv(split)
            split_out.append(split)
        split_out.append(split_x[-1])
        x = torch.cat(split_out, dim=1)

        batch_size, channels, height, width = x.shape
        x = x.reshape(batch_size, channels, height * width).permute(0, 2, 1)
        if self.pos_embd is not None:
            pos_encoding = self.pos_embd((batch_size, height, width)).reshape(batch_size, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding
        xca = self.xca(self.norm_xca(x))
        if self.gamma_xca is not None:
            xca = self.gamma_xca * xca
        if self.drop_path is not None:
            xca = self.drop_path(xca)
        x = x + xca
        x = x.reshape(batch_size, height, width, channels)

        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        if self.drop_path is not None:
            x = self.drop_path(x)
        return shortcut + x


class EdgeNeXtStage(nn.Module, FreezableMixin):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        depth: int = 2,
        num_global_blocks: int = 1,
        num_heads: int = 4,
        scales: int = 2,
        kernel_size: int = 7,
        expand_ratio: float = 4,
        use_pos_emb: bool = False,
        downsample_block: bool = False,
        conv_bias: bool = True,
        ls_init_value: float = 1e-6,
        drop_path_rates: list[float] | None = None,
        norm: str | dict | nn.Module = {"name": "LN2d", "eps": 1e-6},
        activation: str | dict | nn.Module = "GELU",
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth

        if downsample_block or stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                get_norm(norm, in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=conv_bias),
            )
            in_channels = out_channels

        stage_blocks = []
        for i in range(depth):
            if i < depth - num_global_blocks:
                block = ConvBlock(
                    dim=in_channels,
                    dim_out=out_channels,
                    stride=stride if downsample_block and i == 0 else 1,
                    conv_bias=conv_bias,
                    kernel_size=kernel_size,
                    expand_ratio=expand_ratio,
                    ls_init_value=ls_init_value,
                    drop_path=drop_path_rates[i],
                    norm={"name": "LN", "eps": 1e-6},
                    activation=activation,
                )
            else:
                block = SplitTransposeBlock(
                    dim=in_channels,
                    num_scales=scales,
                    num_heads=num_heads,
                    expand_ratio=expand_ratio,
                    use_pos_emb=use_pos_emb,
                    conv_bias=conv_bias,
                    ls_init_value=ls_init_value,
                    drop_path=drop_path_rates[i],
                    norm={"name": "LN", "eps": 1e-6},
                    activation=activation,
                )
            stage_blocks.append(block)
            in_channels = out_channels
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        if self.use_checkpoint:
            x = checkpoint.checkpoint_sequential(self.blocks, len(self.blocks), x, use_reentrant=False)
        else:
            x = self.blocks(x)
        return x


class EdgeNeXt(Backbone):
    """
    support EdgeNeXt (https://arxiv.org/abs/2206.10589).
    """

    def __init__(
        self,
        dims: tuple[int, ...],
        depths: tuple[int, ...],
        global_block_counts: tuple[int, ...] = (0, 1, 1, 1),
        kernel_sizes: tuple[int, ...] = (3, 5, 7, 9),
        heads: tuple[int, ...] = (8, 8, 8, 8),
        scales: tuple[int, ...] = (2, 2, 3, 4),
        use_pos_emb: tuple[bool, ...] = (False, True, False, False),
        input_shape: ShapeSpec | None = None,
        norm: str | dict | nn.Module = {"name": "LN2d", "eps": 1e-6},
        ls_init_value: float = 1e-6,
        expand_ratio: float = 4,
        downsample_block: bool = False,
        conv_bias: bool = True,
        stem_type: str = "patch",
        norm_first: bool = False,
        activation: str | dict | nn.Module = "GELU",
        drop_path_rate: float = 0.0,
        drop_rate: float = 0.0,
        use_checkpoint: bool = False,
        out_features: list[str] | None = None,
        freeze_at: int = 0,
        num_classes: int | None = None,
    ):
        super().__init__()

        if input_shape is None:
            input_shape = ShapeSpec(channels=3)

        in_channels = input_shape.channels
        num_stages = len(depths)
        drop_path_rates = [x.tolist() for x in torch.linspace(0.0, drop_path_rate, sum(depths)).split(depths)]
        if out_features is not None:
            num_stages = max([{"stage2": 1, "stage3": 2, "stage4": 3, "stage5": 4}.get(f, 0) for f in out_features])
            dims = dims[:num_stages]
            depths = depths[:num_stages]
            global_block_counts = global_block_counts[:num_stages]
            kernel_sizes = kernel_sizes[:num_stages]
            heads = heads[:num_stages]
            scales = scales[:num_stages]
            use_pos_emb = use_pos_emb[:num_stages]
            drop_path_rates = drop_path_rates[:num_stages]

        self.num_classes = num_classes

        if stem_type not in {"patch", "overlap"}:
            raise ValueError(f"Unsupported stem_type: {stem_type}. Expected one of: 'patch', 'overlap'")

        if stem_type == "patch":
            self.stem = Conv2d(
                in_channels,
                dims[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=conv_bias,
                norm=norm,
            )
        else:
            self.stem = Conv2d(
                in_channels,
                dims[0],
                kernel_size=9,
                stride=4,
                padding=9 // 2,
                bias=conv_bias,
                norm=norm,
            )

        self._out_feature_channels["stem"] = dims[0]
        self._out_feature_strides["stem"] = 4

        current_stride = 4
        in_channels = dims[0]
        self.stages: list[str] = []
        for i in range(num_stages):
            stage_name = f"stage{i + 2}"
            stride = 2 if current_stride == 2 or i > 0 else 1
            current_stride *= stride
            stage = EdgeNeXtStage(
                in_channels=in_channels,
                out_channels=dims[i],
                stride=stride,
                depth=depths[i],
                num_global_blocks=global_block_counts[i],
                num_heads=heads[i],
                scales=scales[i],
                expand_ratio=expand_ratio,
                kernel_size=kernel_sizes[i],
                use_pos_emb=use_pos_emb[i],
                ls_init_value=ls_init_value,
                downsample_block=downsample_block,
                conv_bias=conv_bias,
                norm=norm,
                activation=activation,
                drop_path_rates=drop_path_rates[i],
                use_checkpoint=use_checkpoint,
            )
            self.add_module(stage_name, stage)
            self.stages.append(stage_name)
            self._out_feature_channels[stage_name] = dims[i]
            self._out_feature_strides[stage_name] = current_stride
            in_channels = dims[i]

        if num_classes is not None:
            head = []
            if norm_first:
                head.append(get_norm(norm, in_channels))
            head.append(nn.AdaptiveAvgPool2d(1))
            if not norm_first:
                head.append(get_norm(norm, in_channels))
            head.append(nn.Flatten(1))
            head.append(nn.Dropout(drop_rate))
            head.append(nn.Linear(in_channels, num_classes))
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
            "EdgeNeXt-XXSmall": edgenext_xx_small,
            "EdgeNeXt-XSmall": edgenext_x_small,
            "EdgeNeXt-Small": edgenext_small,
            "EdgeNeXt-Base": edgenext_base,
            "EdgeNeXt-Small-RW": edgenext_small_rw,
        }

        if name not in predefined:
            raise ValueError(f"`{name}` is not predefined for EdgeNeXt. Available names: {', '.join(predefined)}.")
        return predefined[name](**kwargs)

    def freeze(self, freeze_at: int = 0):
        if freeze_at >= 1:
            self.stem.freeze()
        for index, stage_name in enumerate(self.stages, start=2):
            if freeze_at >= index:
                stage = getattr(self, stage_name)
                stage.freeze()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}
        x = self.stem(x)

        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if stage_name in self._out_features:
                outputs[stage_name] = x

        if self.num_classes is not None:
            outputs["linear"] = self.head(x)

        return outputs


@validate_factory_kwargs("dims", "depths", "heads")
def edgenext_xx_small(**kwargs):
    return EdgeNeXt(
        dims=(24, 48, 88, 168),
        depths=(2, 2, 6, 2),
        heads=(4, 4, 4, 4),
        **kwargs,
    )


@validate_factory_kwargs("dims", "depths", "heads")
def edgenext_x_small(**kwargs):
    return EdgeNeXt(
        dims=(32, 64, 100, 192),
        depths=(3, 3, 9, 3),
        heads=(4, 4, 4, 4),
        **kwargs,
    )


@validate_factory_kwargs("dims", "depths")
def edgenext_small(**kwargs):
    return EdgeNeXt(
        dims=(48, 96, 160, 304),
        depths=(3, 3, 9, 3),
        **kwargs,
    )


@validate_factory_kwargs("dims", "depths")
def edgenext_base(**kwargs):
    return EdgeNeXt(
        dims=(80, 160, 288, 584),
        depths=(3, 3, 9, 3),
        **kwargs,
    )


@validate_factory_kwargs("dims", "depths", "downsample_block", "conv_bias", "stem_type")
def edgenext_small_rw(**kwargs):
    return EdgeNeXt(
        dims=(48, 96, 192, 384),
        depths=(3, 3, 9, 3),
        downsample_block=True,
        conv_bias=False,
        stem_type="overlap",
        **kwargs,
    )
