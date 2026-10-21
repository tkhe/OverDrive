import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint

from overdrive.layers import CNNBlock
from overdrive.layers import DropPath
from overdrive.layers import LayerScale
from overdrive.layers import MLP
from overdrive.layers import ShapeSpec
from overdrive.layers import get_norm

from .base import Backbone
from .base import validate_factory_kwargs

__all__ = ["DINOv2"]


class PatchEmbed(CNNBlock):
    """
    Project non-overlapping image patches to tokens without internal padding.
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__(in_channels, embed_dim, patch_size)

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, proj_bias: bool):
        super().__init__()

        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
        x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class SelfAttentionBlock(nn.Module):
    """
    DINOv2 block with stochastic depth and independently sampled residuals.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float,
        norm: str | dict,
        qkv_bias: bool,
        proj_bias: bool,
        ffn_bias: bool,
        layerscale_init: float | None,
        drop_path: float,
    ):
        super().__init__()

        self.norm1 = get_norm(norm, dim)
        self.norm2 = get_norm(norm, dim)
        if self.norm1 is None or self.norm2 is None:
            raise ValueError("DINOv2 requires a normalization layer.")

        self.attn = SelfAttention(dim, num_heads, qkv_bias, proj_bias)
        self.ls1 = LayerScale(dim, scale=layerscale_init, inplace=False) if layerscale_init else None
        self.ls2 = LayerScale(dim, scale=layerscale_init, inplace=False) if layerscale_init else None
        self.mlp = MLP(dim, int(dim * ffn_ratio), activation="GELU", bias=ffn_bias)
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)
        self.sample_drop_ratio = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.sample_drop_ratio > 0.1:
            B = x.shape[0]
            subset_size = max(int(B * (1 - self.sample_drop_ratio)), 1)
            scale = B / subset_size
            indices = torch.randperm(B, device=x.device)[:subset_size]
            residual = self.attn(self.norm1(x[indices]))
            if self.ls1 is not None:
                residual = self.ls1(residual)
            x = torch.index_add(x, dim=0, index=indices, source=residual, alpha=scale)
            indices = torch.randperm(B, device=x.device)[:subset_size]
            residual = self.mlp(self.norm2(x[indices]))
            if self.ls2 is not None:
                residual = self.ls2(residual)
            return torch.index_add(x, dim=0, index=indices, source=residual, alpha=scale)

        residual = self.attn(self.norm1(x))
        if self.ls1 is not None:
            residual = self.ls1(residual)
        x = x + self.drop_path1(residual)
        residual = self.mlp(self.norm2(x))
        if self.ls2 is not None:
            residual = self.ls2(residual)
        return x + self.drop_path2(residual)


class DINOv2(Backbone):
    """
    DINOv2 ViT backbone with native, same-resolution intermediate features.
    """

    def __init__(
        self,
        input_shape: ShapeSpec | None = None,
        patch_size: int = 14,
        image_size: int = 518,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        norm: str | dict = {"name": "LN", "eps": 1e-6},
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        layerscale_init: float | None = 1.0,
        num_register_tokens: int = 0,
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        use_checkpoint: bool = False,
        out_features: list[str] | None = None,
        freeze_at: int = 0,
        num_classes: int | None = None,
        dropout: float = 0.0,
    ):
        """
        Args:
            input_shape (ShapeSpec, optional): Input image shape. Defaults to 3 channels.
            patch_size (int): Patch size and output stride. Padding is handled by the caller.
            image_size (int): Square reference image size for learned positional embeddings.
            embed_dim (int): Token embedding dimension.
            depth (int): Number of Transformer blocks.
            num_heads (int): Number of attention heads.
            ffn_ratio (float): MLP hidden dimension ratio.
            norm (str or dict): Token normalization. Default: LN with epsilon 1e-6.
            qkv_bias, proj_bias, ffn_bias (bool): Enable the respective projection biases.
            layerscale_init (float, optional): Initial residual scale; None or zero disables it.
            num_register_tokens (int): Number of register tokens following CLS.
            interpolate_antialias (bool): Apply antialiasing when resizing positional embeddings.
            interpolate_offset (float): Offset for backward-compatible interpolation scale factors.
            drop_path_rate (float): Maximum stochastic depth rate.
            drop_path_uniform (bool): Use the same rate for every block instead of a linear schedule.
            use_checkpoint (bool): Recompute blocks during backward to save memory.
            out_features (list[str], optional): Zero-based block names, e.g. ["block2", "block5"].
                Defaults to the final block. All outputs share the same stride and channels.
                The classifier concatenates CLS tokens from these blocks in network order.
            freeze_at (int): Values <= 0 do not freeze layers. Freeze embedding at 1 and the first N-1 blocks at N >= 2.
                Output norm and the optional classifier remain trainable.
            num_classes (int, optional): Classify concatenated normalized CLS tokens from out_features
                and mean normalized patch tokens from the final block.
            dropout (float): Dropout before the optional classifier.
        """
        super().__init__()

        if patch_size <= 0 or depth <= 0 or embed_dim <= 0 or ffn_ratio <= 0:
            raise ValueError("patch_size, depth, embed_dim, and ffn_ratio must be positive.")

        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        if image_size < patch_size or image_size % patch_size != 0:
            raise ValueError("image_size must be a positive multiple of patch_size.")

        if not 0 <= interpolate_offset < 1:
            raise ValueError("interpolate_offset must be between 0 (inclusive) and 1 (exclusive).")

        if num_register_tokens < 0:
            raise ValueError("num_register_tokens must be nonnegative.")

        if not 0 <= drop_path_rate < 1:
            raise ValueError("drop_path_rate must be between 0 (inclusive) and 1 (exclusive).")

        if num_classes is not None and num_classes <= 0:
            raise ValueError("num_classes must be positive or None.")

        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1.")

        if input_shape is None:
            input_shape = ShapeSpec(channels=3)

        drop_path_rates = [drop_path_rate] * depth if drop_path_uniform else torch.linspace(0, drop_path_rate, depth).tolist()

        if out_features is not None:
            depth = max([{f"block{i}": i + 1 for i in range(depth)}.get(f, 0) for f in out_features])

        self.image_size = image_size
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self._freeze_at = freeze_at

        self.patch_embed = PatchEmbed(input_shape.channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, embed_dim))
        self.pos_embed = nn.Parameter(torch.empty(1, (image_size // patch_size) ** 2 + 1, embed_dim))
        self.blocks = nn.ModuleList()
        for i in range(depth):
            name = f"block{i}"
            block = SelfAttentionBlock(
                embed_dim,
                num_heads,
                ffn_ratio,
                norm,
                qkv_bias,
                proj_bias,
                ffn_bias,
                layerscale_init,
                drop_path_rates[i],
            )
            self.blocks.append(block)
            self._out_feature_channels[name] = embed_dim
            self._out_feature_strides[name] = patch_size

        self.norm = get_norm(norm, embed_dim)
        if num_classes is not None:
            self.dropout = nn.Dropout(dropout)
            self.linear = nn.Linear((1 + (len(out_features) if out_features is not None else 1)) * embed_dim, num_classes)

        if out_features is None:
            out_features = [f"block{depth - 1}"]

        self._out_features = out_features
        if len(self._out_features) == 0:
            raise ValueError("out_features must contain at least one feature name.")

        if len(set(self._out_features)) != len(self._out_features):
            raise ValueError("out_features must contain unique block names.")

        children = [f"block{x[0]}" for x in self.blocks.named_children()]
        for out_feature in self._out_features:
            if out_feature not in children:
                raise ValueError(f"Available children: {', '.join(children)}")

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if num_register_tokens > 0:
            nn.init.normal_(self.register_tokens, std=1e-6)

        self.freeze(freeze_at)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.reset_parameters()

    @staticmethod
    def build(name, **kwargs):
        predefined = {
            "DINOv2-Small": dinov2_vits14,
            "DINOv2-Base": dinov2_vitb14,
            "DINOv2-Large": dinov2_vitl14,
            "DINOv2-Small-Register": dinov2_vits14_register,
            "DINOv2-Base-Register": dinov2_vitb14_register,
            "DINOv2-Large-Register": dinov2_vitl14_register,
        }
        if name not in predefined:
            raise ValueError(f"`{name}` is not predefined for DINOv2. Available names: {', '.join(predefined)}.")
        return predefined[name](**kwargs)

    def freeze(self, freeze_at: int = 0):
        if freeze_at >= 1:
            self.patch_embed.freeze()
            self.pos_embed.requires_grad_(False)
            self.cls_token.requires_grad_(False)
            self.mask_token.requires_grad_(False)
            if self.num_register_tokens > 0:
                self.register_tokens.requires_grad_(False)

        for idx, block in enumerate(self.blocks, start=2):
            if freeze_at >= idx:
                block.eval()
                block.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)

        self.freeze(self._freeze_at)
        return self

    @property
    def padding_constraints(self) -> dict[str, int]:
        return {"size_divisiblity": self.patch_size, "square_size": 0}

    def interpolate_pos_encoding(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Resize patch positions in float32, preserving the learned CLS position.
        """
        if x.shape[1] - 1 == self.pos_embed.shape[1] - 1 and H == W:
            return self.pos_embed

        M = self.image_size // self.patch_size
        pos_embed = self.pos_embed.float()
        patch_pos_embed = pos_embed[:, 1:].reshape(1, M, M, self.embed_dim).permute(0, 3, 1, 2)
        H, W = H // self.patch_size, W // self.patch_size
        if self.interpolate_offset:
            kwargs = {"scale_factor": ((H + self.interpolate_offset) / M, (W + self.interpolate_offset) / M)}
        else:
            kwargs = {"size": (H, W)}
        patch_pos_embed = F.interpolate(patch_pos_embed, mode="bicubic", antialias=self.interpolate_antialias, **kwargs)
        if patch_pos_embed.shape[-2:] != (H, W):
            raise ValueError("Interpolated positional embedding does not match the patch grid.")
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, self.embed_dim)
        return torch.cat((pos_embed[:, :1], patch_pos_embed), dim=1).to(x.dtype)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}
        cls_tokens = []

        image_h, image_w = x.shape[-2:]
        x = self.patch_embed(x)
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        cls_token = (self.cls_token + 0 * self.mask_token).expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, image_h, image_w)
        if self.num_register_tokens > 0:
            x = torch.cat((x[:, :1], self.register_tokens.expand(B, -1, -1), x[:, 1:]), dim=1)

        for i, block in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

            name = f"block{i}"
            if name in self._out_features:
                normalized = self.norm(x)
                if self.num_classes is not None:
                    cls_tokens.append(normalized[:, 0])
                patches = normalized[:, self.num_register_tokens + 1 :]
                outputs[name] = patches.transpose(1, 2).reshape(B, self.embed_dim, H, W).contiguous()

        if self.num_classes is not None:
            patch_mean = normalized[:, self.num_register_tokens + 1 :].mean(dim=1)
            linear_input = torch.cat([*cls_tokens, patch_mean], dim=1)
            outputs["linear"] = self.linear(self.dropout(linear_input))
        return outputs


@validate_factory_kwargs(
    "patch_size",
    "embed_dim",
    "depth",
    "num_heads",
    "ffn_ratio",
    "num_register_tokens",
    "interpolate_antialias",
    "interpolate_offset",
)
def dinov2_vits14(**kwargs):
    return DINOv2(
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4.0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        **kwargs,
    )


@validate_factory_kwargs(
    "patch_size",
    "embed_dim",
    "depth",
    "num_heads",
    "ffn_ratio",
    "num_register_tokens",
    "interpolate_antialias",
    "interpolate_offset",
)
def dinov2_vitb14(**kwargs):
    return DINOv2(
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4.0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        **kwargs,
    )


@validate_factory_kwargs(
    "patch_size",
    "embed_dim",
    "depth",
    "num_heads",
    "ffn_ratio",
    "num_register_tokens",
    "interpolate_antialias",
    "interpolate_offset",
)
def dinov2_vitl14(**kwargs):
    return DINOv2(
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4.0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        **kwargs,
    )


@validate_factory_kwargs(
    "patch_size",
    "embed_dim",
    "depth",
    "num_heads",
    "ffn_ratio",
    "num_register_tokens",
    "interpolate_antialias",
    "interpolate_offset",
)
def dinov2_vits14_register(**kwargs):
    return DINOv2(
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4.0,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


@validate_factory_kwargs(
    "patch_size",
    "embed_dim",
    "depth",
    "num_heads",
    "ffn_ratio",
    "num_register_tokens",
    "interpolate_antialias",
    "interpolate_offset",
)
def dinov2_vitb14_register(**kwargs):
    return DINOv2(
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4.0,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


@validate_factory_kwargs(
    "patch_size",
    "embed_dim",
    "depth",
    "num_heads",
    "ffn_ratio",
    "num_register_tokens",
    "interpolate_antialias",
    "interpolate_offset",
)
def dinov2_vitl14_register(**kwargs):
    return DINOv2(
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4.0,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )
