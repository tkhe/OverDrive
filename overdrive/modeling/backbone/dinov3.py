import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint

from overdrive.layers import CNNBlock
from overdrive.layers import LayerScale
from overdrive.layers import MLP
from overdrive.layers import ShapeSpec
from overdrive.layers import SwiGLUFFN
from overdrive.layers import get_norm

from .base import Backbone
from .base import validate_factory_kwargs

__all__ = ["DINOv3"]


class PatchEmbed(CNNBlock):
    """
    Project non-overlapping image patches to tokens without internal padding.
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__(in_channels, embed_dim, patch_size)

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class RopePositionEmbedding(nn.Module):
    """
    Axial rotary positions with optional training-time coordinate augmentation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: str = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = 2.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if num_heads <= 0 or embed_dim % (4 * num_heads) != 0:
            raise ValueError("embed_dim must be divisible by 4 * num_heads.")

        if base is not None:
            if base <= 0 or min_period is not None or max_period is not None:
                raise ValueError("Provide a positive base or min_period and max_period, exclusively.")
        elif min_period is None or max_period is None or not 0 < min_period <= max_period:
            raise ValueError("Expected 0 < min_period <= max_period when base is None.")

        if normalize_coords not in ("min", "max", "separate"):
            raise ValueError("normalize_coords must be 'min', 'max', or 'separate'.")

        if shift_coords is not None and shift_coords < 0:
            raise ValueError("shift_coords must be nonnegative.")

        if any(value is not None and value < 1 for value in (jitter_coords, rescale_coords)):
            raise ValueError("jitter_coords and rescale_coords must be at least 1.")

        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("RoPE dtype must be float32, float16, or bfloat16.")

        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype
        head_dim = embed_dim // num_heads
        if base is not None:
            periods = base ** (2 * torch.arange(head_dim // 4, dtype=dtype) / (head_dim // 2))
        else:
            ratio = max_period / min_period
            periods = ratio ** torch.linspace(0, 1, head_dim // 4, dtype=dtype) / ratio * max_period

        self.register_buffer("periods", periods)

    def forward(self, H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.normalize_coords == "separate":
            scale_h, scale_w = H, W
        else:
            scale_h = scale_w = min(H, W) if self.normalize_coords == "min" else max(H, W)

        coords_h = torch.arange(0.5, H, device=self.periods.device, dtype=self.dtype) / scale_h
        coords_w = torch.arange(0.5, W, device=self.periods.device, dtype=self.dtype) / scale_w
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).flatten(0, 1)
        coords = 2.0 * coords - 1.0
        if self.training:
            if self.shift_coords is not None:
                coords = coords + torch.empty(2, device=self.periods.device, dtype=self.dtype).uniform_(-self.shift_coords, self.shift_coords)[None, :]
            if self.jitter_coords is not None:
                jitter = math.log(self.jitter_coords)
                coords = coords * torch.empty(2, device=self.periods.device, dtype=self.dtype).uniform_(-jitter, jitter).exp()[None, :]
            if self.rescale_coords is not None:
                rescale = math.log(self.rescale_coords)
                coords = coords * torch.empty(1, device=self.periods.device, dtype=self.dtype).uniform_(-rescale, rescale).exp()
        angles = 2 * math.pi * coords[:, :, None] / self.periods.to(self.dtype)[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return angles.sin(), angles.cos()


class LinearKMaskedBias(nn.Linear):
    """
    QKV projection whose key bias is always masked out.
    """

    def __init__(self, dim: int, bias: bool = True):
        super().__init__(dim, 3 * dim, bias=bias)

        if bias:
            bias_mask = torch.ones_like(self.bias)
            bias_mask[dim : 2 * dim] = 0
            self.register_buffer("bias_mask", bias_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(x, self.weight, bias)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, proj_bias: bool, mask_k_bias: bool):
        super().__init__()

        self.num_heads = num_heads
        self.qkv = LinearKMaskedBias(dim, bias=qkv_bias) if mask_k_bias else nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    @staticmethod
    def apply_rope(x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        sin, cos = rope
        dtype = x.dtype
        x = x.to(sin.dtype)
        prefix = x.shape[-2] - sin.shape[-2]
        patch = x[:, :, prefix:, :]
        first, second = patch.chunk(2, dim=-1)
        patch = patch * cos + torch.cat((-second, first), dim=-1) * sin
        return torch.cat((x[:, :, :prefix, :], patch), dim=-2).to(dtype)

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
        q, k = self.apply_rope(q, rope), self.apply_rope(k, rope)
        x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class SelfAttentionBlock(nn.Module):
    """
    DINOv3 block with independently sampled attention and FFN residuals.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float,
        ffn_layer: str,
        norm: str | dict,
        qkv_bias: bool,
        proj_bias: bool,
        ffn_bias: bool,
        mask_k_bias: bool,
        layerscale_init: float | None,
        drop_path: float,
    ):
        super().__init__()

        self.norm1 = get_norm(norm, dim)
        self.norm2 = get_norm(norm, dim)

        if self.norm1 is None or self.norm2 is None:
            raise ValueError("DINOv3 requires a normalization layer.")

        self.attn = SelfAttention(dim, num_heads, qkv_bias, proj_bias, mask_k_bias)
        self.ls1 = LayerScale(dim, scale=layerscale_init, inplace=False) if layerscale_init else None
        self.ls2 = LayerScale(dim, scale=layerscale_init, inplace=False) if layerscale_init else None
        hidden_dim = int(dim * ffn_ratio)
        if ffn_layer == "swiglu":
            hidden_dim = int(hidden_dim * 2 / 3)
            hidden_dim += -hidden_dim % 8
            self.mlp = SwiGLUFFN(embed_dim=dim, hidden_dim=hidden_dim, bias=ffn_bias)
        else:
            self.mlp = MLP(dim, hidden_dim, activation="GELU", bias=ffn_bias)
        self.sample_drop_ratio = drop_path

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if self.training and self.sample_drop_ratio > 0:
            B = x.shape[0]
            subset_size = max(int(B * (1 - self.sample_drop_ratio)), 1)
            scale = B / subset_size
            indices = torch.randperm(B, device=x.device)[:subset_size]
            residual = self.attn(self.norm1(x[indices]), rope)
            if self.ls1 is not None:
                residual = self.ls1(residual)
            x = torch.index_add(x, dim=0, index=indices, source=residual, alpha=scale)
            indices = torch.randperm(B, device=x.device)[:subset_size]
            residual = self.mlp(self.norm2(x[indices]))
            if self.ls2 is not None:
                residual = self.ls2(residual)
            return torch.index_add(x, dim=0, index=indices, source=residual, alpha=scale)

        residual = self.attn(self.norm1(x), rope)
        if self.ls1 is not None:
            residual = self.ls1(residual)
        x = x + residual
        residual = self.mlp(self.norm2(x))
        if self.ls2 is not None:
            residual = self.ls2(residual)
        return x + residual


class DINOv3(Backbone):
    """
    DINOv3 ViT backbone with native, same-resolution intermediate features.
    """

    def __init__(
        self,
        input_shape: ShapeSpec | None = None,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        ffn_layer: str = "mlp",
        norm: str | dict = "LN",
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        mask_k_bias: bool = True,
        layerscale_init: float | None = 1e-5,
        n_storage_tokens: int = 4,
        pos_embed_rope_base: float | None = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: str = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = 2.0,
        pos_embed_rope_dtype: torch.dtype = torch.float32,
        drop_path_rate: float = 0.0,
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
            embed_dim (int): Token embedding dimension.
            depth (int): Number of Transformer blocks.
            num_heads (int): Attention heads; head dimension must be divisible by four.
            ffn_ratio (float): Hidden dimension ratio before SwiGLU's 2/3 adjustment.
            ffn_layer (str): Feed-forward layer, either "mlp" or "swiglu".
            norm (str or dict): Token normalization. Default: LN with epsilon 1e-5.
            qkv_bias, proj_bias, ffn_bias (bool): Enable the respective projection biases.
            mask_k_bias (bool): Mask the key projection bias.
            layerscale_init (float, optional): Initial residual scale; None or zero disables it.
            n_storage_tokens (int): Number of storage tokens following CLS.
            pos_embed_rope_base (float, optional): RoPE period base. Default: 100. Set None to use period bounds.
            pos_embed_rope_min_period, pos_embed_rope_max_period (float, optional): RoPE period bounds. Default: None.
            pos_embed_rope_normalize_coords (str): Coordinate normalization: "min", "max", or "separate". Default: "separate".
            pos_embed_rope_shift_coords (float, optional): Training coordinate shift magnitude. Default: None.
            pos_embed_rope_jitter_coords (float, optional): Independent axis scaling bound. Default: None.
            pos_embed_rope_rescale_coords (float, optional): Joint axis scaling bound during training. Default: 2.
            pos_embed_rope_dtype (torch.dtype): Arithmetic precision for rotary positions. Default: torch.float32.
            drop_path_rate (float): Uniform residual subset sampling rate across all blocks.
            use_checkpoint (bool): Recompute blocks during backward to save memory.
            out_features (list[str], optional): Zero-based block names, e.g. ["block2", "block5"].
                Defaults to the final block. All outputs share the same stride and channels.
            freeze_at (int): Values <= 0 do not freeze layers. Freeze embedding at 1 and the first N-1 blocks at N >= 2.
                At depth+1 or higher also freeze output norm. The optional classifier remains trainable.
            num_classes (int, optional): Classify the concatenation of the normalized final CLS token
                and the mean of normalized final patch tokens, excluding storage tokens.
            dropout (float): Dropout before the optional classifier.
        """
        super().__init__()

        if patch_size <= 0 or depth <= 0 or embed_dim <= 0 or ffn_ratio <= 0:
            raise ValueError("patch_size, depth, embed_dim, and ffn_ratio must be positive.")

        if num_heads <= 0 or embed_dim % (4 * num_heads) != 0:
            raise ValueError("embed_dim must be divisible by 4 * num_heads.")

        if ffn_layer not in ("mlp", "swiglu"):
            raise ValueError("ffn_layer must be 'mlp' or 'swiglu'.")

        if n_storage_tokens < 0:
            raise ValueError("n_storage_tokens must be nonnegative.")

        if not 0 <= drop_path_rate < 1:
            raise ValueError("drop_path_rate must be between 0 (inclusive) and 1 (exclusive).")

        if num_classes is not None and num_classes <= 0:
            raise ValueError("num_classes must be positive or None.")

        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1.")

        if input_shape is None:
            input_shape = ShapeSpec(channels=3)

        if out_features is not None:
            depth = max([{f"block{i}": i + 1 for i in range(depth)}.get(f, 0) for f in out_features])

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.n_storage_tokens = n_storage_tokens
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self._freeze_at = freeze_at

        self.patch_embed = PatchEmbed(input_shape.channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        if n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim))
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=pos_embed_rope_dtype,
        )
        self.blocks = nn.ModuleList()
        for i in range(depth):
            name = f"block{i}"
            block = SelfAttentionBlock(
                embed_dim,
                num_heads,
                ffn_ratio,
                ffn_layer,
                norm,
                qkv_bias,
                proj_bias,
                ffn_bias,
                mask_k_bias,
                layerscale_init,
                drop_path_rate,
            )
            self.blocks.append(block)
            self._out_feature_channels[name] = embed_dim
            self._out_feature_strides[name] = patch_size

        self.norm = get_norm(norm, embed_dim)
        if num_classes is not None:
            self.dropout = nn.Dropout(dropout)
            self.linear = nn.Linear(2 * embed_dim, num_classes)

        if out_features is None:
            out_features = [f"block{depth - 1}"]

        self._out_features = out_features
        if len(self._out_features) == 0:
            raise ValueError("out_features must contain at least one feature name.")

        children = [f"block{x[0]}" for x in self.blocks.named_children()]
        for out_feature in self._out_features:
            if out_feature not in children:
                raise ValueError(f"Available children: {', '.join(children)}")

        self.apply(self._init_weights)
        nn.init.normal_(self.cls_token, std=0.02)
        if n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)

        self.freeze(freeze_at)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.reset_parameters()
        elif isinstance(module, nn.Conv2d):
            bound = 1 / math.sqrt(module.in_channels * math.prod(module.kernel_size))
            nn.init.uniform_(module.weight, -bound, bound)
            if module.bias is not None:
                nn.init.uniform_(module.bias, -bound, bound)

    @staticmethod
    def build(name, **kwargs):
        predefined = {
            "DINOv3-Small": dinov3_vits16,
            "DINOv3-Small-Plus": dinov3_vits16plus,
            "DINOv3-Base": dinov3_vitb16,
            "DINOv3-Large": dinov3_vitl16,
            "DINOv3-Large-Plus": dinov3_vitl16plus,
            "DINOv3-Huge-Plus": dinov3_vith16plus,
        }
        if name not in predefined:
            raise ValueError(f"`{name}` is not predefined for DINOv3. Available names: {', '.join(predefined)}.")
        return predefined[name](**kwargs)

    def freeze(self, freeze_at: int = 0):
        if freeze_at >= 1:
            self.patch_embed.freeze()
            self.cls_token.requires_grad_(False)
            self.mask_token.requires_grad_(False)
            if self.n_storage_tokens > 0:
                self.storage_tokens.requires_grad_(False)

        for idx, block in enumerate(self.blocks, start=2):
            if freeze_at >= idx:
                block.eval()
                block.requires_grad_(False)

        if freeze_at >= self.depth + 1:
            self.norm.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)

        self.freeze(self._freeze_at)
        return self

    @property
    def padding_constraints(self) -> dict[str, int]:
        return {"size_divisiblity": self.patch_size, "square_size": 0}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}

        x = self.patch_embed(x)
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        cls_token = (self.cls_token + 0 * self.mask_token).expand(B, -1, -1)
        tokens = [cls_token]
        if self.n_storage_tokens > 0:
            tokens.append(self.storage_tokens.expand(B, -1, -1))
        x = torch.cat([*tokens, x], dim=1)

        if not self.training:
            rope = self.rope_embed(H=H, W=W)
        for i, block in enumerate(self.blocks):
            if self.training:
                # Sample RoPE once per block, outside checkpoint recomputation.
                rope = self.rope_embed(H=H, W=W)

            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x, rope, use_reentrant=False)
            else:
                x = block(x, rope)

            name = f"block{i}"
            if name in self._out_features:
                patches = self.norm(x)[:, self.n_storage_tokens + 1 :]
                outputs[name] = patches.transpose(1, 2).reshape(B, self.embed_dim, H, W).contiguous()

        if self.num_classes is not None:
            x = self.norm(x)
            x = torch.cat((x[:, 0], x[:, self.n_storage_tokens + 1 :].mean(dim=1)), dim=1)
            outputs["linear"] = self.linear(self.dropout(x))
        return outputs


@validate_factory_kwargs("patch_size", "embed_dim", "depth", "num_heads", "ffn_ratio", "ffn_layer")
def dinov3_vits16(**kwargs):
    return DINOv3(patch_size=16, embed_dim=384, depth=12, num_heads=6, ffn_ratio=4.0, ffn_layer="mlp", **kwargs)


@validate_factory_kwargs("patch_size", "embed_dim", "depth", "num_heads", "ffn_ratio", "ffn_layer")
def dinov3_vits16plus(**kwargs):
    return DINOv3(patch_size=16, embed_dim=384, depth=12, num_heads=6, ffn_ratio=6.0, ffn_layer="swiglu", **kwargs)


@validate_factory_kwargs("patch_size", "embed_dim", "depth", "num_heads", "ffn_ratio", "ffn_layer")
def dinov3_vitb16(**kwargs):
    return DINOv3(patch_size=16, embed_dim=768, depth=12, num_heads=12, ffn_ratio=4.0, ffn_layer="mlp", **kwargs)


@validate_factory_kwargs("patch_size", "embed_dim", "depth", "num_heads", "ffn_ratio", "ffn_layer")
def dinov3_vitl16(**kwargs):
    return DINOv3(patch_size=16, embed_dim=1024, depth=24, num_heads=16, ffn_ratio=4.0, ffn_layer="mlp", **kwargs)


@validate_factory_kwargs("patch_size", "embed_dim", "depth", "num_heads", "ffn_ratio", "ffn_layer")
def dinov3_vitl16plus(**kwargs):
    return DINOv3(patch_size=16, embed_dim=1024, depth=24, num_heads=16, ffn_ratio=6.0, ffn_layer="swiglu", **kwargs)


@validate_factory_kwargs("patch_size", "embed_dim", "depth", "num_heads", "ffn_ratio", "ffn_layer")
def dinov3_vith16plus(**kwargs):
    return DINOv3(patch_size=16, embed_dim=1280, depth=32, num_heads=20, ffn_ratio=6.0, ffn_layer="swiglu", **kwargs)
