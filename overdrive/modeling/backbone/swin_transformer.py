import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as to_2tuple
from torch.utils import checkpoint

from overdrive.layers import MLP
from overdrive.layers import DropPath
from overdrive.layers import ShapeSpec
from overdrive.layers import get_norm

from .base import Backbone
from .base import validate_factory_kwargs

__all__ = ["SwinTransformer"]


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    """

    def __init__(
        self,
        dim: int,
        window_size: int | tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0,
        proj_drop: float = 0,
    ):
        """
        Args:
            dim (int): Number of input channels.
            window_size (int or tuple[int]): The height and width of the window.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value.
                Default: True
            attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
            proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        """
        super().__init__()

        window_size = to_2tuple(window_size)

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm: str | dict = "LN",
        activation: str | dict = "GELU",
        shift_on_small_feature: bool = True,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            window_size (int): Window size.
            shift_size (int): Shift size for SW-MSA.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float, optional): Stochastic depth rate. Default: 0.0
            norm (str or dict): Normalization layer.  Default: LN
            activation (str or dict): Activation layer. Default: GELU
            shift_on_small_feature (bool): If True, keep the configured shift when the shortest
                feature edge is no larger than the window. Default: True.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.shift_on_small_feature = shift_on_small_feature
        if not (0 <= self.shift_size < self.window_size):
            raise ValueError("shift_size must in 0-window_size")

        self.norm1 = get_norm(norm, dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.norm2 = get_norm(norm, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, activation=activation, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x: torch.Tensor, mask_matrix: torch.Tensor | None) -> torch.Tensor:
        B, L, C = x.shape
        H, W = self.H, self.W
        if L != H * W:
            raise ValueError("input feature has wrong size")

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        shift_size = self.shift_size
        if not self.shift_on_small_feature and min(H, W) <= self.window_size:
            shift_size = 0

        # cyclic shift
        if shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if shift_size > 0:
            x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """
    Patch Merging Layer
    """

    def __init__(self, dim: int, norm: str | dict = "LN"):
        """
        Args:
            dim (int): Number of input channels.
            norm (str or dict): Normalization layer.  Default: LN
        """
        super().__init__()

        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = get_norm(norm, 4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        if L != H * W:
            raise ValueError("input feature has wrong size")

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm: str | dict = "LN",
        downsample: type[PatchMerging] | None = None,
        use_checkpoint: bool = False,
        shift_on_small_feature: bool = True,
    ):
        """
        Args:
            dim (int): Number of feature channels
            depth (int): Depths of this stage.
            num_heads (int): Number of attention head.
            window_size (int): Local window size. Default: 7.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
            norm (str, dict): Normalization layer. Default: LN
            downsample (nn.Module | None, optional): Downsample layer at the end of the layer.
                Default: None
            use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
            shift_on_small_feature (bool): If True, keep the configured shift when the shortest
                feature edge is no larger than the window. Default: True.
        """
        super().__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.shift_on_small_feature = shift_on_small_feature

        # build blocks
        blocks = [
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm=norm,
                shift_on_small_feature=shift_on_small_feature,
            )
            for i in range(depth)
        ]

        self.blocks = nn.ModuleList(blocks)

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm=norm)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, int, int, torch.Tensor, int, int]:
        shift_size = self.shift_size
        if not self.shift_on_small_feature and min(H, W) <= self.window_size:
            shift_size = 0

        # calculate attention mask for SW-MSA
        if shift_size > 0:
            Hp = (H + self.window_size - 1) // self.window_size * self.window_size
            Wp = (W + self.window_size - 1) // self.window_size * self.window_size
            img_mask = x.new_zeros((1, Hp, Wp, 1))  # 1 Hp Wp 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -shift_size),
                slice(-shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -shift_size),
                slice(-shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None

        for block in self.blocks:
            block.H, block.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x, attn_mask, use_reentrant=False)
            else:
                x = block(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(
        self,
        patch_size: int | tuple[int, int] = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm: str | dict = "",
    ):
        """
        Args:
            patch_size (int): Patch token size. Default: 4.
            in_chans (int): Number of input image channels. Default: 3.
            embed_dim (int): Number of linear projection output channels. Default: 96.
            norm (str, dict): Normalization layer.
        """
        super().__init__()

        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = get_norm(norm, embed_dim)

    def forward(self, x):
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class SwinTransformer(Backbone):
    """
    Support SwinTransformer (https://arxiv.org/pdf/2103.14030).
    """

    def __init__(
        self,
        patch_size: int | tuple[int, int] = 4,
        input_shape: ShapeSpec | None = None,
        embed_dim: int = 96,
        depths: list[int] = [2, 2, 6, 2],
        num_heads: list[int] = [3, 6, 12, 24],
        window_size: int = 7,
        shift_on_small_feature: bool = True,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        norm: str | dict = "LN",
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        out_features: list[str] | None = None,
        freeze_at: int = 0,
        num_classes: int | None = None,
        dropout: float = 0.0,
    ):
        """
        Args:
            patch_size (int | tuple(int)): Patch size. Default: 4.
            input_shape (ShapeSpec, optional): shape of input image.
                If None, input image has 3 channels by default.
            embed_dim (int): Number of linear projection output channels. Default: 96.
            depths (tuple[int]): Depths of each Swin Transformer stage.
            num_heads (tuple[int]): Number of attention head of each stage.
            window_size (int): Window size. Default: 7.
            shift_on_small_feature (bool): If True, keep shifted-window attention enabled when
                the shortest feature edge is no larger than the window. Default: True.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
            qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
            drop_rate (float): Dropout rate.
            attn_drop_rate (float): Attention dropout rate. Default: 0.
            drop_path_rate (float): Stochastic depth rate. Default: 0.2.
            norm (str, dict): Normalization layer. Default: LN.
            patch_norm (bool): If True, add normalization after patch embedding. Default: True.
            use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
            out_features (list[str]): Output from which stages.
            freeze_at (int): Freeze patch_embed at 1 and stages through the given index at 2 or higher.
                Default: 0.
            num_classes (int): Number of classification classes. No classifier is created if None.
            dropout (float): Dropout probability before the classification layer.
        """
        super().__init__()

        if input_shape is None:
            input_shape = ShapeSpec(channels=3)

        self.num_classes = num_classes

        if out_features is not None:
            num_stages = max([{"stage2": 1, "stage3": 2, "stage4": 3, "stage5": 4}.get(f, 0) for f in out_features])
            depths = depths[:num_stages]
            num_heads = num_heads[:num_stages]

        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=input_shape.channels,
            embed_dim=embed_dim,
            norm=norm if self.patch_norm else "",
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.stages = []
        for i in range(self.num_stages):
            name = f"stage{i + 2}"
            stage = BasicLayer(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm=norm,
                downsample=PatchMerging if (i < self.num_stages - 1) else None,
                use_checkpoint=use_checkpoint,
                shift_on_small_feature=shift_on_small_feature,
            )
            self.add_module(name, stage)
            self.stages.append(name)

            # add a norm layer for each output
            if (out_features is not None and name in out_features) or (out_features is None and i == self.num_stages - 1):
                layer = get_norm(norm, int(embed_dim * 2**i))
                self.add_module(f"norm{i + 2}", layer)

            self._out_feature_strides[name] = 2 ** (i + 2)
            self._out_feature_channels[name] = embed_dim * 2**i

        num_features = [int(embed_dim * 2**i) for i in range(self.num_stages)]
        self.num_features = num_features

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(p=dropout)
            self.linear = nn.Linear(num_features[-1], num_classes)

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

        self.apply(self._init_weights)

    @staticmethod
    def build(name, **kwargs):
        predefined = {
            "SwinTransformer-Tiny": swin_transformer_tiny,
            "SwinTransformer-Small": swin_transformer_small,
            "SwinTransformer-Base": swin_transformer_base,
            "SwinTransformer-Large": swin_transformer_large,
        }

        if name not in predefined:
            raise ValueError(f"`{name}` is not predefined for SwinTransformer. Available names: {', '.join(predefined)}.")

        return predefined[name](**kwargs)

    def freeze(self, freeze_at: int = 0):
        if freeze_at >= 1:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if freeze_at >= 2:
            self.pos_drop.eval()

        for idx, stage_name in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                stage = getattr(self, stage_name)
                stage.eval()
                for param in stage.parameters():
                    param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def padding_constraints(self) -> dict[str, int]:
        return {
            "size_divisiblity": max(self._out_feature_strides.values()),
            "square_size": 0,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outputs = {}
        for i, stage_name in enumerate(self.stages):
            stage = getattr(self, stage_name)
            x_out, H, W, x, Wh, Ww = stage(x, Wh, Ww)

            if stage_name in self._out_features:
                norm_layer = getattr(self, f"norm{i + 2}")
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outputs[stage_name] = out

        if self.num_classes is not None:
            x = self.avgpool(x_out.transpose(1, 2))
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.linear(x)
            outputs["linear"] = x

        return outputs


@validate_factory_kwargs("embed_dim", "depths", "num_heads")
def swin_transformer_tiny(**kwargs):
    return SwinTransformer(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], **kwargs)


@validate_factory_kwargs("embed_dim", "depths", "num_heads")
def swin_transformer_small(**kwargs):
    return SwinTransformer(embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], **kwargs)


@validate_factory_kwargs("embed_dim", "depths", "num_heads")
def swin_transformer_base(**kwargs):
    return SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], **kwargs)


@validate_factory_kwargs("embed_dim", "depths", "num_heads")
def swin_transformer_large(**kwargs):
    return SwinTransformer(embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], **kwargs)
