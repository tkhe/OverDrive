import logging
import math

import torch
import torch.nn as nn
from torch.utils import checkpoint

from overdrive.layers import FFN
from overdrive.layers import MLP
from overdrive.layers import get_norm

from .detr_transformer import MultiHeadAttention

__all__ = [
    "ConditionalDETRTransformer",
    "ConditionalDETRTransformerDecoder",
    "ConditionalDETRTransformerEncoder",
]

logger = logging.getLogger(__name__)


class ConditionalSelfAttention(nn.Module):
    """
    Conditional Self-Attention proposed in Conditional DETR.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_drop: float = 0,
        proj_drop: float = 0,
    ):
        """
        Args:
            embed_dim (int): the embedding dimension for attention.
            num_heads (int): the number of attention head.
            attn_drop (float): A Dropout layer on attn_output_weights. Default: 0.0
            proj_drop (float): A Dropout layer after `MultiHeadAttention`. Default: 0.0
        """
        super().__init__()

        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5

        self.query_content_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.key_content_proj = nn.Linear(embed_dim, embed_dim)
        self.key_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        identity: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        key_pos: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ):
        if key is None:
            key = query

        if value is None:
            value = key

        if identity is None:
            identity = query

        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    logger.warning(f"position encoding of key is missing in {self.__class__.__name__}.")

        if query_pos is None or key_pos is None:
            raise ValueError("query_pos and key_pos must be passed into ConditionalSelfAttention")

        # query/key/value content and position embedding projection
        query_content = self.query_content_proj(query)
        query_pos = self.query_pos_proj(query_pos)
        key_content = self.key_content_proj(key)
        key_pos = self.key_pos_proj(key_pos)
        value = self.value_proj(value)

        B, N, C = query_content.shape
        q = query_content + query_pos
        k = key_content + key_pos
        v = value

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(attn_mask, float("-inf"))
            else:
                attn += attn_mask

        if key_padding_mask is not None:
            attn.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        return identity + self.proj_drop(out)


class ConditionalCrossAttention(nn.Module):
    """
    Conditional Cross-Attention proposed in Conditional DETR.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """
        Args:
            embed_dim (int): the embedding dimension for attention.
            num_heads (int): the number of attention head.
            attn_drop (float): A Dropout layer on attn_output_weights. Default: 0.0
            proj_drop (float): A Dropout layer after `MultiHeadAttention`. Default: 0.0
        """
        super().__init__()

        self.num_heads = num_heads

        self.query_content_proj = nn.Linear(embed_dim, embed_dim)
        self.key_content_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.key_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_sine_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor | None = None,
        identity: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        key_pos: torch.Tensor | None = None,
        query_sine_embed: torch.Tensor | None = None,
        is_first: bool = False,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ):
        if value is None:
            value = key

        if identity is None:
            identity = query

        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    logger.warning(f"position encoding of key is missing in {self.__class__.__name__}.")

        if query_pos is None or key_pos is None:
            raise ValueError("query_pos and key_pos must be passed into ConditionalCrossAttention")

        # content projection
        query_content = self.query_content_proj(query)
        key_content = self.key_content_proj(key)
        value = self.value_proj(value)

        B, N, C = query_content.shape
        _, HW, _ = key_content.shape

        key_pos = self.key_pos_proj(key_pos)
        if is_first:
            query_pos = self.query_pos_proj(query_pos)
            q = query_content + query_pos
            k = key_content + key_pos
        else:
            q = query_content
            k = key_content
        v = value

        q = q.view(B, N, self.num_heads, C // self.num_heads)
        query_sine_embed = self.query_pos_sine_proj(query_sine_embed).view(B, N, self.num_heads, C // self.num_heads)
        q = torch.cat([q, query_sine_embed], dim=3).view(B, N, C * 2)

        k = k.view(B, HW, self.num_heads, C // self.num_heads)
        key_pos = key_pos.view(B, HW, self.num_heads, C // self.num_heads)
        k = torch.cat([k, key_pos], dim=3).view(B, HW, C * 2)

        q = q.reshape(B, N, self.num_heads, C * 2 // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, HW, self.num_heads, C * 2 // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, HW, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        scale = (C * 2 // self.num_heads) ** -0.5
        q = q * scale

        attn = q @ k.transpose(-2, -1)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(attn_mask, float("-inf"))
            else:
                attn += attn_mask

        if key_padding_mask is not None:
            attn.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        return identity + self.proj_drop(out)


def coordinate_to_embedding(coord_tensor: torch.Tensor, embed_dim: int = 128, temperature: int = 10000, scale: float | None = None):
    """
    Convert coordinate tensor to positional encoding.

    Args:
        coord_tensor (Tensor): Coordinate tensor to be converted to
            positional encoding. With the last dimension as 2 or 4.
        num_feats (int, optional): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value. Defaults to 128.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
    Returns:
        Tensor: Returned encoded positional tensor.
    """
    if scale is None:
        scale = 2 * math.pi

    dim_t = torch.arange(embed_dim, dtype=torch.float32, device=coord_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / embed_dim)
    x_embed = coord_tensor[..., 0] * scale
    y_embed = coord_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(2)

    if coord_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=-1)
    elif coord_tensor.size(-1) == 4:
        w_embed = coord_tensor[..., 2] * scale
        pos_w = w_embed[..., None] / dim_t
        pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=-1).flatten(2)

        h_embed = coord_tensor[..., 3] * scale
        pos_h = h_embed[..., None] / dim_t
        pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=-1).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1): {coord_tensor.size(-1)}")
    return pos


class ConditionalDETRTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_channels: int = 1024,
        ffn_dropout: float = 0.0,
        norm: str | dict = "LN",
        activation: str | dict = "ReLU",
    ):
        super().__init__()

        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_dropout,
            batch_first=True,
        )
        self.norm1 = get_norm(norm, embed_dim)
        self.ffn = FFN(
            in_features=embed_dim,
            hidden_features=feedforward_channels,
            out_features=embed_dim,
            activation=activation,
            drop=ffn_dropout,
        )
        self.norm2 = get_norm(norm, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ):
        query = self.attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=key_padding_mask,
        )
        query = self.norm1(query)
        query = self.ffn(query) + query
        query = self.norm2(query)
        return query


class ConditionalDETRTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_channels: int = 1024,
        ffn_dropout: float = 0.0,
        norm: str | dict = "LN",
        activation: str | dict = "ReLU",
    ):
        super().__init__()

        self.self_attn = ConditionalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_dropout,
        )
        self.norm1 = get_norm(norm, embed_dim)
        self.cross_attn = ConditionalCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_dropout,
        )
        self.norm2 = get_norm(norm, embed_dim)
        self.ffn = FFN(
            in_features=embed_dim,
            hidden_features=feedforward_channels,
            out_features=embed_dim,
            activation=activation,
            drop=ffn_dropout,
        )
        self.norm3 = get_norm(norm, embed_dim)

    def forward(
        self,
        query,
        key,
        query_pos,
        key_pos,
        key_padding_mask,
        query_sine_embed,
        is_first,
    ):
        query = self.self_attn(
            query,
            key=query,
            query_pos=query_pos,
            key_pos=query_pos,
        )
        query = self.norm1(query)
        query = self.cross_attn(
            query,
            key=key,
            query_pos=query_pos,
            key_pos=key_pos,
            key_padding_mask=key_padding_mask,
            query_sine_embed=query_sine_embed,
            is_first=is_first,
        )
        query = self.norm2(query)
        query = self.ffn(query) + query
        query = self.norm3(query)
        return query


class ConditionalDETRTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_channels: int = 1024,
        ffn_dropout: float = 0.0,
        norm: str | dict = "LN",
        activation: str | dict = "ReLU",
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint

        layers = [
            ConditionalDETRTransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                feedforward_channels=feedforward_channels,
                ffn_dropout=ffn_dropout,
                norm=norm,
                activation=activation,
            )
            for _ in range(num_layers)
        ]

        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ):
        for layer in self.layers:
            if self.use_checkpoint:
                query = checkpoint.checkpoint(layer, query, query_pos, key_padding_mask, use_reentrant=False)
            else:
                query = layer(query, query_pos, key_padding_mask)
        return query


class ConditionalDETRTransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_channels: int = 1024,
        ffn_dropout: float = 0.0,
        norm: str | dict = "LN",
        activation: str | dict = "ReLU",
        post_norm: str | dict = "LN",
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint

        layers = [
            ConditionalDETRTransformerDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                feedforward_channels=feedforward_channels,
                ffn_dropout=ffn_dropout,
                norm=norm,
                activation=activation,
            )
            for _ in range(num_layers)
        ]
        self.layers = nn.ModuleList(layers)
        self.post_norm = get_norm(post_norm, embed_dim)
        self.query_scale = MLP(
            in_features=self.embed_dim,
            hidden_features=self.embed_dim,
            out_features=self.embed_dim,
            activation=activation,
        )
        self.ref_point_head = MLP(
            in_features=self.embed_dim,
            hidden_features=self.embed_dim,
            out_features=2,
            activation=activation,
        )

        for layer_id in range(num_layers - 1):
            self.layers[layer_id + 1].cross_attn.query_pos_proj = None

    def forward(
        self,
        query,
        key,
        query_pos,
        key_pos,
        key_padding_mask,
    ):
        reference_unsigmoid = self.ref_point_head(query_pos)
        reference = reference_unsigmoid.sigmoid()
        reference_xy = reference[..., :2]
        intermediate = []
        for layer_id, layer in enumerate(self.layers):
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(query)

            query_sine_embed = coordinate_to_embedding(
                coord_tensor=reference_xy,
                embed_dim=self.embed_dim // 2,
            )
            query_sine_embed = query_sine_embed * pos_transformation
            if self.use_checkpoint:
                query = checkpoint.checkpoint(
                    layer,
                    query,
                    key=key,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    key_padding_mask=key_padding_mask,
                    query_sine_embed=query_sine_embed,
                    is_first=(layer_id == 0),
                    use_reentrant=False,
                )
            else:
                query = layer(
                    query,
                    key=key,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    key_padding_mask=key_padding_mask,
                    query_sine_embed=query_sine_embed,
                    is_first=(layer_id == 0),
                )
            if self.post_norm is not None:
                intermediate.append(self.post_norm(query))
            else:
                intermediate.append(query)

        return torch.stack(intermediate), reference


class ConditionalDETRTransformer(nn.Module):
    def __init__(
        self,
        in_features: list[str],
        encoder: ConditionalDETRTransformerEncoder,
        decoder: ConditionalDETRTransformerDecoder,
    ):
        super().__init__()

        if len(in_features) != 1:
            raise ValueError(f"ConditionalDETRTransformer requires exactly one input feature, but got {len(in_features)}.")

        self.encoder = encoder
        self.decoder = decoder

        self.embed_dim = self.encoder.embed_dim
        self.in_features = in_features

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        query_embed: torch.Tensor,
        pos_embed: torch.Tensor,
    ):
        x = features[self.in_features[0]]

        N, C, _, _ = x.shape
        # [N, C, H, W] -> [N, HxW, C]
        x = x.view(N, C, -1).permute(0, 2, 1)
        pos_embed = pos_embed.view(N, C, -1).permute(0, 2, 1)
        query_embed = query_embed.unsqueeze(0).repeat(N, 1, 1)  # [num_query, dim] -> [N, num_query, dim]
        mask = mask.view(N, -1)  # [N, H, W] -> [N, HxW]
        memory = self.encoder(
            query=x,
            query_pos=pos_embed,
            key_padding_mask=mask,
        )
        target = torch.zeros_like(query_embed)
        hidden_states, references = self.decoder(
            query=target,
            key=memory,
            query_pos=query_embed,
            key_pos=pos_embed,
            key_padding_mask=mask,
        )
        return hidden_states, references
