import logging

import torch
import torch.nn as nn
from torch.utils import checkpoint

from overdrive.layers import FFN
from overdrive.layers import get_norm

__all__ = [
    "DETRTransformer",
    "DETRTransformerDecoder",
    "DETRTransformerEncoder",
]

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """
    A wrapper for `torch.nn.MultiheadAttention`.

    implement MultiHeadAttention with identity connection,
    and position embedding is also passed as input.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        batch_first: bool = False,
    ):
        """
        Args:
            embed_dim (int): the embedding dimension for attention.
            num_heads (int): the number of attention head.
            attn_drop (float): A Dropout layer on attn_output_weights. Default: 0.0
            proj_drop (float): A Dropout layer after `MultiHeadAttention`. Default: 0.0
            batch_first (bool): If True, then the input and output tensors will be provided as (B, N, C).
                Default: False.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=batch_first,
        )
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
        """
        Args:
            query (Tensor): query embedding with shape `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`.
            key (Tensor): key embedding with shape `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`.
            value (Tensor): value embedding with the same shape as `key`. if None, it will be set to `key`.
            identity (Tensor): the tensor, with the same shape as query, will be used for identity addition.
                Default: None. If None, `query` will be used.
            query_pos (Tensor): the position embedding of query with the shape as `query`. Default: None.
            key_pos (Tensor): the position embedding of key with the shape as `key`. Default: None.
            attn_mask (Tensor): ByteTensor mask with shape `(num_query, num_key)`. Default: None.
            key_padding_mask (Tensor): ByteTensor mask with shape `(bs, num_key)` which indicates
                which elements within `key` to be ignored in attention. Default: None.
        """
        if key is None:
            key = query

        if value is None:
            value = key

        if identity is None:
            identity = query

        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    logger.warning(f"position encoding of key is missing in {self.__class__.__name__}.")

        if query_pos is not None:
            query = query + query_pos

        if key_pos is not None:
            key = key + key_pos

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]

        return identity + self.proj_drop(out)


class DETRTransformerEncoderLayer(nn.Module):
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


class DETRTransformerDecoderLayer(nn.Module):
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

        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_dropout,
            batch_first=True,
        )
        self.norm1 = get_norm(norm, embed_dim)
        self.cross_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_dropout,
            batch_first=True,
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
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        key_pos: torch.Tensor | None = None,
        self_attn_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ):
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
        )
        query = self.norm1(query)
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
        )
        query = self.norm2(query)
        query = self.ffn(query) + query
        query = self.norm3(query)
        return query


class DETRTransformerEncoder(nn.Module):
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
            DETRTransformerEncoderLayer(
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


class DETRTransformerDecoder(nn.Module):
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
            DETRTransformerDecoderLayer(
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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ):
        intermediate = []
        for layer in self.layers:
            if self.use_checkpoint:
                query = checkpoint.checkpoint(
                    layer,
                    query,
                    key=key,
                    value=value,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    key_padding_mask=key_padding_mask,
                    use_reentrant=False,
                )
            else:
                query = layer(
                    query,
                    key=key,
                    value=value,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    key_padding_mask=key_padding_mask,
                )

            if self.post_norm is not None:
                intermediate.append(self.post_norm(query))
            else:
                intermediate.append(query)

        return torch.stack(intermediate)


class DETRTransformer(nn.Module):
    def __init__(
        self,
        in_features: list[str],
        encoder: DETRTransformerEncoder,
        decoder: DETRTransformerDecoder,
    ):
        super().__init__()

        if len(in_features) != 1:
            raise ValueError(f"DETRTransformer requires exactly one input feature, but got {len(in_features)}.")

        self.encoder = encoder
        self.decoder = decoder

        self.embed_dim = self.encoder.embed_dim
        self.in_features = in_features

        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
        query_embed: torch.Tensor,
        pos_embed: torch.Tensor,
    ):
        # DETR only use single level feature, so we only take the first element in `in_features`.
        x = features[self.in_features[0]]

        N, C, H, W = x.shape
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
        decoder_output = self.decoder(
            query=target,
            key=memory,
            value=memory,
            query_pos=query_embed,
            key_pos=pos_embed,
            key_padding_mask=mask,
        )
        memory = memory.permute(0, 2, 1).reshape(N, C, H, W)
        return decoder_output, memory
