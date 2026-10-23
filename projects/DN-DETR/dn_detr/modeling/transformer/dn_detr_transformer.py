import torch
import torch.nn as nn
from overdrive.layers import FFN
from overdrive.layers import MLP
from overdrive.layers import get_norm
from overdrive.layers import inverse_sigmoid
from overdrive.modeling.transformer.conditional_detr_transformer import ConditionalCrossAttention
from overdrive.modeling.transformer.conditional_detr_transformer import ConditionalSelfAttention
from overdrive.modeling.transformer.conditional_detr_transformer import coordinate_to_embedding
from overdrive.modeling.transformer.detr_transformer import MultiHeadAttention
from torch.utils import checkpoint

__all__ = [
    "DNDETRTransformer",
    "DNDETRTransformerDecoder",
    "DNDETRTransformerEncoder",
]


class DNDETRTransformerEncoderLayer(nn.Module):
    """
    A position-aware encoder layer for DN-DETR.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_channels: int = 2048,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        norm: str | dict = "LN",
        activation: str | dict = "PReLU",
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
            activation=nn.PReLU() if activation == "PReLU" else activation,
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
        return self.norm2(query)


class DNDETRTransformerEncoder(nn.Module):
    """
    Encode one feature sequence with learned position scales.
    """

    def __init__(
        self,
        num_layers: int = 6,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_channels: int = 1024,
        ffn_dropout: float = 0.0,
        norm: str | dict = "LN",
        activation: str | dict = "PReLU",
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        self.query_scale = MLP(
            in_features=embed_dim,
            hidden_features=embed_dim,
            out_features=embed_dim,
            activation="ReLU",
        )
        self.layers = nn.ModuleList(
            DNDETRTransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                feedforward_channels=feedforward_channels,
                ffn_dropout=ffn_dropout,
                norm=norm,
                activation=activation,
            )
            for _ in range(num_layers)
        )

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ):
        for layer in self.layers:
            scaled_pos = query_pos * self.query_scale(query)
            if self.use_checkpoint:
                query = checkpoint.checkpoint(layer, query, scaled_pos, key_padding_mask, use_reentrant=False)
            else:
                query = layer(query, scaled_pos, key_padding_mask)
        return query


class DNDETRTransformerDecoderLayer(nn.Module):
    """
    Decode noisy and matching queries with a shared self-attention mask.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_channels: int = 1024,
        ffn_dropout: float = 0.0,
        norm: str | dict = "LN",
        activation: str | dict = "PReLU",
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
            activation=nn.PReLU() if activation == "PReLU" else activation,
            drop=ffn_dropout,
        )
        self.norm3 = get_norm(norm, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        key_padding_mask: torch.Tensor,
        query_sine_embed: torch.Tensor,
        is_first: bool,
        self_attn_mask: torch.Tensor | None = None,
    ):
        query = self.self_attn(
            query,
            key=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
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


class DNDETRTransformerDecoder(nn.Module):
    """
    Decode DN-DETR queries with an isolated denoising attention mask.
    """

    def __init__(
        self,
        num_layers: int = 6,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_channels: int = 2048,
        ffn_dropout: float = 0.0,
        norm: str | dict = "LN",
        activation: str | dict = "PReLU",
        with_modulate_hw_attn: bool = True,
        post_norm: str | dict = "LN",
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.with_modulate_hw_attn = with_modulate_hw_attn
        self.use_checkpoint = use_checkpoint
        self.ref_point_head = MLP(
            in_features=2 * embed_dim,
            hidden_features=embed_dim,
            out_features=embed_dim,
            activation="ReLU",
        )
        self.query_scale = MLP(
            in_features=embed_dim,
            hidden_features=embed_dim,
            out_features=embed_dim,
            activation="ReLU",
        )
        if with_modulate_hw_attn:
            self.ref_anchor_head = MLP(
                in_features=embed_dim,
                hidden_features=embed_dim,
                out_features=2,
                activation="ReLU",
            )
        self.layers = nn.ModuleList(
            DNDETRTransformerDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                feedforward_channels=feedforward_channels,
                ffn_dropout=ffn_dropout,
                norm=norm,
                activation=activation,
            )
            for _ in range(num_layers)
        )
        self.post_norm = get_norm(post_norm, embed_dim)
        for layer_id in range(num_layers - 1):
            self.layers[layer_id + 1].cross_attn.query_pos_proj = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        key_padding_mask: torch.Tensor,
        reg_branches: nn.Module,
        self_attn_mask: torch.Tensor | None = None,
    ):
        query_dim = query_pos.shape[-1]
        reference_boxes = query_pos.sigmoid()
        intermediate_references = [reference_boxes]
        intermediate = []

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_boxes[..., :query_dim]
            query_sine_embed = coordinate_to_embedding(obj_center, embed_dim=self.embed_dim // 2)
            query_pos_embed = self.ref_point_head(query_sine_embed)
            position_transform = 1 if layer_id == 0 else self.query_scale(query)
            query_sine_embed = query_sine_embed[..., : self.embed_dim] * position_transform

            if self.with_modulate_hw_attn:
                ref_hw = self.ref_anchor_head(query).sigmoid()
                query_sine_embed[..., self.embed_dim // 2 :] *= (ref_hw[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                query_sine_embed[..., : self.embed_dim // 2] *= (ref_hw[..., 1] / obj_center[..., 3]).unsqueeze(-1)

            layer_kwargs = dict(
                query_pos=query_pos_embed,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                query_sine_embed=query_sine_embed,
                is_first=layer_id == 0,
                self_attn_mask=self_attn_mask,
            )
            if self.use_checkpoint:
                query = checkpoint.checkpoint(layer, query, key, use_reentrant=False, **layer_kwargs)
            else:
                query = layer(query, key, **layer_kwargs)

            offsets = reg_branches(query)
            new_reference_boxes = (offsets[..., :query_dim] + inverse_sigmoid(reference_boxes)).sigmoid()
            if layer_id != self.num_layers - 1:
                intermediate_references.append(new_reference_boxes)
            reference_boxes = new_reference_boxes.detach()
            intermediate.append(self.post_norm(query) if self.post_norm is not None else query)

        return torch.stack(intermediate), torch.stack(intermediate_references)


class DNDETRTransformer(nn.Module):
    """
    Encode one feature map and decode matching plus denoising queries.
    """

    def __init__(
        self,
        in_features: list[str],
        encoder: DNDETRTransformerEncoder,
        decoder: DNDETRTransformerDecoder,
    ):
        super().__init__()

        self.embed_dim = encoder.embed_dim
        self.in_features = in_features
        self.encoder = encoder
        self.decoder = decoder

        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
        box_queries: torch.Tensor,
        pos_embed: torch.Tensor,
        target: torch.Tensor,
        reg_branches: nn.Module,
        self_attn_mask: torch.Tensor | None = None,
    ):
        x = features[self.in_features[0]]
        batch_size, channels, _, _ = x.shape
        x = x.view(batch_size, channels, -1).transpose(1, 2)
        pos_embed = pos_embed.view(batch_size, channels, -1).transpose(1, 2)
        mask = mask.flatten(1)

        memory = self.encoder(query=x, query_pos=pos_embed, key_padding_mask=mask)
        return self.decoder(
            query=target,
            key=memory,
            query_pos=box_queries,
            key_pos=pos_embed,
            key_padding_mask=mask,
            reg_branches=reg_branches,
            self_attn_mask=self_attn_mask,
        )
