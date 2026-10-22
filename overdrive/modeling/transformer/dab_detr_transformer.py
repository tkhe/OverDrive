import torch
import torch.nn as nn
from torch.utils import checkpoint

from overdrive.layers import FFN
from overdrive.layers import MLP
from overdrive.layers import get_norm
from overdrive.layers import inverse_sigmoid

from .conditional_detr_transformer import ConditionalCrossAttention
from .conditional_detr_transformer import ConditionalSelfAttention
from .conditional_detr_transformer import coordinate_to_embedding
from .detr_transformer import MultiHeadAttention

__all__ = [
    "DABDETRTransformer",
    "DABDETRTransformerDecoder",
    "DABDETRTransformerEncoder",
]


class DABDETRTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_channels: int = 2048,
        attn_dropout: float = 0.0,
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


class DABDETRTransformerDecoderLayer(nn.Module):
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
        query: torch.Tensor,
        key: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        key_padding_mask: torch.Tensor,
        query_sine_embed: torch.Tensor,
        is_first: bool,
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


class DABDETRTransformerEncoder(nn.Module):
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

        self.query_scale = MLP(
            in_features=embed_dim,
            hidden_features=embed_dim,
            out_features=embed_dim,
            activation="ReLU",
        )

        layers = [
            DABDETRTransformerEncoderLayer(
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
            pos_scales = self.query_scale(query)
            if self.use_checkpoint:
                query = checkpoint.checkpoint(
                    layer,
                    query,
                    query_pos * pos_scales,
                    key_padding_mask,
                    use_reentrant=False,
                )
            else:
                query = layer(
                    query,
                    query_pos * pos_scales,
                    key_padding_mask,
                )
        return query


class DABDETRTransformerDecoder(nn.Module):
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
            in_features=2 * self.embed_dim,
            hidden_features=self.embed_dim,
            out_features=self.embed_dim,
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
                in_features=self.embed_dim,
                hidden_features=self.embed_dim,
                out_features=2,
                activation="ReLU",
            )

        layers = [
            DABDETRTransformerDecoderLayer(
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
    ):
        query_dim = query_pos.shape[-1]
        reference_boxes = query_pos.sigmoid()
        intermediate_reference_boxes = [reference_boxes]

        intermediate = []
        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_boxes[..., :query_dim]
            query_sine_embed = coordinate_to_embedding(
                coord_tensor=obj_center,
                embed_dim=self.embed_dim // 2,
            )
            query_pos = self.ref_point_head(query_sine_embed)

            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(query)

            query_sine_embed = query_sine_embed[..., : self.embed_dim] * pos_transformation

            if self.with_modulate_hw_attn:
                ref_hw = self.ref_anchor_head(query).sigmoid()
                query_sine_embed[..., self.embed_dim // 2 :] *= (ref_hw[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                query_sine_embed[..., : self.embed_dim // 2] *= (ref_hw[..., 1] / obj_center[..., 3]).unsqueeze(-1)

            if self.use_checkpoint:
                query = checkpoint.checkpoint(
                    layer,
                    query,
                    key,
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
                    key,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    key_padding_mask=key_padding_mask,
                    query_sine_embed=query_sine_embed,
                    is_first=(layer_id == 0),
                )

            offsets = reg_branches(query)
            offsets[..., :query_dim] += inverse_sigmoid(reference_boxes)
            new_reference_boxes = offsets[..., :query_dim].sigmoid()

            if layer_id != self.num_layers - 1:
                intermediate_reference_boxes.append(new_reference_boxes)
            reference_boxes = new_reference_boxes.detach()

            if self.post_norm is not None:
                intermediate.append(self.post_norm(query))
            else:
                intermediate.append(query)

        return torch.stack(intermediate), torch.stack(intermediate_reference_boxes)


class DABDETRTransformer(nn.Module):
    def __init__(
        self,
        in_features: list[str],
        encoder: DABDETRTransformerEncoder,
        decoder: DABDETRTransformerDecoder,
        num_patterns: int = 0,
    ):
        super().__init__()

        if len(in_features) != 1:
            raise ValueError(f"DABDETRTransformer requires exactly one input feature, but got {len(in_features)}.")

        self.embed_dim = encoder.embed_dim
        self.in_features = in_features
        self.num_patterns = num_patterns

        self.encoder = encoder
        self.decoder = decoder

        if num_patterns > 0:
            self.patterns = nn.Embedding(num_patterns, self.embed_dim)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
        query_embed: torch.Tensor,
        pos_embed: torch.Tensor,
        reg_branches: nn.Module,
    ):
        x = features[self.in_features[0]]

        N, C, _, _ = x.shape
        # [N, C, H, W] -> [N, HxW, C]
        x = x.view(N, C, -1).permute(0, 2, 1)
        pos_embed = pos_embed.view(N, C, -1).permute(0, 2, 1)
        query_embed = query_embed.unsqueeze(0).repeat(N, 1, 1)
        mask = mask.view(N, -1)
        memory = self.encoder(
            query=x,
            query_pos=pos_embed,
            key_padding_mask=mask,
        )
        num_queries = query_embed.shape[1]

        if self.num_patterns == 0:
            target = pos_embed.new_zeros(N, num_queries, self.embed_dim)
        else:
            target = self.patterns.weight[:, None, None, :].repeat(1, num_queries, N, 1).flatten(0, 1).permute(1, 0, 2)
            query_embed = query_embed.repeat(1, self.num_patterns, 1)

        hidden_states, reference_boxes = self.decoder(
            query=target,
            key=memory,
            query_pos=query_embed,
            key_pos=pos_embed,
            key_padding_mask=mask,
            reg_branches=reg_branches,
        )
        return hidden_states, reference_boxes
