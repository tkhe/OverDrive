import math

import torch
import torch.nn as nn
from overdrive.layers import FFN
from overdrive.layers import MLP
from overdrive.layers import get_norm
from overdrive.layers import inverse_sigmoid
from overdrive.ops import MultiScaleDeformableAttention
from torch.utils import checkpoint

__all__ = [
    "DINOTransformer",
    "DINOTransformerDecoder",
    "DINOTransformerEncoder",
]


def box_position_embedding(boxes: torch.Tensor, num_feats: int):
    """
    Encode normalized CXCYWH boxes in the DINO order (y, x, w, h).
    """
    frequencies = torch.arange(num_feats, device=boxes.device, dtype=torch.float32)
    frequencies = 10000 ** (2 * torch.div(frequencies, 2, rounding_mode="floor") / num_feats)
    angles = boxes[..., [1, 0, 2, 3], None] * (2 * math.pi) / frequencies
    return torch.stack((angles[..., 0::2].sin(), angles[..., 1::2].cos()), dim=-1).flatten(-3)


class DINOTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4, attn_dropout=0.0, feedforward_channels=2048, ffn_dropout=0.0, norm="LN", activation="ReLU"):
        super().__init__()

        self.attn = MultiScaleDeformableAttention(embed_dim=embed_dim, num_heads=num_heads, num_levels=num_levels, num_points=num_points, dropout=attn_dropout, batch_first=True)
        self.norm1 = get_norm(norm, embed_dim)
        self.ffn = FFN(embed_dim, feedforward_channels, embed_dim, activation=activation, drop=ffn_dropout)
        self.norm2 = get_norm(norm, embed_dim)

    def forward(self, query, query_pos, key_padding_mask, reference_points, spatial_shapes, level_start_index):
        query = self.norm1(
            self.attn(
                query=query,
                value=query,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )
        )
        return self.norm2(query + self.ffn(query))


class DINOTransformerEncoder(nn.Module):
    """
    Multi-scale DINO encoder, composed directly from deformable attention and FFNs.
    """

    def __init__(
        self,
        num_layers=6,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        attn_dropout=0.0,
        feedforward_channels=2048,
        ffn_dropout=0.0,
        norm="LN",
        activation="ReLU",
        use_checkpoint=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_feature_levels = num_levels
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList(
            [DINOTransformerEncoderLayer(embed_dim, num_heads, num_levels, num_points, attn_dropout, feedforward_channels, ffn_dropout, norm, activation) for _ in range(num_layers)]
        )

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios):
        references = []
        for level, (height, width) in enumerate(spatial_shapes.tolist()):
            y, x = torch.meshgrid(
                torch.arange(height, device=valid_ratios.device, dtype=torch.float32) + 0.5,
                torch.arange(width, device=valid_ratios.device, dtype=torch.float32) + 0.5,
                indexing="ij",
            )
            coordinates = torch.stack((x, y), -1).reshape(1, -1, 2)
            scale = valid_ratios[:, level] * valid_ratios.new_tensor([width, height])
            references.append(coordinates / scale[:, None].clamp(min=1e-6))
        return torch.cat(references, 1)[:, :, None] * valid_ratios[:, None]

    def forward(self, query, query_pos, key_padding_mask, reference_points, spatial_shapes, level_start_index):
        for layer in self.layers:
            if self.use_checkpoint:
                query = checkpoint.checkpoint(layer, query, query_pos, key_padding_mask, reference_points, spatial_shapes, level_start_index, use_reentrant=False)
            else:
                query = layer(query, query_pos, key_padding_mask, reference_points, spatial_shapes, level_start_index)
        return query


class DINOTransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4, attn_dropout=0.0, feedforward_channels=2048, ffn_dropout=0.0, norm="LN", activation="ReLU"):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.norm1 = get_norm(norm, embed_dim)
        self.cross_attn = MultiScaleDeformableAttention(embed_dim=embed_dim, num_heads=num_heads, num_levels=num_levels, num_points=num_points, dropout=attn_dropout, batch_first=True)
        self.norm2 = get_norm(norm, embed_dim)
        self.ffn = FFN(embed_dim, feedforward_channels, embed_dim, activation=activation, drop=ffn_dropout)
        self.norm3 = get_norm(norm, embed_dim)

    def forward(self, query, value, query_pos, key_padding_mask, reference_points, spatial_shapes, level_start_index, self_attn_mask=None, cross_attn_mask=None):
        positioned_query = query + query_pos
        attended = self.self_attn(positioned_query, positioned_query, query, attn_mask=self_attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        query = self.norm1(query + attended)
        query = self.norm2(
            self.cross_attn(
                query=query,
                value=value,
                query_pos=query_pos,
                key_padding_mask=cross_attn_mask,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )
        )
        return self.norm3(query + self.ffn(query))


class DINOTransformerDecoder(nn.Module):
    """
    Dynamic anchor queries and Look Forward Twice reference refinement.
    """

    def __init__(
        self,
        num_layers=6,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        attn_dropout=0.0,
        feedforward_channels=2048,
        ffn_dropout=0.0,
        norm="LN",
        activation="ReLU",
        use_checkpoint=False,
    ):
        super().__init__()

        if embed_dim % 4:
            raise ValueError("embed_dim must be divisible by four for box positional encoding")

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_feature_levels = num_levels
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList(
            [DINOTransformerDecoderLayer(embed_dim, num_heads, num_levels, num_points, attn_dropout, feedforward_channels, ffn_dropout, norm, activation) for _ in range(num_layers)]
        )
        self.ref_point_head = MLP(embed_dim * 2, embed_dim, embed_dim, activation="ReLU")
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, value, key_padding_mask, reference_points, spatial_shapes, level_start_index, valid_ratios, reg_branches, self_attn_mask=None, query_padding_mask=None):
        if reference_points.shape[-1] != 4 or reg_branches is None:
            raise ValueError("DINO requires four-dimensional reference boxes and regression branches")

        states, references = [], [reference_points]
        for layer_id, layer in enumerate(self.layers):
            reference_input = reference_points[:, :, None] * torch.cat((valid_ratios, valid_ratios), -1)[:, None]
            query_pos = self.ref_point_head(box_position_embedding(reference_input[:, :, 0], num_feats=self.embed_dim // 2))
            layer_kwargs = dict(
                value=value,
                query_pos=query_pos,
                key_padding_mask=query_padding_mask,
                reference_points=reference_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=key_padding_mask,
            )
            if self.use_checkpoint:
                query = checkpoint.checkpoint(layer, query, use_reentrant=False, **layer_kwargs)
            else:
                query = layer(query, **layer_kwargs)
            refined = (reg_branches[layer_id](query) + inverse_sigmoid(reference_points, eps=1e-3)).sigmoid()
            # Attention in the next layer stops gradients through coordinates, while its
            # prediction loss also supervises this layer's refinement (Look Forward Twice).
            reference_points = refined.detach()
            references.append(refined)
            states.append(self.norm(query))
        return torch.stack(states), references


class DINOTransformer(nn.Module):
    """
    Independent DINO transformer and multi-scale feature preparation.
    """

    def __init__(self, in_features: list[str], encoder: DINOTransformerEncoder, decoder: DINOTransformerDecoder):
        super().__init__()

        if encoder.embed_dim != decoder.embed_dim:
            raise ValueError("encoder and decoder embedding dimensions must agree")
        if len(in_features) != encoder.num_feature_levels or len(in_features) != decoder.num_feature_levels:
            raise ValueError("input feature count must agree with encoder and decoder attention levels")
        if min(encoder.num_layers, decoder.num_layers) < 1:
            raise ValueError("encoder and decoder must each have at least one layer")

        self.in_features = in_features
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = encoder.embed_dim
        self.num_feature_levels = len(in_features)
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

        # Restore radial sampling offsets after the generic transformer initialization.
        for module in self.modules():
            if isinstance(module, MultiScaleDeformableAttention):
                module.init_weights()

    def flatten_features(self, features, position_embeddings, level_embed):
        flattened, positions = [], []
        for level, name in enumerate(self.in_features):
            flattened.append(features[name].flatten(2).transpose(1, 2))
            positions.append(position_embeddings[level].flatten(2).transpose(1, 2) + level_embed[level])
        return torch.cat(flattened, 1), torch.cat(positions, 1)

    def forward(self, query_embed, memory, key_padding_mask, valid_ratios, spatial_shapes, level_start_index, reference_points, reg_branches, self_attn_mask=None):
        return self.decoder(
            query=query_embed,
            value=memory,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            self_attn_mask=self_attn_mask,
        )
