import torch
import torch.nn as nn
from torch.utils import checkpoint

from overdrive.layers import FFN
from overdrive.layers import get_norm
from overdrive.layers import inverse_sigmoid
from overdrive.ops import MultiScaleDeformableAttention

from .detr_transformer import MultiHeadAttention

__all__ = [
    "DeformableDETRTransformer",
    "DeformableDETRTransformerDecoder",
    "DeformableDETRTransformerEncoder",
]


class DeformableDETRTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        attn_dropout: float = 0.0,
        feedforward_channels: int = 1024,
        ffn_dropout: float = 0.0,
        norm: str | dict = "LN",
        activation: str | dict = "ReLU",
    ):
        super().__init__()

        self.attn = MultiScaleDeformableAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=attn_dropout,
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
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ):
        query = self.attn(
            query=query,
            value=query,
            query_pos=query_pos,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        query = self.norm1(query)
        query = self.ffn(query) + query
        query = self.norm2(query)
        return query


class DeformableDETRTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
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
        self.cross_attn = MultiScaleDeformableAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=attn_dropout,
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
        value: torch.Tensor,
        query_pos: torch.Tensor,
        key_padding_mask: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
    ):
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            key_padding_mask=key_padding_mask,
        )
        query = self.norm1(query)
        query = self.cross_attn(
            query=query,
            value=value,
            query_pos=query_pos,
            key_padding_mask=cross_attn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        query = self.norm2(query)
        query = self.ffn(query) + query
        query = self.norm3(query)
        return query


class DeformableDETRTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
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
            DeformableDETRTransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                attn_dropout=attn_dropout,
                feedforward_channels=feedforward_channels,
                ffn_dropout=ffn_dropout,
                norm=norm,
                activation=activation,
            )
            for _ in range(num_layers)
        ]
        self.layers = nn.ModuleList(layers)

    @staticmethod
    def get_reference_points(spatial_shapes: torch.Tensor, valid_ratios: torch.Tensor) -> torch.Tensor:
        """
        Get the reference points used by the encoder self attention.

        Args:
            spatial_shapes (Tensor): Spatial shapes of features in all levels, with shape `(num_levels, 2)`.
                The last dimension represents `(h, w)`.
            valid_ratios (Tensor): The ratios of the valid width and the valid height relative to the
                width and the height of features in all levels, with shape `(bs, num_levels, 2)`.

        Returns:
            Tensor: Reference points with shape `(bs, sum(h_l * w_l), num_levels, 2)`. All elements are
            in range [0, 1].
        """
        # the offset_normalizer normalization: image border (0.5 / H) is mapped to the level border
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, int(H), dtype=torch.float32, device=valid_ratios.device),
                torch.linspace(0.5, W - 0.5, int(W), dtype=torch.float32, device=valid_ratios.device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H + 1e-6)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W + 1e-6)
            reference_points_list.append(torch.stack((ref_x, ref_y), -1))
        reference_points = torch.cat(reference_points_list, 1)
        # [bs, sum(hw), num_level, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        key_padding_mask: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ):
        for layer in self.layers:
            if self.use_checkpoint:
                query = checkpoint.checkpoint(
                    layer,
                    query,
                    query_pos,
                    key_padding_mask,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    use_reentrant=False,
                )
            else:
                query = layer(
                    query=query,
                    query_pos=query_pos,
                    key_padding_mask=key_padding_mask,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                )
        return query


class DeformableDETRTransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        attn_dropout: float = 0.0,
        feedforward_channels: int = 1024,
        ffn_dropout: float = 0.0,
        norm: str | dict = "LN",
        activation: str | dict = "ReLU",
        return_intermediate: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.use_checkpoint = use_checkpoint

        layers = [
            DeformableDETRTransformerDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
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
        value: torch.Tensor,
        key_padding_mask: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        reg_branches: list[nn.Module] | None = None,
        self_attn_mask: torch.Tensor | None = None,
        query_padding_mask: torch.Tensor | None = None,
    ):
        output = query
        intermediate = []
        intermediate_reference_points = []

        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                if reference_points.shape[-1] != 2:
                    raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]} instead.")
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            if self.use_checkpoint:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    value,
                    query_pos=query_pos,
                    key_padding_mask=query_padding_mask,
                    reference_points=reference_points_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    self_attn_mask=self_attn_mask,
                    cross_attn_mask=key_padding_mask,
                    use_reentrant=False,
                )
            else:
                output = layer(
                    output,
                    value=value,
                    query_pos=query_pos,
                    key_padding_mask=query_padding_mask,
                    reference_points=reference_points_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    self_attn_mask=self_attn_mask,
                    cross_attn_mask=key_padding_mask,
                )

            if reg_branches is not None:
                tmp_reg_preds = reg_branches[layer_id](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = (tmp_reg_preds + inverse_sigmoid(reference_points)).sigmoid()
                else:
                    new_reference_points = tmp_reg_preds.clone()
                    new_reference_points[..., :2] += inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output.unsqueeze(0), reference_points.unsqueeze(0)


class DeformableDETRTransformer(nn.Module):
    def __init__(
        self,
        in_features: list[str],
        encoder: DeformableDETRTransformerEncoder,
        decoder: DeformableDETRTransformerDecoder,
        num_feature_levels: int | None = None,
    ):
        super().__init__()

        if num_feature_levels is None:
            num_feature_levels = len(in_features)

        if num_feature_levels != len(in_features):
            raise ValueError(f"num_feature_levels ({num_feature_levels}) must match len(in_features) ({len(in_features)})")

        self.encoder = encoder
        self.decoder = decoder

        self.embed_dim = self.encoder.embed_dim
        self.in_features = in_features
        self.num_feature_levels = num_feature_levels

        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

        # The sampling grid of `MultiScaleDeformableAttention` is overwritten by the
        # xavier initialization above, so it is restored once all parameters exist.
        for module in self.modules():
            if isinstance(module, MultiScaleDeformableAttention):
                module.init_weights()

    def flatten_features(
        self,
        features: dict[str, torch.Tensor],
        position_embeddings: dict[str, torch.Tensor] | list[torch.Tensor],
        level_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Flatten and concatenate the multi-level features along the spatial dimension.

        Args:
            features (dict[str, Tensor]): Multi-level features from the neck, each with shape
                `(bs, c, h_lvl, w_lvl)`.
            position_embeddings (dict[str, Tensor] | list[Tensor]): The position embeddings of each
                level listed in `in_features`, each with shape `(bs, c, h_lvl, w_lvl)`.
            level_embed (Tensor): The learnable level embedding with shape `(num_levels, c)`.

        Returns:
            tuple[Tensor, Tensor]: The flattened features and position embeddings, both with shape
            `(bs, sum(h_lvl * w_lvl), c)`.
        """
        feat_flatten = []
        pos_embed_flatten = []
        for lvl, in_feature in enumerate(self.in_features):
            feat = features[in_feature]
            pos_embed = position_embeddings[lvl]
            batch_size, c, _, _ = feat.shape
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat_flatten.append(feat.view(batch_size, c, -1).permute(0, 2, 1))
            pos_embed_flatten.append(pos_embed.view(batch_size, c, -1).permute(0, 2, 1) + level_embed[lvl].view(1, 1, -1))

        return torch.cat(feat_flatten, 1), torch.cat(pos_embed_flatten, 1)

    def forward(
        self,
        query_embed: torch.Tensor,
        query_pos_embed: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: torch.Tensor,
        valid_ratios: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        reference_points: torch.Tensor,
        reg_branches: list[nn.Module] | None = None,
        self_attn_mask: torch.Tensor | None = None,
        query_padding_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            query_embed (Tensor): The content queries with shape `(bs, num_queries, embed_dim)`.
            query_pos_embed (Tensor): The positional queries with shape `(bs, num_queries, embed_dim)`.
            memory (Tensor): The output embeddings of the encoder with shape
                `(bs, sum(h_lvl * w_lvl), embed_dim)`.
            key_padding_mask (Tensor): ByteTensor of shape `(bs, sum(h_lvl * w_lvl))` where non-zero
                values represent ignored positions. It is used by the decoder cross attention.
            valid_ratios (Tensor): The ratios of the valid width and the valid height relative to the
                width and the height of features in all levels, with shape `(bs, num_levels, 2)`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels, with shape
                `(num_levels, 2)`.
            level_start_index (Tensor): The start index of each level with shape `(num_levels, )`.
            reference_points (Tensor): The reference points of the decoder, with shape
                `(bs, num_queries, 4)` arranged as `(cx, cy, w, h)` when the detector is two-stage,
                otherwise `(bs, num_queries, 2)` arranged as `(cx, cy)`.
            reg_branches (list[nn.Module]): The regression branches used to refine the reference
                points layer by layer. Only passed when box refinement is enabled.
            self_attn_mask (Tensor): ByteTensor with shape `(num_queries, num_queries)`. Default: None.
            query_padding_mask (Tensor): ByteTensor with shape `(bs, num_queries)`. Default: None.

        Returns:
            tuple[Tensor, list[Tensor]]: The hidden states of all decoder layers with shape
            `(num_layers, bs, num_queries, embed_dim)`, and the list of reference points whose first
            element is the initial reference and the remaining ones are the intermediate references.
        """
        hidden_states, inter_references = self.decoder(
            query=query_embed,
            query_pos=query_pos_embed,
            value=memory,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            self_attn_mask=self_attn_mask,
            query_padding_mask=query_padding_mask,
        )

        return hidden_states, [reference_points, *inter_references]
