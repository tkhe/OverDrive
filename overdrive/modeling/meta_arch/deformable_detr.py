import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from overdrive.layers import weight_init
from overdrive.modeling.backbone import Backbone
from overdrive.modeling.dense_head import DenseHead
from overdrive.modeling.neck import Neck
from overdrive.modeling.postprocessing import detector_postprocess
from overdrive.structures import ImageList

__all__ = ["DeformableDETR"]


class DeformableDETR(nn.Module):
    """
    support Deformable DETR (https://arxiv.org/abs/2010.04159).
    """

    def __init__(
        self,
        *,
        backbone: Backbone,
        neck: Neck,
        position_embedding: nn.Module,
        transformer: nn.Module,
        box_head: DenseHead,
        num_queries: int,
        num_feature_levels: int = 4,
        with_box_refine: bool = False,
        as_two_stage: bool = False,
        pixel_mean: list[float],
        pixel_std: list[float],
    ):
        super().__init__()

        if not isinstance(backbone, Backbone):
            raise TypeError(f"backbone must be a Backbone, got {type(backbone).__name__}")
        if not isinstance(box_head, DenseHead):
            raise TypeError(f"box_head must be a DenseHead, got {type(box_head).__name__}")
        if not isinstance(neck, Neck):
            raise TypeError(f"neck must be a Neck, got {type(neck).__name__}")

        if num_feature_levels != len(transformer.in_features):
            raise ValueError(f"num_feature_levels ({num_feature_levels}) must match the number of input features ({len(transformer.in_features)})")

        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage

        self.backbone = backbone
        self.neck = neck
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.box_head = box_head

        self.embed_dim = transformer.embed_dim

        # The query embedding is split into content query and positional query when the detector
        # is not two-stage, hence the embedding dimension is doubled.
        if not as_two_stage:
            self.query_embed = nn.Embedding(num_queries, self.embed_dim * 2)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, self.embed_dim))

        if as_two_stage:
            self.memory_trans_fc = nn.Linear(self.embed_dim, self.embed_dim)
            self.memory_trans_norm = nn.LayerNorm(self.embed_dim)
            self.pos_trans_fc = nn.Linear(self.embed_dim * 2, self.embed_dim * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dim * 2)
        else:
            self.reference_points_fc = nn.Linear(self.embed_dim, 2)

        self.init_weights()

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    def init_weights(self):
        """
        Initialize the parameters that are not covered by the transformer. The prediction branches
        of `box_head` are initialized last because the head has to be built before the transformer
        sharing its regression branches.
        """
        if self.as_two_stage:
            weight_init.xavier_uniform_fill(self.memory_trans_fc)
            weight_init.xavier_uniform_fill(self.pos_trans_fc)
        else:
            weight_init.xavier_uniform_fill(self.reference_points_fc)

        nn.init.normal_(self.level_embed)

        self.box_head.init_weights()

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    @staticmethod
    def get_valid_ratio(mask: torch.Tensor) -> torch.Tensor:
        """
        Get the valid ratios of a padded feature map.

        Args:
            mask (Tensor): Binary mask of a feature map where non-zero values represent ignored
                positions, with shape `(bs, H, W)`.

        Returns:
            Tensor: The ratios of the valid width and the valid height relative to the width and the
            height of the feature map, with shape `(bs, 2)`.
        """
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        return torch.stack([valid_ratio_w, valid_ratio_h], -1)

    @staticmethod
    def get_proposal_pos_embed(
        proposals: torch.Tensor,
        num_pos_feats: int = 128,
        temperature: int = 10000,
    ) -> torch.Tensor:
        """
        Get the position embedding of the proposals.

        Args:
            proposals (Tensor): Not normalized proposals with shape `(bs, num_queries, 4)` arranged
                as `(cx, cy, w, h)`.
            num_pos_feats (int): The feature dimension for each position along the cx, cy, w, and
                h axis. The final returned dimension is 4 times of this value.
            temperature (int): The temperature used for scaling the position embedding.

        Returns:
            Tensor: The position embedding of the proposals with shape
            `(bs, num_queries, num_pos_feats * 4)`.
        """
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ):
        """
        Generate the normalized box proposals of the encoder output. The proposals of the ignored
        positions are masked out so that they are never selected by the top-k query selection.

        Args:
            memory (Tensor): The output embeddings of the transformer encoder with shape
                `(bs, num_feat_points, embed_dim)`.
            memory_mask (Tensor): ByteTensor of the padding mask with shape `(bs, num_feat_points)`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels with shape
                `(num_levels, 2)`.

        Returns:
            tuple[Tensor, Tensor]: The transformed memory with shape `(bs, num_feat_points, embed_dim)`
            and the proposals with shape `(bs, num_feat_points, 4)` arranged as `(cx, cy, w, h)`.
        """
        bs = memory.size(0)
        proposals = []
        cur = 0  # start index in the sequence of the current level
        for lvl, (H, W) in enumerate(spatial_shapes):
            H, W = int(H), int(W)
            if memory_mask is not None:
                mask_flatten = memory_mask[:, cur : cur + H * W].view(bs, H, W, 1)
                valid_H = torch.sum(~mask_flatten[:, :, 0, 0], 1).unsqueeze(-1)
                valid_W = torch.sum(~mask_flatten[:, 0, :, 0], 1).unsqueeze(-1)
                scale = torch.cat([valid_W, valid_H], 1).view(bs, 1, 1, 2)
            else:
                scale = memory.new_tensor([W, H]).view(1, 1, 1, 2)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device),
                indexing="ij",
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposals.append(torch.cat((grid, wh), -1).view(bs, -1, 4))
            cur += H * W

        output_proposals = torch.cat(proposals, 1)
        # do not use `all` to make it exportable to onnx
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).sum(-1, keepdim=True) == output_proposals.shape[-1]
        # inverse_sigmoid without the numerical clamp of `overdrive.layers.inverse_sigmoid`, the
        # range of `output_proposals` is guaranteed by the validity check above.
        output_proposals = torch.log(output_proposals / (1 - output_proposals))

        if memory_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        if memory_mask is not None:
            output_memory = output_memory.masked_fill(memory_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.memory_trans_norm(self.memory_trans_fc(output_memory))

        return output_memory, output_proposals

    def forward(self, batched_inputs: list[dict[str, torch.Tensor]]):
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        features = self.neck(features)

        N, _, H, W = images.tensor.size()
        image_masks = images.tensor.new_ones(N, H, W)
        for image_id, (h, w) in enumerate(images.image_sizes):
            image_masks[image_id, :h, :w] = 0

        # the position embedding, the padding mask and the valid ratios of every input feature level
        feature_masks = []
        valid_ratios = []
        position_embeddings = []
        spatial_shapes = []
        for in_feature in self.transformer.in_features:
            feat = features[in_feature]
            mask = F.interpolate(image_masks[None], size=feat.shape[-2:]).to(torch.bool)[0]
            feature_masks.append(mask.flatten(1))
            position_embeddings.append(self.position_embedding(mask))
            valid_ratios.append(self.get_valid_ratio(mask))
            spatial_shapes.append([feat.shape[-2], feat.shape[-1]])

        spatial_shapes = image_masks.new_tensor(spatial_shapes, dtype=torch.long)
        # the start index of each level in the flattened features
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(valid_ratios, 1)
        # (bs, sum(h_lvl * w_lvl)), where non-zero values represent ignored positions
        feature_masks = torch.cat(feature_masks, 1)

        feat_flatten, pos_embed_flatten = self.transformer.flatten_features(features, position_embeddings, self.level_embed)

        encoder_reference_points = self.transformer.encoder.get_reference_points(spatial_shapes, valid_ratios)

        # The encoder output is needed before the decoder in both cases: it is the memory of the
        # decoder cross attention, and it is also the source of the query proposals when the
        # detector is two-stage.
        memory = self.transformer.encoder(
            query=feat_flatten,
            query_pos=pos_embed_flatten,
            key_padding_mask=feature_masks,
            reference_points=encoder_reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        enc_cls_scores = None
        enc_bbox_preds = None
        if self.as_two_stage:
            if memory.shape[1] < self.num_queries:
                raise ValueError(
                    f"The two-stage detector selects {self.num_queries} queries from the encoder output, "
                    f"but the input images only provide {memory.shape[1]} feature points. "
                    "Use larger images or a smaller `num_queries`."
                )

            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, feature_masks, spatial_shapes)

            enc_cls_scores = self.box_head.cls_branches[self.box_head.num_decoder_layers](output_memory)
            enc_bbox_preds_unact = self.box_head.reg_branches[self.box_head.num_decoder_layers](output_memory) + output_proposals
            # only the first channel of the classification scores is used to select the top-k
            # queries, and the class targets of the proposals are set to the same foreground class
            topk_proposals = torch.topk(enc_cls_scores[..., 0], self.num_queries, dim=1)[1]
            topk_coords_unact = torch.gather(enc_bbox_preds_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).detach()
            reference_points = topk_coords_unact.sigmoid()
            enc_bbox_preds = enc_bbox_preds_unact.sigmoid()

            pos_trans_out = self.pos_trans_norm(self.pos_trans_fc(self.get_proposal_pos_embed(topk_coords_unact, self.embed_dim // 2)))
            query_pos_embed, query_embed = torch.split(pos_trans_out, self.embed_dim, dim=2)
        else:
            query_pos_embed, query_embed = torch.split(self.query_embed.weight, self.embed_dim, dim=1)
            query_pos_embed = query_pos_embed.unsqueeze(0).expand(N, -1, -1)
            query_embed = query_embed.unsqueeze(0).expand(N, -1, -1)
            reference_points = self.reference_points_fc(query_pos_embed).sigmoid()

        hidden_states, references = self.transformer(
            query_embed=query_embed,
            query_pos_embed=query_pos_embed,
            memory=memory,
            key_padding_mask=feature_masks,
            valid_ratios=valid_ratios,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reference_points=reference_points,
            reg_branches=self.box_head.reg_branches if self.with_box_refine else None,
        )

        cls_scores, bbox_preds = self.box_head(hidden_states, references)

        if self.training:
            if "instances" not in batched_inputs[0]:
                raise KeyError("Instance annotations are missing in training!")

            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            losses = self.box_head.loss(cls_scores, bbox_preds, gt_instances, enc_cls_scores, enc_bbox_preds)
            return losses
        else:
            results = self.box_head.inference(cls_scores, bbox_preds, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes, strict=True):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                transforms = input_per_image.get("transforms", None)
                r = detector_postprocess(results_per_image, height, width, transforms=transforms)
                processed_results.append({"instances": r})
            return processed_results

    def preprocess_image(self, batched_inputs: list[dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        padding_constraints = {}
        padding_constraints.update(self.backbone.padding_constraints)
        padding_constraints.update(self.neck.padding_constraints)
        images = ImageList.from_tensors(images, padding_constraints)
        return images

    def export_onnx(self, tensors):
        raise NotImplementedError
