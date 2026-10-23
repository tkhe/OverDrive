import torch
import torch.nn as nn
import torch.nn.functional as F
from overdrive.layers import weight_init
from overdrive.modeling.backbone import Backbone
from overdrive.modeling.neck import Neck
from overdrive.modeling.postprocessing import detector_postprocess
from overdrive.structures import ImageList

from dino.modeling.denoising import CDNQueryGenerator
from dino.modeling.dense_head import DINOHead

__all__ = ["DINO"]


class DINO(nn.Module):
    """
    DINO with mixed query selection, contrastive denoising and box refinement.
    """

    def __init__(
        self,
        *,
        backbone: Backbone,
        neck: Neck,
        position_embedding: nn.Module,
        transformer: nn.Module,
        box_head: DINOHead,
        denoising_generator: CDNQueryGenerator,
        num_queries: int = 900,
        num_feature_levels: int = 4,
        pixel_mean: list[float],
        pixel_std: list[float],
    ):
        super().__init__()

        if not isinstance(backbone, Backbone) or not isinstance(neck, Neck) or not isinstance(box_head, DINOHead):
            raise TypeError("DINO expects a Backbone, Neck and DINOHead")
        if num_queries <= 0 or num_feature_levels != len(transformer.in_features):
            raise ValueError("invalid query count or number of feature levels")
        if box_head.num_decoder_layers != transformer.decoder.num_layers:
            raise ValueError("head and decoder layer counts must agree")
        if (denoising_generator.num_queries, denoising_generator.num_classes, denoising_generator.embed_dim) != (num_queries, box_head.num_classes, transformer.embed_dim):
            raise ValueError("denoising query count, classes and embedding dimension must match the detector")
        self.backbone = backbone
        self.neck = neck
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.box_head = box_head
        self.denoising_generator = denoising_generator
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.embed_dim = transformer.embed_dim
        self.query_embed = nn.Embedding(num_queries, self.embed_dim)
        self.level_embed = nn.Parameter(torch.empty(num_feature_levels, self.embed_dim))
        self.memory_trans_fc = nn.Linear(self.embed_dim, self.embed_dim)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dim)
        self.init_weights()
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    def init_weights(self):
        weight_init.xavier_uniform_fill(self.memory_trans_fc)
        nn.init.xavier_uniform_(self.query_embed.weight)
        nn.init.normal_(self.level_embed)
        self.box_head.init_weights()

    def pre_decoder(self, memory, memory_mask, spatial_shapes, gt_instances=None):
        output_memory, proposals = self.gen_encoder_output_proposals(memory, memory_mask, spatial_shapes)
        valid = torch.isfinite(proposals).all(-1)
        if (valid.sum(1) < self.num_queries).any():
            raise ValueError(f"DINO needs at least {self.num_queries} valid encoder proposals per image; use larger images or fewer queries")
        enc_scores = self.box_head.cls_branches[-1](output_memory)
        enc_boxes_unact = self.box_head.reg_branches[-1](output_memory) + proposals
        topk = enc_scores.max(-1).values.masked_fill(~valid, -torch.inf).topk(self.num_queries, dim=1).indices
        scores = enc_scores.gather(1, topk.unsqueeze(-1).expand(-1, -1, self.box_head.num_classes))
        boxes_unact = enc_boxes_unact.gather(1, topk.unsqueeze(-1).expand(-1, -1, 4))
        # Encoder supervision retains gradients; decoder anchors stop them at selection.
        enc_boxes = boxes_unact.sigmoid()
        references = boxes_unact.detach()
        query = self.query_embed.weight.unsqueeze(0).expand(memory.shape[0], -1, -1)
        dn_mask, metadata = None, None
        if self.training:
            dn_labels, dn_boxes, dn_mask, metadata = self.denoising_generator(gt_instances)
            if dn_labels is not None:
                query = torch.cat((dn_labels, query), dim=1)
                references = torch.cat((dn_boxes, references), dim=1)
            # Keep the embedding in the graph when DN is disabled or this rank has no GT.
            query = query + self.denoising_generator.label_embedding.weight.sum() * 0.0
        return query, references.sigmoid(), dn_mask, metadata, scores, enc_boxes

    def forward(self, batched_inputs: list[dict[str, torch.Tensor]]):
        gt_instances = None
        if self.training:
            if any("instances" not in x for x in batched_inputs):
                raise KeyError("Instance annotations are missing in training!")
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
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

        # Encoder memory supplies both cross-attention values and initial box proposals.
        memory = self.transformer.encoder(
            query=feat_flatten,
            query_pos=pos_embed_flatten,
            key_padding_mask=feature_masks,
            reference_points=encoder_reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        query, references, dn_mask, metadata, enc_scores, enc_boxes = self.pre_decoder(memory, feature_masks, spatial_shapes, gt_instances)
        hidden_states, references = self.transformer(
            query_embed=query,
            memory=memory,
            key_padding_mask=feature_masks,
            valid_ratios=valid_ratios,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reference_points=references,
            reg_branches=self.box_head.reg_branches,
            self_attn_mask=dn_mask,
        )
        cls_scores, bbox_preds = self.box_head(hidden_states, references)
        if self.training:
            return self.box_head.loss(cls_scores, bbox_preds, gt_instances, enc_scores, enc_boxes, metadata)
        results = self.box_head.inference(cls_scores, bbox_preds, images.image_sizes)
        processed_results = []
        for result, inputs, image_size in zip(results, batched_inputs, images.image_sizes, strict=True):
            height = inputs.get("height", image_size[0])
            width = inputs.get("width", image_size[1])
            result = detector_postprocess(result, height, width, transforms=inputs.get("transforms"))
            processed_results.append({"instances": result})
        return processed_results

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs):
        images = [(inputs["image"].to(self.device) - self.pixel_mean) / self.pixel_std for inputs in batched_inputs]
        padding = {**self.backbone.padding_constraints, **self.neck.padding_constraints}
        return ImageList.from_tensors(images, padding)

    @staticmethod
    def get_valid_ratio(mask):
        height, width = mask.shape[-2:]
        valid_width = (~mask[:, 0, :]).sum(1).float() / width
        valid_height = (~mask[:, :, 0]).sum(1).float() / height
        return torch.stack((valid_width, valid_height), -1)

    def gen_encoder_output_proposals(self, memory, memory_mask, spatial_shapes):
        """
        Seed normalized boxes at valid feature centers with level-dependent sizes.
        """
        batch_size = memory.shape[0]
        proposals, offset = [], 0
        for level, (height, width) in enumerate(spatial_shapes.tolist()):
            if memory_mask is None:
                scale = memory.new_tensor([width, height]).view(1, 1, 1, 2)
            else:
                mask = memory_mask[:, offset : offset + height * width].reshape(batch_size, height, width)
                scale = torch.stack(((~mask[:, 0, :]).sum(1), (~mask[:, :, 0]).sum(1)), -1)[:, None, None]
            y, x = torch.meshgrid(
                torch.arange(height, device=memory.device, dtype=torch.float32),
                torch.arange(width, device=memory.device, dtype=torch.float32),
                indexing="ij",
            )
            centers = (torch.stack((x, y), -1)[None].expand(batch_size, -1, -1, -1) + 0.5) / scale.clamp(min=1)
            sizes = torch.full_like(centers, 0.05 * 2**level)
            proposals.append(torch.cat((centers, sizes), -1).reshape(batch_size, -1, 4))
            offset += height * width
        proposals = torch.cat(proposals, 1)
        valid = ((proposals > 0.01) & (proposals < 0.99)).all(-1)
        if memory_mask is not None:
            valid = valid & ~memory_mask
        # Replace invalid boxes before logit to avoid evaluating log on out-of-range coordinates.
        logits = torch.logit(proposals.masked_fill(~valid[..., None], 0.5)).masked_fill(~valid[..., None], torch.inf)
        output_memory = self.memory_trans_norm(self.memory_trans_fc(memory.masked_fill(~valid[..., None], 0)))
        return output_memory, logits

    def export_onnx(self, tensors):
        raise NotImplementedError("DINO ONNX export is not implemented")
