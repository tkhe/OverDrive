import torch
import torch.nn as nn
import torch.nn.functional as F
from overdrive.layers import inverse_sigmoid
from overdrive.modeling.backbone import Backbone
from overdrive.modeling.neck import Neck
from overdrive.modeling.postprocessing import detector_postprocess
from overdrive.structures import BoxMode
from overdrive.structures import ImageList

from dn_detr.modeling.denoising import DNQueryGenerator
from dn_detr.modeling.dense_head import DNDETRHead

__all__ = ["DNDETR"]


class DNDETR(nn.Module):
    """
    DN-DETR with a DAB-DETR matching branch and noisy ground-truth queries.
    """

    def __init__(
        self,
        *,
        backbone: Backbone,
        neck: Neck,
        position_embedding: nn.Module,
        transformer: nn.Module,
        box_head: DNDETRHead,
        denoising_generator: DNQueryGenerator,
        num_queries: int,
        freeze_anchor_box_center: bool = False,
        pixel_mean: list[float],
        pixel_std: list[float],
    ):
        super().__init__()

        if not isinstance(backbone, Backbone) or not isinstance(neck, Neck) or not isinstance(box_head, DNDETRHead):
            raise TypeError("backbone, neck and box_head must use OverDrive model components")
        if not isinstance(denoising_generator, DNQueryGenerator):
            raise TypeError("denoising_generator must be a DNQueryGenerator")
        if denoising_generator.num_queries != num_queries or denoising_generator.num_classes != box_head.num_classes or denoising_generator.embed_dim != transformer.embed_dim:
            raise ValueError("denoising_generator dimensions must match the model")

        self.num_queries = num_queries
        self.backbone = backbone
        self.neck = neck
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.box_head = box_head
        self.anchor_box_embed = nn.Embedding(num_queries, 4)
        self.denoising_generator = denoising_generator

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

        if freeze_anchor_box_center:
            # Preserve detrex's initialization and slice-level requires_grad behavior.
            self.anchor_box_embed.weight.data[:, :2].uniform_(0, 1)
            self.anchor_box_embed.weight.data[:, :2] = inverse_sigmoid(self.anchor_box_embed.weight.data[:, :2])
            self.anchor_box_embed.weight.data[:, :2].requires_grad = False

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    def forward(self, batched_inputs: list[dict[str, torch.Tensor]]):
        images = self.preprocess_image(batched_inputs)
        gt_instances = None
        if self.training:
            if any("instances" not in inputs for inputs in batched_inputs):
                raise KeyError("Instance annotations are missing in training")
            gt_instances = [inputs["instances"].to(self.device) for inputs in batched_inputs]

        features = self.neck(self.backbone(images.tensor))
        if len(features) != 1:
            raise ValueError(f"DN-DETR uses one feature level, got {len(features)}")

        batch_size, _, height, width = images.tensor.shape
        image_masks = images.tensor.new_ones((batch_size, height, width))
        for image_idx, (image_height, image_width) in enumerate(images.image_sizes):
            image_masks[image_idx, :image_height, :image_width] = 0
        feature_size = features[self.transformer.in_features[0]].shape[-2:]
        image_masks = F.interpolate(image_masks[:, None], size=feature_size, mode="nearest")[:, 0].bool()
        pos_embed = self.position_embedding(image_masks)

        matching_labels = self.denoising_generator.matching_label_queries(batch_size)
        matching_boxes = self.anchor_box_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        if gt_instances is None:
            input_labels = matching_labels
            input_boxes = matching_boxes
            self_attn_mask = None
            dn_metadata = None
        else:
            gt_labels = [gt.gt_classes for gt in gt_instances]
            gt_boxes = []
            for gt in gt_instances:
                image_height, image_width = gt.image_size
                scale = gt.gt_boxes.tensor.new_tensor((image_width, image_height, image_width, image_height))
                normalized_xyxy = gt.gt_boxes.tensor / scale
                gt_boxes.append(BoxMode.convert(normalized_xyxy, BoxMode.XYXY_ABS, BoxMode.CXCYWH_ABS))

            noisy_labels, noisy_boxes, self_attn_mask, dn_metadata = self.denoising_generator(gt_labels, gt_boxes)
            input_labels = torch.cat((noisy_labels, matching_labels), dim=1)
            input_boxes = torch.cat((noisy_boxes, matching_boxes), dim=1)

        hidden_states, references = self.transformer(
            features,
            image_masks,
            input_boxes,
            pos_embed,
            target=input_labels,
            reg_branches=self.box_head.fc_reg,
            self_attn_mask=self_attn_mask,
        )
        cls_scores, bbox_preds = self.box_head(hidden_states, references)

        if gt_instances is not None:
            return self.box_head.loss(cls_scores, bbox_preds, gt_instances, dn_metadata)

        results = self.box_head.inference(cls_scores, bbox_preds, images.image_sizes)
        processed_results = []
        for result, inputs, image_size in zip(results, batched_inputs, images.image_sizes, strict=True):
            output_height = inputs.get("height", image_size[0])
            output_width = inputs.get("width", image_size[1])
            transforms = inputs.get("transforms")
            processed = detector_postprocess(result, output_height, output_width, transforms=transforms)
            processed_results.append({"instances": processed})
        return processed_results

    def preprocess_image(self, batched_inputs: list[dict[str, torch.Tensor]]):
        images = [inputs["image"].to(self.device) for inputs in batched_inputs]
        images = [(image - self.pixel_mean) / self.pixel_std for image in images]
        padding_constraints = {}
        padding_constraints.update(self.backbone.padding_constraints)
        padding_constraints.update(self.neck.padding_constraints)
        return ImageList.from_tensors(images, padding_constraints)

    def export_onnx(self, tensors):
        raise NotImplementedError
