import torch
import torch.nn as nn
import torch.nn.functional as F

from overdrive.layers import inverse_sigmoid
from overdrive.modeling.backbone import Backbone
from overdrive.modeling.dense_head import DenseHead
from overdrive.modeling.neck import Neck
from overdrive.modeling.postprocessing import detector_postprocess
from overdrive.structures import ImageList

__all__ = ["DABDETR"]


class DABDETR(nn.Module):
    """
    support DAB-DETR (https://arxiv.org/abs/2201.12329).
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
        freeze_anchor_box_center: bool = True,
        pixel_mean: list[float],
        pixel_std: list[float],
    ):
        super().__init__()

        if not isinstance(backbone, Backbone):
            raise TypeError(f"Expected backbone to be a Backbone instance, but got {type(backbone).__name__}.")

        if not isinstance(neck, Neck):
            raise TypeError(f"Expected neck to be a Neck instance, but got {type(neck).__name__}.")

        if not isinstance(box_head, DenseHead):
            raise TypeError(f"Expected box_head to be a DenseHead instance, but got {type(box_head).__name__}.")

        self.num_queries = num_queries

        self.backbone = backbone
        self.neck = neck
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.box_head = box_head

        self.query_embed = nn.Embedding(num_queries, 4)

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

        if freeze_anchor_box_center:
            self.query_embed.weight.data[:, :2].uniform_(0, 1)
            self.query_embed.weight.data[:, :2] = inverse_sigmoid(self.query_embed.weight.data[:, :2])
            self.query_embed.weight.data[:, :2].requires_grad = False

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    def forward(self, batched_inputs: list[dict[str, torch.Tensor]]):
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        features = self.neck(features)

        if len(features) != 1:
            raise ValueError(f"DAB-DETR only use the last level feature map, got {len(features)}")

        N, _, H, W = images.tensor.size()
        image_masks = images.tensor.new_ones(N, H, W)
        for image_id, (h, w) in enumerate(images.image_sizes):
            image_masks[image_id, :h, :w] = 0

        image_masks = F.interpolate(image_masks.unsqueeze(1), size=features[self.transformer.in_features[-1]].shape[-2:], mode="nearest").to(torch.bool).squeeze(1)

        position_embedding = self.position_embedding(image_masks)

        hidden_states, references = self.transformer(features, image_masks, self.query_embed.weight, position_embedding, self.box_head.fc_reg)

        cls_scores, bbox_preds = self.box_head(hidden_states, references)

        if self.training:
            if "instances" not in batched_inputs[0]:
                raise KeyError("Instance annotations are missing in training!")

            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            losses = self.box_head.loss(cls_scores, bbox_preds, gt_instances)
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
