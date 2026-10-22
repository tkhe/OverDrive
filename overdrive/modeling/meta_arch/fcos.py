import torch
import torch.nn as nn

from overdrive.modeling.backbone import Backbone
from overdrive.modeling.dense_head import DenseHead
from overdrive.modeling.neck import Neck
from overdrive.modeling.postprocessing import detector_postprocess
from overdrive.structures import ImageList

__all__ = ["FCOS"]


class FCOS(nn.Module):
    """
    support FCOS (https://arxiv.org/abs/1904.01355).
    """

    def __init__(
        self,
        *,
        backbone: Backbone,
        neck: Neck,
        box_head: DenseHead,
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

        self.backbone = backbone
        self.neck = neck
        self.box_head = box_head

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    def forward(self, batched_inputs: list[dict[str, torch.Tensor]]):
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        features = self.neck(features)
        cls_scores, bbox_preds, objectnesses = self.box_head(features)

        if self.training:
            if "instances" not in batched_inputs[0]:
                raise ValueError("Instance annotations are missing in training!")

            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            losses = self.box_head.loss(cls_scores, bbox_preds, objectnesses, gt_instances)
            return losses

        results = self.box_head.inference(cls_scores, bbox_preds, objectnesses, images.image_sizes)

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            results,
            batched_inputs,
            images.image_sizes,
            strict=True,
        ):
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
        tensors = self.backbone(tensors)
        tensors = self.neck(tensors)
        tensors = self.box_head(tensors)
        return tensors
