import torch
import torch.nn as nn

from overdrive.modeling.backbone import Backbone
from overdrive.modeling.dense_head import DenseHead
from overdrive.modeling.neck import Neck
from overdrive.modeling.postprocessing import detector_postprocess
from overdrive.modeling.roi_head import ROIHead
from overdrive.structures import ImageList

__all__ = ["FasterRCNN"]


class FasterRCNN(nn.Module):
    """
    support Faster R-CNN (https://arxiv.org/abs/1506.01497).
    """

    def __init__(
        self,
        *,
        backbone: Backbone,
        neck: Neck | None = None,
        rpn_head: DenseHead,
        roi_head: ROIHead,
        pixel_mean: list[float],
        pixel_std: list[float],
    ):
        super().__init__()

        if not isinstance(backbone, Backbone):
            raise TypeError(f"Expected backbone to be a Backbone instance, but got {type(backbone).__name__}.")

        if not isinstance(rpn_head, DenseHead):
            raise TypeError(f"Expected rpn_head to be a DenseHead instance, but got {type(rpn_head).__name__}.")

        if not isinstance(roi_head, ROIHead):
            raise TypeError(f"Expected roi_head to be a ROIHead instance, but got {type(roi_head).__name__}.")

        if neck is not None and not isinstance(neck, Neck):
            raise TypeError(f"Expected neck to be a Neck instance, but got {type(neck).__name__}.")

        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head
        self.roi_head = roi_head

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def with_neck(self) -> bool:
        return hasattr(self, "neck") and self.neck is not None

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    def forward(self, batched_inputs: list[dict[str, torch.Tensor]]):
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        if self.with_neck:
            features = self.neck(features)
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        proposals, rpn_losses = self.rpn_head.loss_and_inference(images, pred_objectness_logits, pred_anchor_deltas, gt_instances)
        results, roi_losses = self.roi_head(images, features, proposals, gt_instances)

        if self.training:
            losses = {}
            losses.update(rpn_losses)
            losses.update(roi_losses)
            return losses
        else:
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
        if self.with_neck:
            padding_constraints.update(self.neck.padding_constraints)
        images = ImageList.from_tensors(images, padding_constraints)
        return images

    def export_onnx(self, tensors):
        raise NotImplementedError
