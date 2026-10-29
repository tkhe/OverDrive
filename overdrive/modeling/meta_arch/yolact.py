import torch
import torch.nn as nn

from overdrive.modeling.backbone import Backbone
from overdrive.modeling.dense_head import YOLACTHead, YOLACTProtonet
from overdrive.modeling.neck import Neck
from overdrive.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from overdrive.structures import ImageList

__all__ = ["YOLACT"]


class YOLACT(nn.Module):
    """YOLACT: parallel single-stage detection and prototype mask prediction.

    Accepts OverDrive batched image dictionaries and returns a loss dictionary
    during training or Instances with boxes/scores/classes/bitmasks at inference.
    Supports arbitrary padded image sizes through the supplied backbone/neck.
    """

    def __init__(
        self,
        *,
        backbone: Backbone,
        neck: Neck,
        box_head: YOLACTHead,
        mask_head: YOLACTProtonet,
        pixel_mean: list[float],
        pixel_std: list[float],
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.box_head = box_head
        self.mask_head = mask_head
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs):
        images = [(x["image"].to(self.device) - self.pixel_mean) / self.pixel_std for x in batched_inputs]
        constraints = {**self.backbone.padding_constraints, **self.neck.padding_constraints}
        return ImageList.from_tensors(images, constraints)

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.neck(self.backbone(images.tensor))
        predictions = self.box_head(features)
        prototypes, semantic_logits = self.mask_head(features)
        padded_size = images.tensor.shape[-2:]
        if self.training:
            gt = [x["instances"].to(self.device) for x in batched_inputs]
            losses, positives = self.box_head.loss(*predictions, gt, return_positive_info=True)
            losses.update(self.mask_head.loss(prototypes, semantic_logits, positives, gt, padded_size))
            return losses
        detections = self.box_head.inference(*predictions, images.image_sizes)
        detections = self.mask_head.inference(prototypes, detections, padded_size)
        processed = []
        for result, inputs in zip(detections, batched_inputs):
            image_size = result.image_size
            height, width = inputs.get("height", image_size[0]), inputs.get("width", image_size[1])
            transforms = inputs.get("transforms")
            # detector_postprocess filters/clips boxes and corresponding mask_probs
            # together. Full-image masks must not enter its ROI-mask paste path.
            result = detector_postprocess(result, height, width, transforms=transforms)
            probs = result.mask_probs
            if len(result):
                probs = sem_seg_postprocess(probs, image_size, height, width, transforms=transforms)
            else:
                probs = probs.new_empty((0, height, width))
            result.pred_masks = probs > self.mask_head.test_mask_threshold
            result.remove("mask_probs")
            processed.append({"instances": result})
        return processed

    def export_onnx(self, tensors):
        """Raw classes, box deltas, coefficients and prototypes; NMS is external."""
        features = self.neck(self.backbone(tensors))
        prototypes, _ = self.mask_head(features)
        return (*self.box_head(features), prototypes)
