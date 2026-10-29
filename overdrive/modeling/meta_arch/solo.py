import torch
import torch.nn as nn

from overdrive.modeling.backbone import Backbone
from overdrive.modeling.dense_head import DenseHead
from overdrive.modeling.neck import Neck
from overdrive.modeling.postprocessing import instance_seg_postprocess
from overdrive.structures import ImageList

__all__ = ["SOLO"]


class SOLO(nn.Module):
    """Single-stage instance segmentation with a SOLO mask head."""

    def __init__(self, *, backbone: Backbone, neck: Neck, mask_head: DenseHead,
                 pixel_mean: list[float], pixel_std: list[float]):
        super().__init__()
        if not isinstance(backbone, Backbone) or not isinstance(neck, Neck) or not isinstance(mask_head, DenseHead):
            raise TypeError("SOLO requires a Backbone, Neck and DenseHead")
        self.backbone = backbone
        self.neck = neck
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
        predictions = self.mask_head(features)
        padded_size = images.tensor.shape[-2:]
        if self.training:
            if any("instances" not in x for x in batched_inputs):
                raise ValueError("Instance annotations are missing in training!")
            gt = [x["instances"].to(self.device) for x in batched_inputs]
            return self.mask_head.loss(*predictions, gt, padded_size)

        results = self.mask_head.inference(*predictions, images.image_sizes, padded_size)
        processed = []
        for result, inputs, image_size in zip(results, batched_inputs, images.image_sizes, strict=True):
            result = instance_seg_postprocess(
                result, inputs.get("height", image_size[0]), inputs.get("width", image_size[1]),
                mask_threshold=self.mask_head.test_mask_threshold, transforms=inputs.get("transforms"),
            )
            processed.append({"instances": result})
        return processed

    def export_onnx(self, tensors):
        """Export raw head outputs; dynamic mask decoding/NMS remains external."""
        return self.mask_head(self.neck(self.backbone(tensors)))
