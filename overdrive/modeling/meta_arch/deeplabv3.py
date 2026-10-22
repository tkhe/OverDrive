import torch
import torch.nn as nn

from overdrive.modeling.backbone import Backbone
from overdrive.modeling.postprocessing import sem_seg_postprocess
from overdrive.modeling.sem_seg_head import SemSegHead
from overdrive.structures import ImageList

__all__ = ["DeepLabV3"]


class DeepLabV3(nn.Module):
    """
    support DeepLabV3 (https://arxiv.org/abs/1706.05587).
    """

    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: SemSegHead,
        pixel_mean: list[float],
        pixel_std: list[float],
    ):
        super().__init__()

        if not isinstance(backbone, Backbone):
            raise TypeError(f"Expected backbone to be a Backbone instance, but got {type(backbone).__name__}.")

        if not isinstance(sem_seg_head, SemSegHead):
            raise TypeError(f"Expected sem_seg_head to be a SemSegHead instance, but got {type(sem_seg_head).__name__}.")

        self.backbone = backbone
        self.sem_seg_head = sem_seg_head

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    def forward(self, batched_inputs: list[dict[str, torch.Tensor]]):
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        predictions = self.sem_seg_head(features)

        if self.training:
            if "sem_seg" not in batched_inputs[0]:
                raise ValueError("semantic segmentation annotations are missing in training!")

            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            padding_constraints = {}
            padding_constraints.update(self.backbone.padding_constraints)
            targets = ImageList.from_tensors(
                targets,
                padding_constraints,
                pad_value=self.sem_seg_head.ignore_index,
            ).tensor

            losses = self.sem_seg_head.loss(predictions, targets)
            return losses
        else:
            results = self.sem_seg_head.inference(predictions, images.tensor.shape[-2:])

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes, strict=True):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                transforms = input_per_image.get("transforms", None)
                r = sem_seg_postprocess(
                    results_per_image,
                    image_size,
                    height,
                    width,
                    transforms=transforms,
                    align_corners=self.sem_seg_head.align_corners,
                )
                processed_results.append({"sem_seg": r})
            return processed_results

    def preprocess_image(self, batched_inputs: list[dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        padding_constraints = {}
        padding_constraints.update(self.backbone.padding_constraints)
        images = ImageList.from_tensors(images, padding_constraints)
        return images

    def export_onnx(self, tensors):
        tensors = self.backbone(tensors)
        tensors = self.sem_seg_head(tensors)
        return tensors
