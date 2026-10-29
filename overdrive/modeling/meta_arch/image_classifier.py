from collections.abc import Mapping

import torch
import torch.nn as nn

from overdrive.modeling.backbone import Backbone
from overdrive.structures import ImageList

__all__ = ["ImageClassifier"]


class ImageClassifier(nn.Module):
    """Single-label image classification using the backbone's ``linear`` output.

    Configure the backbone with ``num_classes`` to enable its classification head.
    Inputs are dictionaries containing a CHW ``image`` tensor and, during training,
    a ``label`` (Python integer or integer scalar tensor). Inference returns one
    dictionary per image with a ``pred_scores`` vector of class probabilities.
    """

    def __init__(
        self,
        *,
        backbone: Backbone,
        loss_cls: nn.Module | Mapping[str, nn.Module],
        pixel_mean: list[float],
        pixel_std: list[float],
    ):
        super().__init__()

        if not isinstance(backbone, Backbone):
            raise TypeError(f"Expected backbone to be a Backbone instance, but got {type(backbone).__name__}.")

        self.backbone = backbone

        if isinstance(loss_cls, nn.Module):
            self.loss_cls = loss_cls
        elif isinstance(loss_cls, Mapping):
            self.loss_cls = nn.ModuleDict(loss_cls)
        else:
            raise TypeError(f"loss_cls must be a nn.Module or dict[str, nn.Module], but got {type(loss_cls)}")

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    def forward(self, batched_inputs: list[dict[str, torch.Tensor | int]]):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        logits = features["linear"]

        if self.training:
            targets = torch.stack([torch.as_tensor(x["label"], dtype=torch.long, device=self.device) for x in batched_inputs])

            if isinstance(self.loss_cls, nn.ModuleDict):
                return {name: loss(logits, targets) for name, loss in self.loss_cls.items()}
            return {"loss_cls": self.loss_cls(logits, targets)}

        return [{"pred_scores": scores} for scores in logits.softmax(dim=-1)]

    def preprocess_image(self, batched_inputs: list[dict[str, torch.Tensor | int]]):
        """Normalize, pad and batch images using the backbone's padding constraints."""
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        return ImageList.from_tensors(images, self.backbone.padding_constraints)

    def export_onnx(self, tensors):
        """Return raw logits for an already normalized and batched image tensor."""
        return self.backbone(tensors)["linear"]
