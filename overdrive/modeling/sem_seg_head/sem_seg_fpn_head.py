import math
from collections.abc import Mapping

import torch.nn as nn
import torch.nn.functional as F

from overdrive.layers import Conv2d
from overdrive.layers import ShapeSpec
from overdrive.layers import weight_init

from .base import SemSegHead

__all__ = ["SemSegFPNHead"]


class SemSegFPNHead(SemSegHead):
    """
    Panoptic FPN semantic head with additive multiscale fusion.
    """

    def __init__(
        self,
        input_shape: dict[str, ShapeSpec],
        in_features: list[str],
        *,
        num_classes: int,
        channels: int = 128,
        common_stride: int = 4,
        norm: str | dict = "GN",
        activation: str | dict = "ReLU",
        ignore_index: int = 255,
        loss_sem_seg: nn.Module | Mapping[str, nn.Module],
    ):
        super().__init__()

        if not in_features or len(set(in_features)) != len(in_features):
            raise ValueError("in_features must be nonempty and unique")

        if common_stride <= 0:
            raise ValueError("common_stride must be positive")

        self.in_features = sorted(in_features, key=lambda name: input_shape[name].stride)
        strides = [input_shape[name].stride for name in self.in_features]
        self.common_stride = common_stride
        self.ignore_index = ignore_index
        self.align_corners = False
        for stride in strides:
            ratio = stride / common_stride
            if ratio < 1 or not math.log2(ratio).is_integer():
                raise ValueError("Feature strides must be power-of-two multiples of common_stride")

        self.scale_heads = nn.ModuleDict()
        for name, stride in zip(self.in_features, strides, strict=True):
            shape = input_shape[name]
            ratio = stride / common_stride
            layers = []
            for i in range(max(1, int(math.log2(ratio)))):
                conv = Conv2d(
                    shape.channels if i == 0 else channels,
                    channels,
                    kernel_size=3,
                    norm=norm,
                    activation=activation,
                )
                layers.append(conv)
                if ratio != 1:
                    layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.scale_heads[name] = nn.Sequential(*layers)
        self.predictor = nn.Conv2d(channels, num_classes, kernel_size=1)

        if isinstance(loss_sem_seg, nn.Module):
            self.loss_sem_seg = loss_sem_seg
        elif isinstance(loss_sem_seg, Mapping):
            self.loss_sem_seg = nn.ModuleDict(loss_sem_seg)
        else:
            raise TypeError("loss_sem_seg must be a module or mapping of modules")

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                weight_init.c2_msra_fill(module)

    def forward(self, features):
        x = self.scale_heads[self.in_features[0]](features[self.in_features[0]])
        for name in self.in_features[1:]:
            x = x + self.scale_heads[name](features[name])
        return self.predictor(x)

    def loss(self, predictions, targets):
        predictions = F.interpolate(
            predictions.float(),
            size=targets.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # Empty semantic supervision must still produce a differentiable loss.
        if not (targets != self.ignore_index).any():
            if isinstance(self.loss_sem_seg, nn.ModuleDict):
                return {name: predictions.sum() * 0.0 for name in self.loss_sem_seg}
            return {"loss_sem_seg": predictions.sum() * 0.0}

        losses = {}
        if isinstance(self.loss_sem_seg, nn.ModuleDict):
            for loss in self.loss_sem_seg:
                losses[loss] = self.loss_sem_seg[loss](predictions, targets)
        else:
            losses["loss_sem_seg"] = self.loss_sem_seg(predictions, targets)

        return losses

    def inference(self, predictions, image_size):
        return F.interpolate(predictions, size=image_size, mode="bilinear", align_corners=False)
