from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from overdrive.layers import Conv2d
from overdrive.layers import ShapeSpec

from .base import SemSegHead

__all__ = ["DeepLabV3Head"]


class ASPP(nn.ModuleList):
    def __init__(
        self,
        dilations: list[int],
        in_channels: int,
        channels: int,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
    ):
        super().__init__()

        for dilation in dilations:
            layer = Conv2d(
                in_channels,
                channels,
                kernel_size=1 if dilation == 1 else 3,
                dilation=dilation,
                norm=norm,
                activation=activation,
            )
            self.append(layer)

    def forward(self, x):
        outs = []
        for layer in self:
            outs.append(layer(x))
        return outs


class DeepLabV3Head(SemSegHead):
    """
    support DeepLabV3Head (https://arxiv.org/abs/1706.05587).
    """

    def __init__(
        self,
        input_shape: dict[str, ShapeSpec],
        in_features: list[str],
        *,
        num_classes: int,
        channels: int,
        dilations: list[int],
        dropout: float = 0.1,
        ignore_index: int = -1,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
        loss_sem_seg: nn.Module | Mapping[str, nn.Module],
        align_corners: bool = False,
    ):
        super().__init__()

        if len(in_features) != 1:
            raise ValueError("DeepLabV3Head only allows one input feature map")

        in_channels = input_shape[in_features[0]].channels

        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.in_features = in_features

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2d(in_channels, channels, kernel_size=1, norm=norm, activation=activation),
        )
        self.aspp = ASPP(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            norm=norm,
            activation=activation,
        )
        self.bottleneck = Conv2d(
            (len(dilations) + 1) * channels,
            channels,
            kernel_size=3,
            norm=norm,
            activation=activation,
        )
        self.dropout = nn.Dropout2d(dropout)
        self.predictor = nn.Conv2d(channels, num_classes, kernel_size=1)

        if isinstance(loss_sem_seg, nn.Module):
            self.loss_sem_seg = loss_sem_seg
        elif isinstance(loss_sem_seg, Mapping):
            self.loss_sem_seg = nn.ModuleDict(loss_sem_seg)
        else:
            raise TypeError(f"loss_sem_seg must be a nn.Module or dict[str, nn.Module], but got {type(loss_sem_seg)}")

    def forward(self, bottom_up_features: dict[str, torch.Tensor]):
        x = bottom_up_features[self.in_features[0]]

        outs = [F.interpolate(self.image_pool(x), size=x.size()[2:], mode="bilinear", align_corners=self.align_corners)]
        outs.extend(self.aspp(x))
        outs = torch.cat(outs, dim=1)
        outs = self.bottleneck(outs)
        outs = self.dropout(outs)
        return self.predictor(outs)

    def loss(self, predictions, targets):
        if predictions.size()[-2:] != targets.size()[-2:]:
            predictions = F.interpolate(
                predictions,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

        losses = {}
        if isinstance(self.loss_sem_seg, nn.ModuleDict):
            for loss in self.loss_sem_seg:
                losses[loss] = self.loss_sem_seg[loss](predictions, targets)
        else:
            losses["loss_sem_seg"] = self.loss_sem_seg(predictions, targets)

        return losses

    def inference(self, predictions, image_size):
        return F.interpolate(predictions, size=image_size, mode="bilinear", align_corners=self.align_corners)
