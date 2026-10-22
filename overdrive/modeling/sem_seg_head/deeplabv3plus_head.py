from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from overdrive.layers import Conv2d
from overdrive.layers import DepthwiseSeparableConv2d
from overdrive.layers import ShapeSpec

from .base import SemSegHead

__all__ = ["DeepLabV3PlusHead"]


class DepthwiseSeparableASPP(nn.ModuleList):
    """
    ASPP with depthwise separable convolutions in the atrous branches.
    """

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
            conv = DepthwiseSeparableConv2d if dilation > 1 else Conv2d
            self.append(
                conv(
                    in_channels,
                    channels,
                    kernel_size=1 if dilation == 1 else 3,
                    dilation=dilation,
                    norm=norm,
                    activation=activation,
                )
            )

    def forward(self, x):
        return [layer(x) for layer in self]


class DeepLabV3PlusHead(SemSegHead):
    """
    support DeepLabV3PlusHead (https://arxiv.org/abs/1802.02611).
    """

    def __init__(
        self,
        input_shape: dict[str, ShapeSpec],
        in_features: list[str],
        *,
        num_classes: int,
        channels: int,
        dilations: list[int],
        low_level_channels: int = 48,
        dropout: float = 0.1,
        ignore_index: int = -1,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
        loss_sem_seg: nn.Module | Mapping[str, nn.Module] = None,
        align_corners: bool = False,
    ):
        super().__init__()

        if len(in_features) != 2:
            raise ValueError("DeepLabV3PlusHead requires exactly two input feature maps")

        # in_features[0] is the low-level feature, in_features[1] is the high-level feature
        low_level_in_channels = input_shape[in_features[0]].channels
        high_level_in_channels = input_shape[in_features[1]].channels

        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.in_features = in_features

        # Encoder: ASPP on high-level features
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2d(high_level_in_channels, channels, kernel_size=1, norm=norm, activation=activation),
        )
        self.aspp = DepthwiseSeparableASPP(
            dilations=dilations,
            in_channels=high_level_in_channels,
            channels=channels,
            norm=norm,
            activation=activation,
        )
        self.aspp_bottleneck = Conv2d(
            (len(dilations) + 1) * channels,
            channels,
            kernel_size=3,
            norm=norm,
            activation=activation,
        )

        # Decoder: low-level feature processing
        self.low_level_conv = Conv2d(
            low_level_in_channels,
            low_level_channels,
            kernel_size=1,
            norm=norm,
            activation=activation,
        )
        self.decoder = nn.Sequential(
            DepthwiseSeparableConv2d(
                channels + low_level_channels,
                channels,
                kernel_size=3,
                norm=norm,
                activation=activation,
            ),
            DepthwiseSeparableConv2d(
                channels,
                channels,
                kernel_size=3,
                norm=norm,
                activation=activation,
            ),
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
        low_level_feat = bottom_up_features[self.in_features[0]]
        high_level_feat = bottom_up_features[self.in_features[1]]

        # Encoder: ASPP on high-level features
        outs = [F.interpolate(self.image_pool(high_level_feat), size=high_level_feat.size()[2:], mode="bilinear", align_corners=self.align_corners)]
        outs.extend(self.aspp(high_level_feat))
        outs = torch.cat(outs, dim=1)
        outs = self.aspp_bottleneck(outs)

        # Decoder: fuse with low-level features
        outs = F.interpolate(outs, size=low_level_feat.size()[2:], mode="bilinear", align_corners=self.align_corners)
        low_level_feat = self.low_level_conv(low_level_feat)
        outs = torch.cat([outs, low_level_feat], dim=1)
        outs = self.decoder(outs)

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
