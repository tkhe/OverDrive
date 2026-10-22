import math
from collections.abc import Mapping
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from overdrive.layers import Conv2d
from overdrive.layers import ShapeSpec
from overdrive.layers import weight_init

from .base import SemSegHead

__all__ = ["UPerNetHead"]


class PPM(nn.Module):
    """
    Pyramid Pooling Module.
    """

    def __init__(
        self,
        pool_scales: Sequence[int],
        in_channels: int,
        channels: int,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
        align_corners: bool = False,
    ):
        """
        Args:
            pool_scales (Sequence[int]): Pooling scales used in PPM.
            in_channels (int): Input channels.
            channels (int): Output channels for each branch.
            norm (str or dict): Normalization layer.
            activation (str or dict): Activation layer.
            align_corners (bool): Argument of F.interpolate.
        """
        super().__init__()

        poolers = []
        for pool_scale in pool_scales:
            layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_scale),
                Conv2d(in_channels, channels, kernel_size=1, norm=norm, activation=activation),
            )
            poolers.append(layer)

        self.poolers = nn.ModuleList(poolers)
        self.conv = Conv2d(
            in_channels + len(pool_scales) * channels,
            channels,
            kernel_size=3,
            stride=1,
            norm=norm,
            activation=activation,
        )
        self.align_corners = align_corners

    def forward(self, x):
        outs = [x]
        for pool in self.poolers:
            out = pool(x)
            out = F.interpolate(out, size=x.size()[2:], mode="bilinear", align_corners=self.align_corners)
            outs.append(out)
        outs = torch.cat(outs, dim=1)
        outs = self.conv(outs)
        return outs


class UPerNetHead(SemSegHead):
    """
    support UPerNetHead (https://arxiv.org/abs/1807.10221).
    """

    def __init__(
        self,
        input_shape: dict[str, ShapeSpec],
        in_features: list[int],
        *,
        num_classes: int,
        channels: int,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        dropout: float = 0.1,
        ignore_index: int = -1,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
        loss_sem_seg: nn.Module | Mapping[str, nn.Module],
        align_corners: bool = False,
    ):
        """
        Args:
            pool_scales (Sequence[int]): Pooling scales used in PPM.
            dropout (float): Ratio of dropout layer.
            norm (str or dict): Normalization layer.
            activation (str or dict): Activation layer.
            loss_sem_seg (nn.Module or dict[str, nn.Module]): loss(es) for semantic segmentation.
            align_corners (bool): Argument of F.interpolate.
        """
        super().__init__()

        strides = [input_shape[f].stride for f in in_features]
        self.in_channels = [input_shape[f].channels for f in in_features]
        self.align_corners = align_corners
        self.in_features = in_features
        self.ignore_index = ignore_index

        lateral_convs = []
        output_convs = []
        for idx, in_channels in enumerate(self.in_channels):
            if idx == len(self.in_channels) - 1:
                lateral_conv = PPM(pool_scales, self.in_channels[-1], channels, norm, activation, align_corners)
                output_conv = None
            else:
                lateral_conv = Conv2d(in_channels, channels, kernel_size=1, norm=norm, activation=activation)
                output_conv = Conv2d(channels, channels, kernel_size=3, stride=1, norm=norm, activation=activation)
                weight_init.c2_msra_fill(lateral_conv)
                weight_init.c2_msra_fill(output_conv)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"fpn_lateral{stage}", lateral_conv)
            if output_conv is not None:
                self.add_module(f"fpn_output{stage}", output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.conv = Conv2d(
            len(self.in_channels) * channels,
            channels,
            kernel_size=3,
            stride=1,
            norm=norm,
            activation=activation,
        )
        self.dropout = nn.Dropout2d(dropout)
        self.predictor = Conv2d(channels, num_classes, kernel_size=1)

        if isinstance(loss_sem_seg, nn.Module):
            self.loss_sem_seg = loss_sem_seg
        elif isinstance(loss_sem_seg, Mapping):
            self.loss_sem_seg = nn.ModuleDict(loss_sem_seg)
        else:
            raise TypeError(f"loss_sem_seg must be a nn.Module or dict[str, nn.Module], but got {type(loss_sem_seg)}")

    def forward(self, bottom_up_features: dict[str, torch.Tensor]):
        results = []

        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(prev_features)

        for idx, (lateral_conv, output_conv) in enumerate(zip(self.lateral_convs, self.output_convs, strict=True)):
            if idx > 0:
                features = bottom_up_features[self.in_features[-idx - 1]]
                top_down_features = F.interpolate(
                    prev_features,
                    size=features.shape[-2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                results.insert(0, output_conv(prev_features))

        for i in range(1, len(results)):
            results[i] = F.interpolate(
                results[i],
                size=results[0].shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        results = torch.cat(results, dim=1)
        results = self.conv(results)
        results = self.dropout(results)
        results = self.predictor(results)
        return results

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
        return F.interpolate(
            predictions,
            size=image_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )
