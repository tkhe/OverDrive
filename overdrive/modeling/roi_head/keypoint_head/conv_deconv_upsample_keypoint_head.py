from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from overdrive.layers import Conv2d
from overdrive.layers import ShapeSpec
from overdrive.layers import cat
from overdrive.layers import weight_init
from overdrive.structures import Instances
from overdrive.structures import heatmaps_to_keypoints
from overdrive.utils.events import get_event_storage

__all__ = ["ConvDeconvUpsampleKeypointHead"]

_TOTAL_SKIPPED = 0


class ConvDeconvUpsampleKeypointHead(nn.Module):
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_keypoints: int,
        conv_dims: list[int],
        norm: str | dict = "",
        activation: str | dict = "ReLU",
        loss_keypoint: nn.Module,
        loss_normalizer: str | float = "visible",
    ):
        super().__init__()

        if loss_normalizer != "visible" and not isinstance(loss_normalizer, float):
            raise TypeError(loss_normalizer)

        in_channels = input_shape.channels

        layers = OrderedDict()
        for idx, out_channels in enumerate(conv_dims, start=1):
            conv = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, norm=norm, activation=activation)
            weight_init.c2_msra_fill(conv)
            layers[f"conv{idx}"] = conv
            in_channels = out_channels

        self.layers = nn.Sequential(layers) if len(layers) > 0 else None
        self.score_lowres = nn.ConvTranspose2d(in_channels, num_keypoints, 4, stride=2, padding=1)

        self.up_scale = 2.0
        self.num_keypoints = num_keypoints
        self.loss_normalizer = loss_normalizer

        self.loss_keypoint = loss_keypoint

        weight_init.c2_msra_fill(self.score_lowres)

    def forward(self, x):
        if self.layers is not None:
            x = self.layers(x)

        x = self.score_lowres(x)
        x = F.interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        return x

    def loss(self, pred_keypoint_logits: torch.Tensor, instances: list[Instances]):
        heatmaps = []
        valid = []

        keypoint_side_len = pred_keypoint_logits.shape[2]
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            keypoints = instances_per_image.gt_keypoints
            heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
                instances_per_image.proposal_boxes.tensor,
                keypoint_side_len,
            )
            heatmaps.append(heatmaps_per_image.view(-1))
            valid.append(valid_per_image.view(-1))

        if len(heatmaps):
            keypoint_targets = cat(heatmaps, dim=0)
            valid = cat(valid, dim=0).to(dtype=torch.uint8)
            valid = torch.nonzero(valid).squeeze(1)

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if len(heatmaps) == 0 or valid.numel() == 0:
            global _TOTAL_SKIPPED

            _TOTAL_SKIPPED += 1
            storage = get_event_storage()
            storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
            return {"loss_keypoint": pred_keypoint_logits.sum() * 0}

        N, K, H, W = pred_keypoint_logits.shape
        pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)

        keypoint_loss = self.loss_keypoint(pred_keypoint_logits[valid], keypoint_targets[valid])

        num_images = len(instances)
        normalizer = None if self.loss_normalizer == "visible" else num_images * self.loss_normalizer
        # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
        if normalizer is None:
            normalizer = valid.numel()

        return {"loss_keypoint": keypoint_loss / normalizer}

    def inference(self, pred_keypoint_logits: torch.Tensor, pred_instances: list[Instances]):
        # flatten all bboxes from all images together (list[Boxes] -> Rx4 tensor)
        bboxes_flat = cat([b.pred_boxes.tensor for b in pred_instances], dim=0)

        pred_keypoint_logits = pred_keypoint_logits.detach()
        keypoint_results = heatmaps_to_keypoints(pred_keypoint_logits, bboxes_flat.detach())
        num_instances_per_image = [len(i) for i in pred_instances]
        keypoint_results = keypoint_results[:, :, [0, 1, 3]].split(num_instances_per_image, dim=0)
        heatmap_results = pred_keypoint_logits.split(num_instances_per_image, dim=0)

        for keypoint_results_per_image, heatmap_results_per_image, instances_per_image in zip(
            keypoint_results,
            heatmap_results,
            pred_instances,
            strict=True,
        ):
            # keypoint_results_per_image is (num instances)x(num keypoints)x(x, y, score)
            # heatmap_results_per_image is (num instances)x(num keypoints)x(side)x(side)
            instances_per_image.pred_keypoints = keypoint_results_per_image
            instances_per_image.pred_keypoint_heatmaps = heatmap_results_per_image

        return pred_instances
