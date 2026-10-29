# Copyright (c) OpenMMLab. All rights reserved.
# Adapted for OverDrive from MMDetection (Apache-2.0); see LICENSES/mmdetection.txt.
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from overdrive.layers import Conv2d, ShapeSpec, mask_matrix_nms
from overdrive.modeling.loss import SigmoidDiceLoss, SigmoidFocalLoss
from overdrive.structures import BitMasks, Instances, PolygonMasks
from overdrive.utils.comm import reduce_mean

from .base import DenseHead

__all__ = ["SOLOHead"]


def _coordinate_features(x: torch.Tensor) -> torch.Tensor:
    y = torch.linspace(-1, 1, x.shape[-2], device=x.device, dtype=x.dtype)
    x_coord = torch.linspace(-1, 1, x.shape[-1], device=x.device, dtype=x.dtype)
    yy, xx = torch.meshgrid(y, x_coord, indexing="ij")
    return torch.stack((xx, yy)).unsqueeze(0).expand(x.shape[0], -1, -1, -1)


def _point_nms(logits: torch.Tensor) -> torch.Tensor:
    scores = logits.sigmoid()
    local_max = F.max_pool2d(scores, 2, stride=1, padding=1)[:, :, :-1, :-1]
    return scores * (scores == local_max)


def _init_convs(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight, std=0.01)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class _BaseSOLOHead(DenseHead):
    """Shared grid assignment, losses and mask postprocessing for SOLO heads."""

    def __init__(
        self,
        input_shape: dict[str, ShapeSpec],
        in_features: list[str],
        *,
        num_classes: int,
        num_grids: tuple[int, ...] = (40, 36, 24, 16, 12),
        scale_ranges: tuple[tuple[int, int], ...] = ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        pos_scale: float = 0.2,
        loss_cls: nn.Module | None = None,
        loss_mask: nn.Module | None = None,
        test_score_threshold: float = 0.1,
        test_mask_threshold: float = 0.5,
        test_topk_candidates: int = 500,
        test_filter_threshold: float = 0.05,
        test_nms_kernel: str = "gaussian",
        test_nms_sigma: float = 2.0,
        max_detections_per_image: int = 100,
    ):
        super().__init__()
        if len(in_features) < 2 or not len(in_features) == len(num_grids) == len(scale_ranges):
            raise ValueError("in_features, num_grids and scale_ranges must have the same length (at least two)")
        if len({input_shape[f].channels for f in in_features}) != 1:
            raise ValueError("All SOLO input features must have the same channel count")
        strides = [input_shape[f].stride for f in in_features]
        if any(s is None or s <= 0 for s in strides) or any(b != 2 * a for a, b in zip(strides, strides[1:])):
            raise ValueError("Input strides must be positive and log2 contiguous")
        if num_classes <= 0 or any(g <= 0 for g in num_grids) or pos_scale <= 0:
            raise ValueError("num_classes, grid sizes and pos_scale must be positive")
        if any(low < 0 or high < low for low, high in scale_ranges):
            raise ValueError("scale_ranges must contain ordered, nonnegative bounds")
        if test_nms_kernel not in ("gaussian", "linear"):
            raise ValueError("test_nms_kernel must be 'gaussian' or 'linear'")
        self.in_features = list(in_features)
        self.in_channels = input_shape[in_features[0]].channels
        self.input_strides = strides
        self.strides = [strides[0] * 2, *strides[1:-1], strides[-2]]
        self.num_classes = num_classes
        self.num_grids = tuple(num_grids)
        self.scale_ranges = tuple(scale_ranges)
        self.pos_scale = pos_scale
        self.loss_cls = loss_cls if loss_cls is not None else SigmoidFocalLoss(alpha=0.25, reduction="sum")
        self.loss_mask = loss_mask if loss_mask is not None else SigmoidDiceLoss(reduction="sum", loss_weight=3.0)
        self.test_score_threshold = test_score_threshold
        self.test_mask_threshold = test_mask_threshold
        self.test_topk_candidates = test_topk_candidates
        self.test_filter_threshold = test_filter_threshold
        self.test_nms_kernel = test_nms_kernel
        self.test_nms_sigma = test_nms_sigma
        self.max_detections_per_image = max_detections_per_image

    def _resize_features(self, features):
        feats = [features[f] for f in self.in_features]
        return [
            F.interpolate(feats[0], scale_factor=0.5, mode="bilinear", align_corners=False),
            *feats[1:-1],
            F.interpolate(feats[-1], size=feats[-2].shape[-2:], mode="bilinear", align_corners=False),
        ]

    @torch.no_grad()
    def _get_targets_single(self, instances, mask_sizes, padded_size, *, keep_duplicates=False):
        """Assign mass centers and their scale-limited 3x3 neighborhoods.

        Targets use the batch padded canvas, not the individual image size.
        SOLO overwrites colliding cells; SOLOv2 retains each instance/kernel
        pair, as in MMDetection. Only positive mask targets are materialized.
        """
        device = instances.gt_classes.device
        boxes = instances.gt_boxes.tensor.detach().cpu()
        classes = instances.gt_classes.detach().cpu()
        masks = instances.gt_masks
        if isinstance(masks, PolygonMasks):
            masks = BitMasks.from_polygon_masks(masks, *instances.image_size)
        if isinstance(masks, BitMasks):
            masks = masks.tensor
        if not isinstance(masks, torch.Tensor) or masks.ndim != 3:
            raise TypeError("gt_masks must be BitMasks, PolygonMasks or an (N, H, W) tensor")
        masks = masks.detach().to(device="cpu", dtype=torch.bool)
        if masks.shape[-2] > padded_size[0] or masks.shape[-1] > padded_size[1]:
            raise ValueError("Ground truth masks exceed the padded image size")
        masks = F.pad(masks, (0, padded_size[1] - masks.shape[-1], 0, padded_size[0] - masks.shape[-2]))
        wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0)
        scales = wh.prod(1).sqrt()
        mass = masks.sum((1, 2)).float()
        center_y = (masks.sum(2) * torch.arange(padded_size[0])).sum(1) / mass.clamp_min(1)
        center_x = (masks.sum(1) * torch.arange(padded_size[1])).sum(1) / mass.clamp_min(1)
        all_labels, all_indices, all_targets = [], [], []
        for grid, (low, high), (height, width) in zip(self.num_grids, self.scale_ranges, mask_sizes, strict=True):
            labels = torch.full((grid, grid), self.num_classes, dtype=torch.long)
            assignments = []
            for gt in torch.where((scales >= low) & (scales <= high) & (mass > 0))[0].tolist():
                cy, cx = float(center_y[gt]), float(center_x[gt])
                ry, rx = (wh[gt] * (0.5 * self.pos_scale)).flip(0).tolist()
                cell_y, cell_x = int(cy / padded_size[0] * grid), int(cx / padded_size[1] * grid)
                top = max(0, int((cy - ry) / padded_size[0] * grid), cell_y - 1)
                bottom = min(grid - 1, int((cy + ry) / padded_size[0] * grid), cell_y + 1)
                left = max(0, int((cx - rx) / padded_size[1] * grid), cell_x - 1)
                right = min(grid - 1, int((cx + rx) / padded_size[1] * grid), cell_x + 1)
                labels[top:bottom + 1, left:right + 1] = classes[gt]
                assignments.extend((y * grid + x, gt) for y in range(top, bottom + 1) for x in range(left, right + 1))
            if not keep_duplicates:
                assignments = sorted(dict(assignments).items())
            indices = torch.tensor([idx for idx, _ in assignments], dtype=torch.long, device=device)
            # OpenCV uint8 bilinear resizing matches MMDetection's imrescale;
            # interpolating float masks with torch changes thin-object targets.
            resized = {}
            for _, gt in assignments:
                if gt not in resized:
                    mask = masks[gt].numpy().astype(np.uint8)
                    resized[gt] = torch.from_numpy(cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR))
            targets = torch.stack([resized[gt] for _, gt in assignments]) if assignments else torch.zeros((0, height, width), dtype=torch.uint8)
            all_labels.append(labels.to(device))
            all_indices.append(indices)
            all_targets.append(targets.to(device))
        return all_labels, all_indices, all_targets

    def _classification_loss(self, cls_preds, targets, normalizer):
        logits = torch.cat([x.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for x in cls_preds])
        labels = torch.cat([torch.cat([t[0][level].flatten() for t in targets]) for level in range(len(cls_preds))])
        one_hot = F.one_hot(labels, self.num_classes + 1)[:, :-1].to(logits.dtype)
        return self.loss_cls(logits, one_hot).sum() / (normalizer + 1)

    def _normalizer(self, targets, reference):
        count = sum(len(indices.unique()) for target in targets for indices in target[1])
        return reduce_mean(reference.new_tensor(float(count), dtype=torch.float32))

    def _candidate_scores(self, cls_preds, image_id):
        scores = torch.cat([_point_nms(x[image_id:image_id + 1])[0].permute(1, 2, 0).reshape(-1, self.num_classes) for x in cls_preds])
        cells, labels = torch.where(scores > self.test_score_threshold)
        strides = scores.new_tensor(self.strides).repeat_interleave(scores.new_tensor(self.num_grids, dtype=torch.long).square())
        return scores[cells, labels], labels, cells, strides[cells]

    def _inference_single(self, probs, scores, labels, strides, image_size, padded_size):
        probs = probs.float()
        masks = probs > self.test_mask_threshold
        areas = masks.sum((1, 2)).float()
        keep = areas > strides
        probs, masks, areas, scores, labels = probs[keep], masks[keep], areas[keep], scores[keep], labels[keep]
        scores = scores * (probs * masks).sum((1, 2)) / areas
        scores, labels, _, keep = mask_matrix_nms(
            masks, labels, scores, mask_area=areas,
            nms_pre=self.test_topk_candidates, max_num=self.max_detections_per_image,
            kernel=self.test_nms_kernel, sigma=self.test_nms_sigma, filter_thr=self.test_filter_threshold,
        )
        probs = probs[keep]
        if len(probs):
            probs = F.interpolate(probs.unsqueeze(0), size=padded_size, mode="bilinear", align_corners=False)[0]
            probs = probs[:, :image_size[0], :image_size[1]]
        else:
            probs = probs.new_empty((0, *image_size))
        # These are full-image probabilities. The meta architecture restores
        # the requested resolution, thresholds masks, and computes tight boxes.
        return Instances(image_size, scores=scores, pred_classes=labels, pred_masks=probs)


class SOLOHead(_BaseSOLOHead):
    """SOLO with shared category/mask towers and one mask per grid cell.

    Forward returns raw ``(mask_logits, class_logits)`` in both train and eval
    modes. ``loss`` expects Instances with gt_boxes, gt_classes and gt_masks.
    Loss modules must use sum or none reduction (the head normalizes positives).
    """

    def __init__(self, input_shape, in_features, *, feat_channels=256, stacked_convs=7,
                 norm="GN", cls_down_index=0, prior_prob=0.01, **kwargs):
        super().__init__(input_shape, in_features, **kwargs)
        if stacked_convs <= 0 or not 0 <= cls_down_index < stacked_convs:
            raise ValueError("cls_down_index must index a nonempty convolution tower")
        if not 0 < prior_prob < 1:
            raise ValueError("prior_prob must be in (0, 1)")
        self.cls_down_index = cls_down_index
        self.mask_convs = nn.Sequential(*[
            Conv2d(self.in_channels + 2 if i == 0 else feat_channels, feat_channels, 3, norm=norm, activation="ReLU", bias=not bool(norm))
            for i in range(stacked_convs)
        ])
        self.cls_convs = nn.ModuleList([
            Conv2d(self.in_channels if i == 0 else feat_channels, feat_channels, 3, norm=norm, activation="ReLU", bias=not bool(norm))
            for i in range(stacked_convs)
        ])
        self.conv_mask_list = nn.ModuleList([nn.Conv2d(feat_channels, grid * grid, 1) for grid in self.num_grids])
        self.conv_cls = nn.Conv2d(feat_channels, self.num_classes, 3, padding=1)
        _init_convs(self)
        for layer in [self.conv_cls, *self.conv_mask_list]:
            nn.init.constant_(layer.bias, math.log(prior_prob / (1 - prior_prob)))

    def forward(self, features):
        mask_preds, cls_preds = [], []
        for x, grid, mask_layer in zip(self._resize_features(features), self.num_grids, self.conv_mask_list, strict=True):
            mask_features = self.mask_convs(torch.cat((x, _coordinate_features(x)), dim=1))
            mask_features = F.interpolate(mask_features, scale_factor=2, mode="bilinear", align_corners=False)
            mask_preds.append(mask_layer(mask_features))
            for i, layer in enumerate(self.cls_convs):
                if i == self.cls_down_index:
                    x = F.interpolate(x, size=(grid, grid), mode="bilinear", align_corners=False)
                x = layer(x)
            cls_preds.append(self.conv_cls(x))
        return mask_preds, cls_preds

    def loss(self, mask_preds, cls_preds, gt_instances, padded_size):
        sizes = [x.shape[-2:] for x in mask_preds]
        targets = [self._get_targets_single(gt, sizes, padded_size) for gt in gt_instances]
        normalizer = self._normalizer(targets, cls_preds[0])
        # Keep every level connected on empty batches for DDP without unused parameters.
        loss_mask = sum(x.reshape(-1)[:0].sum() for x in mask_preds)
        for image, (_, indices, masks) in enumerate(targets):
            for level, index in enumerate(indices):
                if len(index):
                    loss_mask = loss_mask + self.loss_mask(mask_preds[level][image, index], masks[level]).sum()
        return {"loss_mask": loss_mask / normalizer.clamp_min(1), "loss_cls": self._classification_loss(cls_preds, targets, normalizer)}

    @torch.no_grad()
    def inference(self, mask_preds, cls_preds, image_sizes, padded_size):
        results = []
        common_size = mask_preds[0].shape[-2:]
        for image, image_size in enumerate(image_sizes):
            scores, labels, cells, strides = self._candidate_scores(cls_preds, image)
            # Select first, but retain the order of (cell, class) candidates.
            probs = []
            offset = 0
            for grid, pred in zip(self.num_grids, mask_preds, strict=True):
                selected = cells[(cells >= offset) & (cells < offset + grid * grid)] - offset
                p = pred[image, selected].sigmoid()
                if len(p):
                    p = F.interpolate(p.unsqueeze(0), size=common_size, mode="bilinear", align_corners=False)[0]
                else:
                    p = p.new_empty((0, *common_size))
                probs.append(p)
                offset += grid * grid
            results.append(self._inference_single(torch.cat(probs), scores, labels, strides, image_size, padded_size))
        return results
