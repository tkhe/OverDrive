# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from MMDetection for OverDrive; see LICENSES/mmdetection.txt.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from overdrive.layers import Conv2d, ShapeSpec, fast_nms, permute_to_N_HWA_K
from overdrive.modeling.anchor_generator import AnchorGenerator
from overdrive.modeling.box_coder import BoxCoder, DeltaXYWHBoxCoder
from overdrive.modeling.matcher import Matcher
from overdrive.structures import BitMasks, Boxes, Instances, PolygonMasks, pairwise_iou
from overdrive.utils.comm import reduce_mean

from .base import DenseHead

__all__ = ["YOLACTHead", "YOLACTProtonet"]


def _init_convs(module):
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class YOLACTHead(DenseHead):
    """Shared detection tower with softmax classes, box deltas and coefficients.

    The caller supplies the anchor generator and matcher. OHEM defaults to a
    3:1 negative/positive ratio. Anchor spacing and sizes are configured by
    the generator; no input resolution is hardcoded in this head.

    Foreground labels are 0..num_classes-1; background is num_classes.
    A custom matcher uses the repository's {-1, 0, 1} match labels. With
    match_low_quality=True, the best valid anchor for each GT is additionally
    assigned to that GT, matching MMDetection's gt_max_assign_all=False.
    """

    def __init__(
        self,
        input_shape: dict[str, ShapeSpec],
        in_features: list[str],
        *,
        num_classes: int,
        anchor_generator: AnchorGenerator,
        matcher: Matcher,
        num_protos: int = 32,
        conv_dims: tuple[int, ...] = (256,),
        norm: str | dict = "",
        activation: str | dict = "ReLU",
        box_coder: BoxCoder | None = None,
        match_low_quality: bool = True,
        neg_pos_ratio: int = 3,
        loss_cls_weight: float = 1.0,
        loss_reg_weight: float = 1.5,
        smooth_l1_beta: float = 1.0,
        test_score_threshold: float = 0.05,
        test_topk_candidates: int = 1000,
        test_nms_threshold: float = 0.5,
        test_nms_topk: int = 200,
        max_detections_per_image: int = 100,
    ):
        super().__init__()
        self.in_features = list(in_features)
        self.num_classes = num_classes
        self.num_protos = num_protos
        self.anchor_generator = anchor_generator
        self.box_coder = box_coder if box_coder is not None else DeltaXYWHBoxCoder((10.0, 10.0, 5.0, 5.0))
        self.matcher = matcher
        self.match_low_quality = match_low_quality
        self.neg_pos_ratio = neg_pos_ratio
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_threshold = test_score_threshold
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_threshold = test_nms_threshold
        self.test_nms_topk = test_nms_topk
        self.max_detections_per_image = max_detections_per_image
        channels = input_shape[in_features[0]].channels
        self.head_convs = nn.Sequential(
            *[Conv2d(in_dim, out_dim, 3, norm=norm, activation=activation) for in_dim, out_dim in zip([channels, *conv_dims[:-1]], conv_dims)]
        )
        anchors_per_cell = anchor_generator.num_anchors[0]
        self.conv_cls = nn.Conv2d(conv_dims[-1], anchors_per_cell * (num_classes + 1), 3, padding=1)
        self.conv_reg = nn.Conv2d(conv_dims[-1], anchors_per_cell * 4, 3, padding=1)
        self.conv_coeff = nn.Conv2d(conv_dims[-1], anchors_per_cell * num_protos, 3, padding=1)
        _init_convs(self)

    def forward(self, features):
        cls_preds, box_preds, coeff_preds = [], [], []
        for name in self.in_features:
            x = self.head_convs(features[name])
            cls_preds.append(self.conv_cls(x))
            box_preds.append(self.conv_reg(x))
            coeff_preds.append(self.conv_coeff(x).tanh())
        return cls_preds, box_preds, coeff_preds

    def _valid_flags(self, predictions, image_size):
        """Keep full anchor indexing while excluding each image's batch padding."""
        flags = []
        for pred, stride, count in zip(predictions, self.anchor_generator.strides, self.anchor_generator.num_anchors):
            height, width = pred.shape[-2:]
            valid_y = torch.arange(height, device=pred.device) < math.ceil(image_size[0] / stride)
            valid_x = torch.arange(width, device=pred.device) < math.ceil(image_size[1] / stride)
            flags.append((valid_y[:, None] & valid_x[None, :]).flatten().repeat_interleave(count))
        return torch.cat(flags)

    @torch.no_grad()
    def _assign(self, anchors, valid, gt):
        indices = torch.where(valid)[0]
        labels = torch.full((len(anchors),), -1, dtype=torch.long, device=anchors.device)
        matched = torch.zeros_like(labels)
        if not len(indices):
            return labels, matched
        if not len(gt):
            labels[indices] = self.num_classes
            return labels, matched
        quality = pairwise_iou(gt.gt_boxes, Boxes(anchors[indices]))
        matches, match_labels = self.matcher.match(quality)
        if self.match_low_quality:
            # Exactly one best anchor per GT; later GTs win collisions.
            for gt_index, best_anchor in enumerate(quality.argmax(dim=1)):
                matches[best_anchor] = gt_index
                match_labels[best_anchor] = 1
        local_labels = gt.gt_classes[matches].clone()
        local_labels[match_labels == 0] = self.num_classes
        local_labels[match_labels == -1] = -1
        labels[indices], matched[indices] = local_labels, matches
        return labels, matched

    def loss(self, cls_preds, box_preds, coeff_preds, gt_instances, *, return_positive_info=False):
        """Compute OHEM/box losses, optionally returning differentiable positives.

        Positive Instances contain coeffs, gt_boxes and gt_indices. Returning
        these explicitly avoids hidden state between the detection/mask heads.
        """
        anchors = Boxes.cat(self.anchor_generator(cls_preds)).tensor
        logits = torch.cat([permute_to_N_HWA_K(x, self.num_classes + 1) for x in cls_preds], dim=1).float()
        deltas = torch.cat([permute_to_N_HWA_K(x, 4) for x in box_preds], dim=1).float()
        coeffs = torch.cat([permute_to_N_HWA_K(x, self.num_protos) for x in coeff_preds], dim=1)
        loss_cls = logits.reshape(-1)[:0].sum()
        loss_reg = deltas.reshape(-1)[:0].sum()
        positive_infos, num_positive = [], 0
        for image, gt in enumerate(gt_instances):
            valid = self._valid_flags(cls_preds, gt.image_size)
            labels, matched = self._assign(anchors, valid, gt)
            positive = torch.where((labels >= 0) & (labels < self.num_classes))[0]
            negative = torch.where(labels == self.num_classes)[0]
            num_positive += len(positive)
            per_anchor_loss = F.cross_entropy(logits[image], labels, ignore_index=-1, reduction="none")
            num_neg = min(len(negative), self.neg_pos_ratio * len(positive)) if len(positive) else len(negative)
            loss_cls = loss_cls + per_anchor_loss[positive].sum() + per_anchor_loss[negative].topk(num_neg).values.sum()
            gt_indices = matched[positive]
            boxes = gt.gt_boxes[gt_indices]
            if len(positive):
                target_deltas = self.box_coder.get_deltas(anchors[positive], boxes.tensor)
                loss_reg = loss_reg + F.smooth_l1_loss(deltas[image, positive], target_deltas, beta=self.smooth_l1_beta, reduction="sum")
            positive_infos.append(Instances(gt.image_size, coeffs=coeffs[image, positive], gt_boxes=boxes, gt_indices=gt_indices))
        normalizer = reduce_mean(logits.new_tensor(float(num_positive))).clamp_min(1)
        losses = {"loss_cls": loss_cls * self.loss_cls_weight / normalizer, "loss_reg": loss_reg * self.loss_reg_weight / normalizer}
        return (losses, positive_infos) if return_positive_info else losses

    @torch.no_grad()
    def inference(self, cls_preds, box_preds, coeff_preds, image_sizes):
        anchors = self.anchor_generator(cls_preds)
        valid_flags = [self._valid_flags(cls_preds, size).split([len(a) for a in anchors]) for size in image_sizes]
        cls_preds = [permute_to_N_HWA_K(x, self.num_classes + 1) for x in cls_preds]
        box_preds = [permute_to_N_HWA_K(x, 4) for x in box_preds]
        coeff_preds = [permute_to_N_HWA_K(x, self.num_protos) for x in coeff_preds]
        results = []
        for image, size in enumerate(image_sizes):
            all_boxes, all_scores, all_coeffs = [], [], []
            for anchor, cls, box, coeff, valid in zip(anchors, cls_preds, box_preds, coeff_preds, valid_flags[image]):
                scores = cls[image].float().softmax(dim=-1)
                best_score = scores[:, :-1].max(dim=1).values
                keep = torch.where((best_score > self.test_score_threshold) & valid)[0]
                if self.test_topk_candidates > 0:
                    keep = keep[best_score[keep].argsort(descending=True)[: self.test_topk_candidates]]
                decoded = Boxes(self.box_coder.apply_deltas(box[image, keep], anchor.tensor[keep]))
                decoded.clip(size)
                nonempty = decoded.nonempty()
                all_boxes.append(decoded.tensor[nonempty])
                all_scores.append(scores[keep][nonempty])
                all_coeffs.append(coeff[image, keep][nonempty])
            dets, labels, coefficients = fast_nms(
                torch.cat(all_boxes),
                torch.cat(all_scores),
                torch.cat(all_coeffs),
                self.test_score_threshold,
                self.test_nms_threshold,
                self.test_nms_topk,
                self.max_detections_per_image,
            )
            results.append(Instances(size, pred_boxes=Boxes(dets[:, :4]), scores=dets[:, 4], pred_classes=labels, coeffs=coefficients))
        return results


class YOLACTProtonet(DenseHead):
    """Shared prototypes and optional training-only semantic segmentation.

    Uses the finest detection FPN feature (usually p3), three 3x3 convolutions,
    2x bilinear upsampling, another 3x3 convolution and a K-channel projection.
    All crop coordinates and mask targets refer to the padded batch canvas.
    """

    def __init__(
        self,
        input_shape: dict[str, ShapeSpec],
        in_feature: str,
        *,
        num_classes: int,
        num_protos: int = 32,
        feat_channels: int = 256,
        with_seg_branch: bool = True,
        include_last_relu: bool = True,
        max_masks_to_train: int = 100,
        loss_mask_weight: float = 6.125,
        loss_segm_weight: float = 1.0,
        test_mask_threshold: float = 0.5,
    ):
        super().__init__()
        self.in_feature = in_feature
        self.num_classes = num_classes
        self.num_protos = num_protos
        self.max_masks_to_train = max_masks_to_train
        self.loss_mask_weight = loss_mask_weight
        self.loss_segm_weight = loss_segm_weight
        self.test_mask_threshold = test_mask_threshold
        channels = input_shape[in_feature].channels
        layers = []
        for i in range(3):
            layers.extend([nn.Conv2d(channels if i == 0 else feat_channels, feat_channels, 3, padding=1), nn.ReLU(inplace=True)])
        layers.extend(
            [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_channels, num_protos, 1),
            ]
        )
        if include_last_relu:
            layers.append(nn.ReLU(inplace=True))
        self.protonet = nn.Sequential(*layers)
        self.segm_branch = nn.Conv2d(channels, num_classes, 1) if with_seg_branch else None
        _init_convs(self)

    def forward(self, features):
        x = features[self.in_feature]
        prototypes = self.protonet(x)
        semantic_logits = self.segm_branch(x) if self.training and self.segm_branch is not None else None
        return prototypes, semantic_logits

    @staticmethod
    def combine_masks(prototypes, coefficients):
        """(K,H,W) prototypes and (N,K) coefficients -> (N,H,W) probabilities."""
        # BCE on sigmoid probabilities must also run outside AMP, below.
        with torch.autocast(device_type=prototypes.device.type, enabled=False):
            return torch.einsum("khw,nk->nhw", prototypes.float(), coefficients.float()).sigmoid()

    @staticmethod
    def crop_masks(masks, boxes, padded_size, padding=1):
        """Zero probabilities outside boxes, allowing one prototype pixel margin."""
        height, width = masks.shape[-2:]
        scale = boxes.new_tensor([width / padded_size[1], height / padded_size[0]] * 2)
        boxes = boxes.float() * scale
        lower = torch.minimum(boxes[:, :2], boxes[:, 2:]) - padding
        upper = torch.maximum(boxes[:, :2], boxes[:, 2:]) + padding
        x = torch.arange(width, device=masks.device)[None, None, :]
        y = torch.arange(height, device=masks.device)[None, :, None]
        keep = (x >= lower[:, 0, None, None]) & (x < upper[:, 0, None, None])
        keep = keep & (y >= lower[:, 1, None, None]) & (y < upper[:, 1, None, None])
        return masks * keep

    @staticmethod
    def _gt_masks(gt, padded_size, device):
        masks = gt.gt_masks
        if isinstance(masks, PolygonMasks):
            masks = BitMasks.from_polygon_masks(masks, *gt.image_size)
        if isinstance(masks, BitMasks):
            masks = masks.tensor
        height, width = masks.shape[-2:]
        return F.pad(masks.to(device=device, dtype=torch.float32), (0, padded_size[1] - width, 0, padded_size[0] - height))

    def loss(self, prototypes, semantic_logits, positive_infos, gt_instances, padded_size):
        loss_mask = prototypes.reshape(-1)[:0].sum().float()
        loss_segm = semantic_logits.reshape(-1)[:0].sum().float() if semantic_logits is not None else None
        num_masks = 0
        for image, (positive, gt) in enumerate(zip(positive_infos, gt_instances)):
            # Keep the coefficient tower connected even on all-background batches.
            loss_mask = loss_mask + positive.coeffs.reshape(-1)[:0].sum().float()
            masks = self._gt_masks(gt, padded_size, prototypes.device)
            if semantic_logits is not None:
                target = torch.zeros_like(semantic_logits[image], dtype=torch.float32)
                if len(gt):
                    resized = F.interpolate(masks[:, None], size=target.shape[-2:], mode="bilinear", align_corners=False)[:, 0] > 0.5
                    for index, label in enumerate(gt.gt_classes):
                        # OverDrive labels are zero-based, including foreground class 0.
                        target[label] = torch.maximum(target[label], resized[index].float())
                loss_segm = loss_segm + F.binary_cross_entropy_with_logits(semantic_logits[image].float(), target, reduction="sum")
            selected = torch.arange(len(positive), device=prototypes.device)
            if len(selected) > self.max_masks_to_train:
                selected = torch.randperm(len(selected), device=selected.device)[: self.max_masks_to_train]
            positive = positive[selected]
            if not len(positive):
                continue
            num_masks += len(positive)
            probs = self.combine_masks(prototypes[image], positive.coeffs)
            probs = self.crop_masks(probs, positive.gt_boxes.tensor, padded_size)
            targets = F.interpolate(masks[positive.gt_indices, None], size=probs.shape[-2:], mode="bilinear", align_corners=False)[:, 0] > 0.5
            with torch.autocast(device_type=prototypes.device.type, enabled=False):
                pixel_loss = F.binary_cross_entropy(probs, targets.float(), reduction="none")
            wh = positive.gt_boxes.tensor[:, 2:] - positive.gt_boxes.tensor[:, :2]
            relative_area = wh.prod(dim=1) / (padded_size[0] * padded_size[1])
            loss_mask = loss_mask + (pixel_loss.mean(dim=(1, 2)) / relative_area.clamp_min(1e-6)).sum()
        normalizer = reduce_mean(prototypes.new_tensor(float(num_masks), dtype=torch.float32)).clamp_min(1)
        losses = {"loss_mask": loss_mask * self.loss_mask_weight / normalizer}
        if semantic_logits is not None:
            batch, _, height, width = semantic_logits.shape
            losses["loss_segm"] = loss_segm * self.loss_segm_weight / (batch * height * width)
        return losses

    @torch.no_grad()
    def inference(self, prototypes, detections, padded_size):
        """Attach full-image mask probabilities, retaining the detected boxes."""
        for image, result in enumerate(detections):
            if len(result):
                masks = self.combine_masks(prototypes[image], result.coeffs)
                masks = self.crop_masks(masks, result.pred_boxes.tensor, padded_size)
                masks = F.interpolate(masks[None], size=padded_size, mode="bilinear", align_corners=False)[0]
                masks = masks[:, : result.image_size[0], : result.image_size[1]]
            else:
                masks = prototypes.new_empty((0, *result.image_size), dtype=torch.float32)
            result.mask_probs = masks
            result.remove("coeffs")
        return detections
