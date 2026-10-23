import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from overdrive.layers import get_activation
from overdrive.layers import inverse_sigmoid
from overdrive.layers import weight_init
from overdrive.modeling.dense_head import DenseHead
from overdrive.modeling.matcher import Matcher
from overdrive.structures import BoxMode
from overdrive.structures import Boxes
from overdrive.structures import Instances
from overdrive.utils.comm import reduce_mean

from dn_detr.modeling.denoising import DNMetadata

__all__ = ["DNDETRHead"]


class FFN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str | dict = "ReLU",
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = get_activation(activation)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x


class DNDETRHead(DenseHead):
    """
    Predict boxes and supervise matching and denoising queries.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        embed_dim: int = 256,
        activation: str | dict = "ReLU",
        matcher: Matcher,
        loss_cls: nn.Module,
        loss_iou: nn.Module,
        loss_reg: nn.Module,
        use_auxiliary_loss: bool = True,
        test_score_threshold: float = 0.0,
        max_detections_per_image: int = 300,
        prior_prob: float = 0.01,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_auxiliary_loss = use_auxiliary_loss
        self.test_score_threshold = test_score_threshold
        self.max_detections_per_image = max_detections_per_image

        self.fc_cls = nn.Linear(embed_dim, num_classes)
        self.fc_reg = FFN(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=4,
            activation=activation,
        )

        self.matcher = matcher

        self.loss_cls = loss_cls
        self.loss_iou = loss_iou
        self.loss_reg = loss_reg

        bias_value = -math.log((1 - prior_prob) / prior_prob)
        weight_init.bias_fill(self.fc_cls, bias_value)

        weight_init.constant_fill(self.fc_reg.fc3, val=0, bias=0)

    def forward(self, hidden_states: torch.Tensor, references: torch.Tensor):
        cls_scores = self.fc_cls(hidden_states)

        references_unsigmoid = inverse_sigmoid(references)
        anchor_box_offsets = self.fc_reg(hidden_states)
        bbox_preds = (references_unsigmoid + anchor_box_offsets).sigmoid()

        return cls_scores, bbox_preds

    def loss(
        self,
        all_layers_cls_scores: torch.Tensor,
        all_layers_bbox_preds: torch.Tensor,
        gt_instances: list[Instances],
        dn_metadata: DNMetadata,
    ):
        losses = {}

        pad_size = dn_metadata.pad_size
        num_gt = sum(len(gt.gt_classes) for gt in gt_instances)
        valid_dn_count = num_gt * dn_metadata.num_groups
        dn_normalizer = all_layers_cls_scores.new_tensor([num_gt])
        dn_normalizer = torch.clamp(reduce_mean(dn_normalizer), min=1).item() * dn_metadata.num_groups

        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]
        match_cls_scores = cls_scores[:, pad_size:]
        match_bbox_preds = bbox_preds[:, pad_size:]

        gt_labels, bbox_targets = self.get_ground_truth(match_cls_scores, match_bbox_preds, gt_instances)
        loss_cls, loss_iou, loss_reg = self.calculate_loss(match_cls_scores, match_bbox_preds, gt_labels, bbox_targets)
        loss_cls_dn, loss_iou_dn, loss_reg_dn = self.calculate_dn_loss(
            cls_scores[:, :pad_size],
            bbox_preds[:, :pad_size],
            gt_instances,
            dn_metadata,
            valid_dn_count,
            dn_normalizer,
        )
        losses["loss_cls"] = loss_cls
        losses["loss_iou"] = loss_iou
        losses["loss_reg"] = loss_reg
        losses["loss_cls_dn"] = loss_cls_dn
        losses["loss_iou_dn"] = loss_iou_dn
        losses["loss_reg_dn"] = loss_reg_dn

        if self.use_auxiliary_loss:
            for idx, (cls_scores, bbox_preds) in enumerate(zip(all_layers_cls_scores[:-1], all_layers_bbox_preds[:-1], strict=True)):
                match_cls_scores = cls_scores[:, pad_size:]
                match_bbox_preds = bbox_preds[:, pad_size:]

                gt_labels, bbox_targets = self.get_ground_truth(match_cls_scores, match_bbox_preds, gt_instances)
                loss_cls, loss_iou, loss_reg = self.calculate_loss(match_cls_scores, match_bbox_preds, gt_labels, bbox_targets)
                loss_cls_dn, loss_iou_dn, loss_reg_dn = self.calculate_dn_loss(
                    cls_scores[:, :pad_size],
                    bbox_preds[:, :pad_size],
                    gt_instances,
                    dn_metadata,
                    valid_dn_count,
                    dn_normalizer,
                )
                losses[f"aux.{idx}.loss_cls"] = loss_cls
                losses[f"aux.{idx}.loss_iou"] = loss_iou
                losses[f"aux.{idx}.loss_reg"] = loss_reg
                losses[f"aux.{idx}.loss_cls_dn"] = loss_cls_dn
                losses[f"aux.{idx}.loss_iou_dn"] = loss_iou_dn
                losses[f"aux.{idx}.loss_reg_dn"] = loss_reg_dn

        return losses

    def calculate_loss(self, cls_scores, bbox_preds, gt_labels, bbox_targets):
        gt_labels = torch.stack(gt_labels)

        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)

        num_pos = pos_mask.sum().item()
        num_pos = cls_scores.new_tensor([num_pos])
        num_pos = torch.clamp(reduce_mean(num_pos), min=1).item()

        cls_targets = F.one_hot(gt_labels, num_classes=self.num_classes + 1)[..., :-1].to(dtype=cls_scores.dtype)
        loss_cls = self.loss_cls(cls_scores, cls_targets) / num_pos

        bbox_targets = torch.stack(bbox_targets).float()

        # Decode in FP32 outside no_grad so GIoU supervises box regression.
        pred_boxes = BoxMode.convert(bbox_preds[pos_mask].to(bbox_targets.dtype), BoxMode.CXCYWH_REL, BoxMode.XYXY_REL)
        gt_boxes = BoxMode.convert(bbox_targets[pos_mask], BoxMode.CXCYWH_REL, BoxMode.XYXY_REL)
        loss_iou = self.loss_iou(pred_boxes, gt_boxes) / num_pos
        loss_reg = self.loss_reg(bbox_preds[pos_mask].to(bbox_targets.dtype), bbox_targets[pos_mask]) / num_pos

        return loss_cls, loss_iou, loss_reg

    def calculate_dn_loss(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        gt_instances: list[Instances],
        metadata: DNMetadata,
        valid_count: int,
        normalizer: float,
    ):
        # Unassigned DN slots are background queries, as in detrex's SetCriterion.
        cls_targets = torch.zeros_like(cls_scores)
        for image_idx, (gt, output_idx, target_idx) in enumerate(zip(gt_instances, metadata.output_indices, metadata.target_indices, strict=True)):
            gt_classes = F.one_hot(gt.gt_classes[target_idx], num_classes=self.num_classes)
            cls_targets[image_idx, output_idx] = gt_classes.to(cls_scores.dtype)
        loss_cls = self.loss_cls(cls_scores, cls_targets) / normalizer

        if valid_count == 0:
            zero = cls_scores.sum() * 0 + bbox_preds.sum() * 0
            return loss_cls, zero, zero

        selected_bbox = torch.cat([bbox_preds[image_idx, output_idx] for image_idx, output_idx in enumerate(metadata.output_indices)], dim=0)
        target_boxes = []
        for gt, target_idx in zip(gt_instances, metadata.target_indices, strict=True):
            height, width = gt.image_size
            scale = selected_bbox.new_tensor((width, height, width, height))
            xyxy = gt.gt_boxes.tensor[target_idx] / scale
            target_boxes.append(BoxMode.convert(xyxy, BoxMode.XYXY_REL, BoxMode.CXCYWH_REL))
        target_boxes = torch.cat(target_boxes, dim=0)

        loss_reg = self.loss_reg(selected_bbox, target_boxes) / normalizer
        pred_xyxy = BoxMode.convert(selected_bbox, BoxMode.CXCYWH_REL, BoxMode.XYXY_REL)
        target_xyxy = BoxMode.convert(target_boxes, BoxMode.CXCYWH_REL, BoxMode.XYXY_REL)
        loss_iou = self.loss_iou(pred_xyxy, target_xyxy) / normalizer

        return loss_cls, loss_iou, loss_reg

    @torch.no_grad()
    def get_ground_truth(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        gt_instances: list[Instances],
    ):
        gt_labels = []
        bbox_targets = []
        for cls_score, bbox_pred, gt_per_image in zip(cls_scores, bbox_preds, gt_instances, strict=True):
            height, width = gt_per_image.image_size
            scale = bbox_pred.new_tensor((width, height, width, height)).unsqueeze(0)
            pred_xyxy = BoxMode.convert(bbox_pred, BoxMode.CXCYWH_REL, BoxMode.XYXY_REL) * scale

            pred_instances = Instances((height, width))
            pred_instances.pred_scores = cls_score
            pred_instances.pred_boxes = Boxes(pred_xyxy)
            matched_indices, matched_labels = self.matcher.match(pred_instances, gt_per_image)

            if len(gt_per_image) > 0:
                matched_boxes = gt_per_image.gt_boxes.tensor[matched_indices]
                labels = gt_per_image.gt_classes[matched_indices]
                labels[matched_labels == 0] = self.num_classes
            else:
                matched_boxes = torch.zeros_like(pred_xyxy)
                labels = torch.full_like(matched_indices, self.num_classes)

            gt_labels.append(labels)
            bbox_targets.append(BoxMode.convert(matched_boxes / scale, BoxMode.XYXY_REL, BoxMode.CXCYWH_REL))

        return gt_labels, bbox_targets

    def inference(self, cls_scores: torch.Tensor, bbox_preds: torch.Tensor, image_sizes: list[tuple[int, int]]):
        results = []

        cls_scores = cls_scores[-1]
        bbox_preds = bbox_preds[-1]

        for cls_score, bbox_pred, image_size in zip(cls_scores, bbox_preds, image_sizes, strict=True):
            result = self.inference_single_image(cls_score, bbox_pred, image_size)
            results.append(result)

        return results

    def inference_single_image(self, cls_score: torch.Tensor, bbox_pred: torch.Tensor, image_size: tuple[int, int]):
        cls_score = cls_score.sigmoid()
        scores, indices = cls_score.flatten().topk(min(self.max_detections_per_image, cls_score.numel()))
        class_idxs = indices % self.num_classes

        bbox_pred = bbox_pred[indices // self.num_classes]

        bbox_pred = BoxMode.convert(bbox_pred, BoxMode.CXCYWH_REL, BoxMode.XYXY_REL)
        bbox_pred[:, 0::2] *= image_size[1]
        bbox_pred[:, 1::2] *= image_size[0]

        if self.test_score_threshold > 0:
            keep = scores > self.test_score_threshold
            bbox_pred = bbox_pred[keep, :]
            scores = scores[keep]
            class_idxs = class_idxs[keep]

        result = Instances(image_size)
        result.pred_boxes = Boxes(bbox_pred)
        result.scores = scores
        result.pred_classes = class_idxs

        return result
