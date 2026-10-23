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

from dino.modeling.denoising import CDNMetadata

__all__ = ["DINOHead"]


class BoxMLP(nn.Module):
    """Three-layer box regression branch."""

    def __init__(self, embed_dim, activation):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc3 = nn.Linear(embed_dim, 4)
        self.activation = get_activation(activation)

    def forward(self, x):
        return self.fc3(self.activation(self.fc2(self.activation(self.fc1(x)))))


class DINOHead(DenseHead):
    """Independent matching, multiclass encoder and contrastive denoising head."""

    def __init__(
        self,
        *,
        num_classes: int,
        embed_dim: int = 256,
        num_decoder_layers: int = 6,
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
        self.num_decoder_layers = num_decoder_layers
        self.use_auxiliary_loss = use_auxiliary_loss
        self.test_score_threshold = test_score_threshold
        self.max_detections_per_image = max_detections_per_image
        self.prior_prob = prior_prob
        # Each decoder layer and the encoder proposal stage have independent branches.
        self.cls_branches = nn.ModuleList([nn.Linear(embed_dim, num_classes) for _ in range(num_decoder_layers + 1)])
        self.reg_branches = nn.ModuleList([BoxMLP(embed_dim, activation) for _ in range(num_decoder_layers + 1)])
        self.matcher = matcher
        self.loss_cls = loss_cls
        self.loss_iou = loss_iou
        self.loss_reg = loss_reg
        self.init_weights()

    def init_weights(self):
        for cls_branch, reg_branch in zip(self.cls_branches, self.reg_branches, strict=True):
            weight_init.bias_fill(cls_branch, -math.log((1 - self.prior_prob) / self.prior_prob))
            weight_init.constant_fill(reg_branch.fc3, val=0, bias=0)

    def forward(self, hidden_states, references):
        # Match DINO's coordinate clamp, including references with gradients from
        # the previous refinement branch. Do not detach them here.
        cls_scores, bbox_preds = [], []
        for layer_id, state in enumerate(hidden_states):
            cls_scores.append(self.cls_branches[layer_id](state))
            bbox_preds.append((self.reg_branches[layer_id](state) + inverse_sigmoid(references[layer_id], eps=1e-3)).sigmoid())
        return torch.stack(cls_scores), torch.stack(bbox_preds)

    def loss(self, all_layers_cls_scores, all_layers_bbox_preds, gt_instances, enc_cls_scores=None, enc_bbox_preds=None, dn_metadata: CDNMetadata | None = None):
        pad_size = dn_metadata.pad_size if dn_metadata is not None else 0
        losses = {}
        indices = range(len(all_layers_cls_scores)) if self.use_auxiliary_loss else [len(all_layers_cls_scores) - 1]
        for layer_id in indices:
            scores = all_layers_cls_scores[layer_id, :, pad_size:]
            boxes = all_layers_bbox_preds[layer_id, :, pad_size:]
            targets = self.get_ground_truth(scores, boxes, gt_instances)
            prefix = "" if layer_id == len(all_layers_cls_scores) - 1 else f"aux.{layer_id}."
            for name, loss in zip(("cls", "iou", "reg"), self.calculate_loss(scores, boxes, *targets), strict=True):
                losses[f"{prefix}loss_{name}"] = loss
        # Encoder supervision uses actual object classes and only top-k proposals.
        if enc_cls_scores is not None:
            targets = self.get_ground_truth(enc_cls_scores, enc_bbox_preds, gt_instances)
            for name, loss in zip(("cls", "iou", "reg"), self.calculate_loss(enc_cls_scores, enc_bbox_preds, *targets), strict=True):
                losses[f"enc_loss_{name}"] = loss
        if dn_metadata is not None:
            indices = range(len(all_layers_cls_scores)) if self.use_auxiliary_loss else [len(all_layers_cls_scores) - 1]
            for layer_id in indices:
                scores = all_layers_cls_scores[layer_id, :, :pad_size]
                boxes = all_layers_bbox_preds[layer_id, :, :pad_size]
                labels, targets = self.get_dn_targets(scores, gt_instances, dn_metadata)
                dn_losses = self.calculate_loss(scores, boxes, labels, targets)
                prefix = "" if layer_id == len(all_layers_cls_scores) - 1 else f"aux.{layer_id}."
                for name, loss in zip(("cls", "iou", "reg"), dn_losses, strict=True):
                    losses[f"{prefix}loss_{name}_dn"] = loss
        return losses

    @torch.no_grad()
    def get_dn_targets(self, scores, gt_instances, metadata):
        labels, targets = [], []
        for gt, positive, target_idx in zip(gt_instances, metadata.positive_indices, metadata.target_indices, strict=True):
            # Negative queries and unassigned padding both contribute background focal loss.
            image_labels = torch.full((metadata.pad_size,), self.num_classes, dtype=torch.long, device=scores.device)
            image_targets = torch.zeros((metadata.pad_size, 4), device=scores.device, dtype=torch.float32)
            image_labels[positive] = gt.gt_classes[target_idx]
            h, w = gt.image_size
            normalized_boxes = gt.gt_boxes.tensor.float() / image_targets.new_tensor([w, h, w, h])
            image_targets[positive] = BoxMode.convert(normalized_boxes[target_idx], BoxMode.XYXY_REL, BoxMode.CXCYWH_REL)
            labels.append(image_labels)
            targets.append(image_targets)
        return labels, targets

    def calculate_loss(self, cls_scores, bbox_preds, gt_labels, bbox_targets):
        labels = torch.stack(gt_labels)
        positive = (labels >= 0) & (labels < self.num_classes)
        valid = labels >= 0
        normalizer = reduce_mean(positive.sum().to(dtype=cls_scores.dtype).reshape(1)).clamp(min=1).item()
        class_targets = F.one_hot(labels[valid], num_classes=self.num_classes + 1)[..., : self.num_classes].to(cls_scores.dtype)
        loss_cls = self.loss_cls(cls_scores[valid], class_targets) / normalizer
        # Decode in FP32 while preserving gradients for GIoU (including empty selections).
        target_boxes = torch.stack(bbox_targets).float()[positive]
        predicted_boxes = bbox_preds[positive].float()
        predicted_xyxy = BoxMode.convert(predicted_boxes, BoxMode.CXCYWH_REL, BoxMode.XYXY_REL)
        target_xyxy = BoxMode.convert(target_boxes, BoxMode.CXCYWH_REL, BoxMode.XYXY_REL)
        loss_iou = self.loss_iou(predicted_xyxy, target_xyxy) / normalizer
        loss_reg = self.loss_reg(predicted_boxes, target_boxes) / normalizer
        return loss_cls, loss_iou, loss_reg

    @torch.no_grad()
    def get_ground_truth(self, cls_scores, bbox_preds, gt_instances):
        labels, targets = [], []
        for scores, boxes, gt in zip(cls_scores, bbox_preds, gt_instances, strict=True):
            height, width = gt.image_size
            scale = boxes.new_tensor([width, height, width, height])
            predictions = Instances(gt.image_size)
            predictions.pred_scores = scores
            predictions.pred_boxes = Boxes(BoxMode.convert(boxes, BoxMode.CXCYWH_REL, BoxMode.XYXY_REL) * scale)
            matched_indices, matched_labels = self.matcher.match(predictions, gt)
            image_labels = torch.full((len(boxes),), self.num_classes, dtype=torch.long, device=boxes.device)
            image_targets = torch.zeros_like(boxes, dtype=torch.float32)
            positive = matched_labels > 0
            if len(gt):
                image_labels[positive] = gt.gt_classes[matched_indices[positive]]
                target_xyxy = gt.gt_boxes.tensor[matched_indices[positive]].float() / scale
                image_targets[positive] = BoxMode.convert(target_xyxy, BoxMode.XYXY_REL, BoxMode.CXCYWH_REL)
            labels.append(image_labels)
            targets.append(image_targets)
        return labels, targets

    def inference(self, cls_scores, bbox_preds, image_sizes):
        return [self.inference_single_image(scores, boxes, size) for scores, boxes, size in zip(cls_scores[-1], bbox_preds[-1], image_sizes, strict=True)]

    def inference_single_image(self, cls_score, bbox_pred, image_size):
        scores, indices = cls_score.sigmoid().flatten().topk(min(self.max_detections_per_image, cls_score.numel()))
        boxes = BoxMode.convert(bbox_pred[indices // self.num_classes], BoxMode.CXCYWH_REL, BoxMode.XYXY_REL)
        boxes = boxes * boxes.new_tensor([image_size[1], image_size[0], image_size[1], image_size[0]])
        keep = scores > self.test_score_threshold
        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes[keep])
        result.scores = scores[keep]
        result.pred_classes = (indices % self.num_classes)[keep]
        return result
