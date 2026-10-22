import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from overdrive.layers import get_activation
from overdrive.layers import inverse_sigmoid
from overdrive.layers import weight_init
from overdrive.modeling.dense_head import DenseHead
from overdrive.modeling.matcher import Matcher
from overdrive.structures import Boxes
from overdrive.structures import BoxMode
from overdrive.structures import Instances
from overdrive.utils.comm import reduce_mean

__all__ = ["DeformableDETRHead"]


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
        self.activation1 = get_activation(activation)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation2 = get_activation(activation)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.fc3(x)
        return x


class DeformableDETRHead(DenseHead):
    """
    Head of Deformable DETR (https://arxiv.org/abs/2010.04159).

    Every decoder layer has its own prediction branches. When `share_pred_layer` is True all the
    prediction layers share parameters, which is the case when the reference points are not refined
    layer by layer. When `as_two_stage` is True an additional prediction layer is used to generate
    the query proposals from the encoder output, and the classification targets of those proposals
    are all set to the single foreground class.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        embed_dim: int = 256,
        activation: str | dict = "ReLU",
        num_decoder_layers: int = 6,
        share_pred_layer: bool = True,
        as_two_stage: bool = False,
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
        self.share_pred_layer = share_pred_layer
        self.as_two_stage = as_two_stage
        self.use_auxiliary_loss = use_auxiliary_loss
        self.test_score_threshold = test_score_threshold
        self.max_detections_per_image = max_detections_per_image
        self.prior_prob = prior_prob

        # The last prediction layer is used to generate the proposals from the encoder feature map
        # when `as_two_stage` is True.
        self.num_pred_layer = num_decoder_layers + 1 if as_two_stage else num_decoder_layers

        fc_cls = nn.Linear(embed_dim, num_classes)
        reg_branch = FFN(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=4,
            activation=activation,
        )

        if share_pred_layer:
            self.cls_branches = nn.ModuleList([fc_cls] * self.num_pred_layer)
            self.reg_branches = nn.ModuleList([reg_branch] * self.num_pred_layer)
        else:
            self.cls_branches = nn.ModuleList([copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)])

        self.matcher = matcher

        self.loss_cls = loss_cls
        self.loss_iou = loss_iou
        self.loss_reg = loss_reg

        self.init_weights()

    def init_weights(self):
        """
        Initialize the prediction branches. It has to be called after the transformer is built,
        because the xavier initialization of the transformer does not cover the head.
        """
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for cls_branch in self.cls_branches:
            weight_init.bias_fill(cls_branch, bias_value)

        for reg_branch in self.reg_branches:
            weight_init.constant_fill(reg_branch.fc3, val=0, bias=0)

        if self.as_two_stage:
            for reg_branch in self.reg_branches:
                nn.init.constant_(reg_branch.fc3.bias.data[2:], 0.0)
        else:
            # the width and height of the first prediction layer start from a small value
            nn.init.constant_(self.reg_branches[0].fc3.bias.data[2:], -2.0)

    def forward(self, hidden_states: torch.Tensor, references: list[torch.Tensor]):
        """
        Args:
            hidden_states (Tensor): Hidden states of all decoder layers with shape
                `(num_layers, bs, num_queries, embed_dim)`.
            references (list[Tensor]): The reference points from the decoder. The first element is
                the initial reference and the remaining ones are the intermediate references, whose
                last one is not used. Each reference has shape `(bs, num_queries, 4)` arranged as
                `(cx, cy, w, h)` when the detector is two-stage, otherwise `(bs, num_queries, 2)`
                arranged as `(cx, cy)`.

        Returns:
            tuple[Tensor, Tensor]: The classification scores with shape
            `(num_layers, bs, num_queries, num_classes)` and the sigmoid bounding boxes with shape
            `(num_layers, bs, num_queries, 4)` arranged as `(cx, cy, w, h)`.
        """
        all_layers_cls_scores = []
        all_layers_bbox_preds = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            hidden_state = hidden_states[layer_id]
            cls_scores = self.cls_branches[layer_id](hidden_state)
            bbox_preds = self.reg_branches[layer_id](hidden_state)

            if reference.shape[-1] == 4:
                bbox_preds = bbox_preds + reference
            else:
                if reference.shape[-1] != 2:
                    raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference.shape[-1]} instead.")
                bbox_preds[..., :2] += reference

            all_layers_cls_scores.append(cls_scores)
            all_layers_bbox_preds.append(bbox_preds.sigmoid())

        return torch.stack(all_layers_cls_scores), torch.stack(all_layers_bbox_preds)

    def loss(
        self,
        all_layers_cls_scores: torch.Tensor,
        all_layers_bbox_preds: torch.Tensor,
        gt_instances: list[Instances],
        enc_cls_scores: torch.Tensor | None = None,
        enc_bbox_preds: torch.Tensor | None = None,
    ):
        losses = {}

        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]

        gt_labels, bbox_targets = self.get_ground_truth(cls_scores, bbox_preds, gt_instances)
        loss_cls, loss_iou, loss_reg = self.calculate_loss(cls_scores, bbox_preds, gt_labels, bbox_targets)
        losses["loss_cls"] = loss_cls
        losses["loss_iou"] = loss_iou
        losses["loss_reg"] = loss_reg

        if self.use_auxiliary_loss:
            for idx, (cls_scores, bbox_preds) in enumerate(zip(all_layers_cls_scores[:-1], all_layers_bbox_preds[:-1], strict=True)):
                gt_labels, bbox_targets = self.get_ground_truth(cls_scores, bbox_preds, gt_instances)
                loss_cls, loss_iou, loss_reg = self.calculate_loss(cls_scores, bbox_preds, gt_labels, bbox_targets)
                losses[f"aux.{idx}.loss_cls"] = loss_cls
                losses[f"aux.{idx}.loss_iou"] = loss_iou
                losses[f"aux.{idx}.loss_reg"] = loss_reg

        # loss of the proposals generated from the encoder feature map. Their classification targets
        # are all set to the single foreground class, following the official implementation.
        if enc_cls_scores is not None:
            proposal_gt_instances = copy.deepcopy(gt_instances)
            for gt_per_image in proposal_gt_instances:
                gt_per_image.gt_classes = torch.zeros_like(gt_per_image.gt_classes)

            gt_labels, bbox_targets = self.get_ground_truth(enc_cls_scores, enc_bbox_preds, proposal_gt_instances)
            loss_cls, loss_iou, loss_reg = self.calculate_loss(enc_cls_scores, enc_bbox_preds, gt_labels, bbox_targets)
            losses["enc_loss_cls"] = loss_cls
            losses["enc_loss_iou"] = loss_iou
            losses["enc_loss_reg"] = loss_reg

        return losses

    def calculate_loss(self, cls_scores, bbox_preds, gt_labels, bbox_targets):
        gt_labels = torch.stack(gt_labels)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)

        num_pos = pos_mask.sum().item()

        num_pos = cls_scores.new_tensor([num_pos])
        num_pos = torch.clamp(reduce_mean(num_pos), min=1).item()

        gt_labels_target = F.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[..., :-1]
        loss_cls = self.loss_cls(cls_scores[valid_mask], gt_labels_target.to(cls_scores.dtype)) / num_pos

        bbox_targets = torch.stack(bbox_targets).float()

        # Decode in FP32 outside no_grad so GIoU supervises box regression.
        pred_boxes = BoxMode.convert(bbox_preds[pos_mask].to(bbox_targets.dtype), BoxMode.CXCYWH_REL, BoxMode.XYXY_REL)
        gt_boxes = BoxMode.convert(bbox_targets[pos_mask], BoxMode.CXCYWH_REL, BoxMode.XYXY_REL)
        loss_iou = self.loss_iou(pred_boxes, gt_boxes) / num_pos
        loss_reg = self.loss_reg(bbox_preds[pos_mask].to(bbox_targets.dtype), bbox_targets[pos_mask]) / num_pos

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
            h, w = gt_per_image.image_size
            scale = bbox_pred.new_tensor([w, h, w, h]).unsqueeze(0)
            bbox_pred = BoxMode.convert(bbox_pred, BoxMode.CXCYWH_REL, BoxMode.XYXY_REL)
            bbox_pred = bbox_pred * scale

            pred_instances = Instances((h, w))
            pred_instances.pred_scores = cls_score
            pred_instances.pred_boxes = Boxes(bbox_pred)
            matched_idxs, matched_labels = self.matcher.match(pred_instances, gt_per_image)

            if len(gt_per_image) > 0:
                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]
                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
                gt_labels_i[matched_labels == 0] = self.num_classes
            else:
                matched_gt_boxes_i = torch.zeros_like(bbox_pred)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes

            gt_labels.append(gt_labels_i)
            bbox_targets.append(BoxMode.convert(matched_gt_boxes_i / scale, BoxMode.XYXY_REL, BoxMode.CXCYWH_REL))

        return gt_labels, bbox_targets

    def inference(self, cls_scores, bbox_preds, image_sizes):
        results = []

        cls_scores = cls_scores[-1]
        bbox_preds = bbox_preds[-1]

        for image_idx, image_size in enumerate(image_sizes):
            cls_score = cls_scores[image_idx]
            bbox_pred = bbox_preds[image_idx]
            result = self.inference_single_image(cls_score, bbox_pred, image_size)
            results.append(result)
        return results

    def inference_single_image(self, cls_score, bbox_pred, image_size):
        cls_score = cls_score.sigmoid()
        scores, idxs = cls_score.view(-1).topk(self.max_detections_per_image)
        class_idxs = idxs % self.num_classes

        bbox_pred = bbox_pred[idxs // self.num_classes]

        bbox_pred = BoxMode.convert(bbox_pred, BoxMode.CXCYWH_ABS, BoxMode.XYXY_ABS)
        bbox_pred[:, 0::2] *= image_size[1]
        bbox_pred[:, 1::2] *= image_size[0]

        keep = scores > self.test_score_threshold
        bbox_pred = bbox_pred[keep, :]
        scores = scores[keep]
        class_idxs = class_idxs[keep]

        result = Instances(image_size)
        result.pred_boxes = Boxes(bbox_pred)
        result.scores = scores
        result.pred_classes = class_idxs

        return result
