from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from overdrive.layers import Conv2d
from overdrive.layers import ShapeSpec
from overdrive.layers import batched_nms
from overdrive.layers import cat
from overdrive.layers import get_activation
from overdrive.layers import weight_init
from overdrive.modeling.box_coder import BoxCoder
from overdrive.structures import Boxes
from overdrive.structures import Instances

__all__ = ["ConvFCBoxHead"]


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: str | dict = "",
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias, device, dtype)
        self.activation = get_activation(activation)

    def forward(self, x):
        x = self.linear(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class ConvFCBoxHead(nn.Module):
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        conv_dims: list[int],
        fc_dims: list[int],
        norm: str | dict = "",
        activation: str | dict = "ReLU",
        num_classes: int,
        box_coder: BoxCoder,
        loss_cls: nn.Module,
        loss_reg: nn.Module,
        reg_decoded_box: bool,
        cls_agnostic_box_reg: bool = False,
        test_score_threshold: float = 0.05,
        test_nms_threshold: float = 0.5,
        max_detections_per_image: int = 100,
    ):
        super().__init__()

        output_size = (input_shape.channels, input_shape.height, input_shape.width)

        convs = OrderedDict()
        for k, conv_dim in enumerate(conv_dims, start=1):
            conv = Conv2d(
                output_size[0],
                conv_dim,
                kernel_size=3,
                stride=1,
                norm=norm,
                activation=activation,
            )
            weight_init.c2_msra_fill(conv)
            convs[f"conv{k}"] = conv
            output_size = (conv_dim, output_size[1], output_size[2])
        self.convs = nn.Sequential(convs) if len(convs) > 0 else None

        fcs = OrderedDict()
        for k, fc_dim in enumerate(fc_dims, start=1):
            fc = Linear(int(np.prod(output_size)), fc_dim, activation=activation)
            weight_init.c2_xavier_fill(fc.linear)
            fcs[f"fc{k}"] = fc
            output_size = fc_dim
        self.fcs = nn.Sequential(fcs) if len(fcs) > 0 else None

        if isinstance(output_size, tuple):
            output_size = output_size[0] * output_size[1] * output_size[2]

        self.cls_score = nn.Linear(output_size, num_classes + 1)
        num_box_reg_classes = 1 if cls_agnostic_box_reg else num_classes
        self.bbox_pred = nn.Linear(output_size, num_box_reg_classes * 4)

        self.box_coder = box_coder

        self.loss_cls = loss_cls
        self.loss_reg = loss_reg
        self.reg_decoded_box = reg_decoded_box

        self.num_classes = num_classes

        self.cls_agnostic_box_reg = cls_agnostic_box_reg
        self.test_score_threshold = test_score_threshold
        self.test_nms_threshold = test_nms_threshold
        self.max_detections_per_image = max_detections_per_image

        weight_init.normal_fill(self.cls_score, std=0.01)
        weight_init.normal_fill(self.bbox_pred, std=0.001)

    def forward(self, x):
        if self.convs is not None:
            x = self.convs(x)

        x = torch.flatten(x, 1)

        if self.fcs is not None:
            x = self.fcs(x)

        scores = self.cls_score(x)
        proprosal_deltas = self.bbox_pred(x)
        return scores, proprosal_deltas

    def loss(self, pred_scores, pred_deltas, proposals):
        gt_classes = cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            if proposal_boxes.requires_grad:
                raise ValueError("Proposals should not require gradients!")

            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat([(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals], dim=0)
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=pred_deltas.device)

        loss_cls = self.loss_cls(pred_scores, gt_classes)

        box_dim = proposal_boxes.shape[1]  # 4
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = ((gt_classes >= 0) & (gt_classes < self.num_classes)).nonzero(as_tuple=True)[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[fg_inds, gt_classes[fg_inds]]

        if self.reg_decoded_box:
            pred_boxes = self.box_coder.apply_deltas(fg_pred_deltas, proposal_boxes[fg_inds])
            loss_reg = self.loss_reg(pred_boxes, gt_boxes[fg_inds])
        else:
            gt_deltas = self.box_coder.get_deltas(proposal_boxes[fg_inds], gt_boxes[fg_inds])
            loss_reg = self.loss_reg(fg_pred_deltas, gt_deltas)

        return {
            "loss_cls": loss_cls,
            "loss_reg": loss_reg / max(gt_classes.numel(), 1.0),
        }

    def inference(self, pred_scores, pred_deltas, proposals):
        scores = self.predict_probs(pred_scores, proposals)
        boxes = self.predict_boxes(pred_deltas, proposals)
        image_shapes = [x.image_size for x in proposals]
        return rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_threshold,
            self.test_nms_threshold,
            self.max_detections_per_image,
        )

    def predict_boxes_for_gt_classes(self, proposal_deltas, proposals):
        if len(proposals) == 0:
            return []

        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        N, B = proposal_boxes.shape
        predict_boxes = self.box_coder.apply_deltas(proposal_deltas, proposal_boxes)  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(self, proposal_deltas: torch.Tensor, proposals: list[Instances]):
        if len(proposals) == 0:
            return []

        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box_coder.apply_deltas(proposal_deltas, proposal_boxes)  # Nx(Kx4)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, pred_scores: torch.Tensor, proposals: list[Instances]):
        num_inst_per_image = [len(p) for p in proposals]
        probs = pred_scores.softmax(dim=-1)
        return probs.split(num_inst_per_image, dim=0)


def rcnn_inference(
    boxes: list[torch.Tensor],
    scores: list[torch.Tensor],
    image_shapes: list[tuple[int, int]],
    score_threshold: float,
    nms_threshold: float,
    topk_per_image: int,
):
    result_per_image = [
        rcnn_inference_single_image(
            boxes_per_image,
            scores_per_image,
            image_shape,
            score_threshold,
            nms_threshold,
            topk_per_image,
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes, strict=True)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: tuple[int, int],
    score_threshold: float,
    nms_threshold: float,
    topk_per_image: int,
):
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_threshold  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_threshold)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]
