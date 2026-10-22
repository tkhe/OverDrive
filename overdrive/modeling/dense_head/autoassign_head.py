import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from overdrive.layers import Conv2d
from overdrive.layers import Scale
from overdrive.layers import ShapeSpec
from overdrive.layers import batched_nms
from overdrive.layers import cat
from overdrive.layers import permute_to_N_HWA_K
from overdrive.layers import weight_init
from overdrive.modeling.anchor_generator import PointGenerator
from overdrive.modeling.box_coder import DistancePointBoxCoder
from overdrive.structures import Boxes
from overdrive.structures import Instances
from overdrive.structures import matched_pairwise_iou
from overdrive.utils.comm import reduce_mean

from .base import DenseHead

__all__ = ["AutoAssignHead"]

EPS = 1e-12


class CenterPrior(nn.Module):
    """
    Center weighting module to adjust the category-specific prior distributions.
    """

    def __init__(
        self,
        num_classes: int,
        strides: list[int],
        force_topk: bool = False,
        topk: int = 9,
    ):
        """
        Args:
            num_classes (int): number of classes.
            strides (List[int]): stride of each input feature map.
            force_topk (bool): when no point falls into gt_bbox, forcibly select
                the k points closest to the center to calculate the center prior.
            topk (int): the number of points used to calculate the center prior
                when no point falls in gt_bbox. Only works when force_topk is True.
        """
        super().__init__()

        self.mean = nn.Parameter(torch.zeros(num_classes, 2))
        self.sigma = nn.Parameter(torch.ones(num_classes, 2))
        self.strides = strides
        self.force_topk = force_topk
        self.topk = topk

    def forward(
        self,
        points: list[torch.Tensor],
        gt_boxes: torch.Tensor,
        gt_classes: torch.Tensor,
        inside_gt_bbox_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the center prior of each point on the feature map for each instance.

        Args:
            points (List[Tensor]): coordinates of points on each feature level, each
                with shape (num_points_i, 2).
            gt_boxes (Tensor): ground truth boxes of shape (num_gt, 4).
            gt_classes (Tensor): ground truth classes of shape (num_gt, ).
            inside_gt_bbox_mask (Tensor): bool tensor of shape (num_points, num_gt),
                marking whether each point falls within a certain gt.

        Returns:
            center_prior_weights (Tensor): float tensor of shape (num_points, num_gt),
                each value is the center weighting coefficient.
            inside_gt_bbox_mask (Tensor): bool tensor of shape (num_points, num_gt),
                marking whether each point falls within a certain gt or is one of the
                topk nearest points for a specific gt when force_topk is enabled.
        """
        inside_gt_bbox_mask = inside_gt_bbox_mask.clone()
        num_gts = len(gt_classes)
        num_points = sum(len(item) for item in points)
        if num_gts == 0:
            return gt_boxes.new_zeros(num_points, num_gts), inside_gt_bbox_mask

        center_prior_list = []
        for slvl_points, stride in zip(points, self.strides, strict=True):
            # slvl_points has shape (h * w, 2), expand to (h * w, num_gt, 2)
            single_level_points = slvl_points[:, None, :].expand((slvl_points.size(0), num_gts, 2))
            gt_center_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
            gt_center_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
            gt_center = torch.stack([gt_center_x, gt_center_y], dim=1)[None]
            # instance_center and instance_sigma have shape (1, num_gt, 2)
            instance_center = self.mean[gt_classes][None]
            instance_sigma = self.sigma[gt_classes][None]
            # distance has shape (num_points, num_gt, 2)
            distance = ((single_level_points - gt_center) / float(stride) - instance_center) ** 2
            center_prior = torch.exp(-distance / (2 * instance_sigma**2)).prod(dim=-1)
            center_prior_list.append(center_prior)
        center_prior_weights = cat(center_prior_list, dim=0)

        if self.force_topk:
            gt_inds_no_points_inside = torch.nonzero(inside_gt_bbox_mask.sum(0) == 0).reshape(-1)
            if gt_inds_no_points_inside.numel():
                topk_center_index = center_prior_weights[:, gt_inds_no_points_inside].topk(self.topk, dim=0)[1]
                temp_mask = inside_gt_bbox_mask[:, gt_inds_no_points_inside]
                inside_gt_bbox_mask[:, gt_inds_no_points_inside] = torch.scatter(
                    temp_mask,
                    dim=0,
                    index=topk_center_index,
                    src=torch.ones_like(topk_center_index, dtype=torch.bool),
                )

        center_prior_weights[~inside_gt_bbox_mask] = 0
        return center_prior_weights, inside_gt_bbox_mask


class AutoAssignHead(DenseHead):
    """
    support AutoAssignHead proposed in AutoAssign (https://arxiv.org/abs/2007.03496).
    """

    def __init__(
        self,
        input_shape: dict[str, ShapeSpec],
        in_features: list[str],
        *,
        num_classes: int,
        conv_dims: list[int],
        norm: str | dict = "GN",
        activation: str | dict = "ReLU",
        box_coder: DistancePointBoxCoder,
        loss_reg: nn.Module,
        force_topk: bool = False,
        topk: int = 9,
        pos_loss_weight: float = 0.25,
        neg_loss_weight: float = 0.75,
        center_loss_weight: float = 0.75,
        prior_prob: float = 0.02,
        test_score_threshold: float = 0.05,
        test_topk_candidates: int = 1000,
        test_nms_threshold: float = 0.6,
        max_detections_per_image: int = 100,
    ):
        """
        Args:
            input_shape (Dict[str, ShapeSpec]): input shape.
            in_features (List[str]): input feature names.
            num_classes (int): number of classes.
            conv_dims (List[int]): dimensions for each convolutional layer.
            norm (str or Dict[str, str]): normalization for conv layers.
            activation (str or Dict[str, str]): activation for conv layers.
            box_coder (DistancePointBoxCoder): defines the transformation from points to boxes.
            loss_reg (nn.Module): regression loss. Should produce per-element loss (reduction="none").
            force_topk (bool): used in center prior to handle extremely small gt.
            topk (int): number of points used to calculate the center prior when no point
                falls in gt_bbox. Only works when force_topk is True.
            pos_loss_weight (float): loss weight of the positive loss.
            neg_loss_weight (float): loss weight of the negative loss.
            center_loss_weight (float): loss weight of the center prior loss.
            prior_prob (float): prior weight for computing bias of the classification layer.
            test_score_threshold (float): inference cls score threshold.
            test_topk_candidates (int): select topk candidates before NMS.
            test_nms_threshold (float): overlap threshold used for NMS.
            max_detections_per_image (int): maximum number of detections per image during inference.
        """
        super().__init__()

        if len(set([input_shape[f].channels for f in in_features])) != 1:
            raise ValueError("Using different channels of input between levels is not currently supported!")

        in_channels = input_shape[in_features[0]].channels
        strides = [input_shape[f].stride for f in in_features]

        self.in_features = in_features
        self.strides = strides
        self.num_classes = num_classes
        self.pos_loss_weight = pos_loss_weight
        self.neg_loss_weight = neg_loss_weight
        self.center_loss_weight = center_loss_weight
        self.test_score_threshold = test_score_threshold
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_threshold = test_nms_threshold
        self.max_detections_per_image = max_detections_per_image

        cls_subnet = []
        bbox_subnet = []
        for in_dim, out_dim in zip([in_channels] + list(conv_dims[:-1]), conv_dims, strict=True):
            cls_subnet.append(Conv2d(in_dim, out_dim, kernel_size=3, stride=1, bias=True, norm=norm, activation=activation))
            bbox_subnet.append(Conv2d(in_dim, out_dim, kernel_size=3, stride=1, bias=True, norm=norm, activation=activation))

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = Conv2d(conv_dims[-1], num_classes, kernel_size=3, stride=1)
        self.bbox_pred = Conv2d(conv_dims[-1], 4, kernel_size=3, stride=1)
        self.objectness = Conv2d(conv_dims[-1], 1, kernel_size=3, stride=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in strides])

        self.point_generator = PointGenerator(strides, offset=0)
        self.center_prior = CenterPrior(num_classes, strides, force_topk=force_topk, topk=topk)
        self.box_coder = box_coder
        self.loss_reg = loss_reg

        # initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred, self.objectness]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    weight_init.normal_fill(layer, mean=0, std=0.01)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        weight_init.bias_fill(self.cls_score, bias_value)
        weight_init.bias_fill(self.bbox_pred, 4.0)

    def forward(self, features: dict[str, torch.Tensor]):
        """
        Args:
            features (Dict[str, torch.Tensor]): FPN feature map tensors.

        Returns:
            cls_scores (list[Tensor]): #lvl tensors, each has shape (N, K, Hi, Wi).
            bbox_preds (list[Tensor]): #lvl tensors, each has shape (N, 4, Hi, Wi),
                predicting (l, t, r, b) distances in stride-normalized units.
            objectnesses (list[Tensor]): #lvl tensors, each has shape (N, 1, Hi, Wi).
        """
        features = [features[f] for f in self.in_features]

        cls_scores = []
        bbox_preds = []
        objectnesses = []
        for feature, scale in zip(features, self.scales, strict=True):
            cls_feat = self.cls_subnet(feature)
            reg_feat = self.bbox_subnet(feature)

            cls_scores.append(self.cls_score(cls_feat))
            objectnesses.append(self.objectness(reg_feat))
            # float to avoid overflow when enabling AMP, clamp instead of relu to keep gradient
            bbox_pred = scale(self.bbox_pred(reg_feat)).float()
            bbox_preds.append(bbox_pred.clamp(min=0))
        return cls_scores, bbox_preds, objectnesses

    def loss(self, cls_scores, bbox_preds, objectnesses, gt_instances):
        num_images = len(gt_instances)
        all_num_gt = sum(len(x) for x in gt_instances)

        all_level_points = self.point_generator(cls_scores)  # #lvl tensors, each (num_points_i, 2)
        mlvl_points = cat(all_level_points, dim=0)  # (num_points, 2)
        concat_strides = cat([all_level_points[i].new_full((all_level_points[i].size(0), 2), self.strides[i]) for i in range(len(self.strides))], dim=0)  # (num_points, 2)
        num_points = mlvl_points.size(0)

        # flatten level predictions to (N, num_points, C)
        cls_scores = cat([permute_to_N_HWA_K(x, self.num_classes) for x in cls_scores], dim=1)
        bbox_preds = cat([permute_to_N_HWA_K(x, 4) for x in bbox_preds], dim=1)
        objectnesses = cat([permute_to_N_HWA_K(x, 1) for x in objectnesses], dim=1)

        pos_loss_list = []
        neg_loss_list = []
        center_loss_list = []
        center_prior_weight_sums = []

        for img_id in range(num_images):
            gt_boxes = gt_instances[img_id].gt_boxes.tensor
            gt_classes = gt_instances[img_id].gt_classes
            num_gt = gt_boxes.size(0)

            cls_score = cls_scores[img_id].sigmoid()
            objectness = objectnesses[img_id].sigmoid()
            bbox_pred = bbox_preds[img_id]

            inside_gt_bbox_mask = self.get_inside_gt_bbox_mask(gt_boxes, mlvl_points)
            center_prior_weights, inside_gt_bbox_mask = self.center_prior(all_level_points, gt_boxes, gt_classes, inside_gt_bbox_mask)

            # decode predicted boxes and compute per (point, gt) regression loss and ious
            expand_points = mlvl_points[:, None, :].expand(num_points, num_gt, 2).reshape(-1, 2)
            expand_strides = concat_strides[:, None, :].expand(num_points, num_gt, 2).reshape(-1, 2)
            expand_bbox_pred = bbox_pred[:, None, :].expand(num_points, num_gt, 4).reshape(-1, 4)
            decoded_bbox_preds = self.box_coder.apply_deltas(expand_bbox_pred, expand_points, expand_strides)
            # the (l, t, r, b) targets always decode back to the gt box itself
            decoded_target_preds = gt_boxes[None].expand(num_points, num_gt, 4).reshape(-1, 4)

            with torch.no_grad():
                ious = matched_pairwise_iou(Boxes(decoded_bbox_preds), Boxes(decoded_target_preds))
                ious = ious.reshape(num_points, num_gt)
                if num_gt:
                    ious = ious.max(dim=-1, keepdim=True).values.repeat(1, num_gt)
                else:
                    ious = ious.new_zeros(num_points, num_gt)
                ious[~inside_gt_bbox_mask] = 0

            reg_loss = self.loss_reg(decoded_bbox_preds, decoded_target_preds).reshape(num_points, num_gt)

            pos_loss_list.append(self.get_pos_loss_single(cls_score, objectness, reg_loss, gt_classes, center_prior_weights))
            neg_loss_list.append(self.get_neg_loss_single(cls_score, objectness, gt_classes, ious, inside_gt_bbox_mask))
            center_prior_weight_sums.append(center_prior_weights.data.sum())

            if inside_gt_bbox_mask.any():
                center_loss_list.append(num_gt / center_prior_weights.sum().clamp(min=EPS))
            # when width or height of gt_bbox is smaller than the stride of p3
            else:
                center_loss_list.append(center_prior_weights.sum() * 0)

        pos_avg_factor = reduce_mean(bbox_preds.new_tensor(all_num_gt)).clamp_(min=1)
        pos_loss = sum(pos_loss_list) / pos_avg_factor

        neg_avg_factor = reduce_mean(sum(center_prior_weight_sums)).clamp_(min=1)
        neg_loss = sum(neg_loss_list) / neg_avg_factor

        center_loss = torch.stack(center_loss_list).mean() * self.center_loss_weight

        # avoid dead lock in DDP
        if all_num_gt == 0:
            pos_loss = bbox_preds.sum() * 0
            dummy_center_prior_loss = self.center_prior.mean.sum() * 0 + self.center_prior.sigma.sum() * 0
            center_loss = objectnesses.sum() * 0 + dummy_center_prior_loss

        return {
            "loss_pos": pos_loss,
            "loss_neg": neg_loss,
            "loss_center": center_loss,
        }

    @torch.no_grad()
    def get_inside_gt_bbox_mask(self, gt_boxes: torch.Tensor, points: torch.Tensor):
        """
        Compute whether each point falls within each gt box.

        Args:
            gt_boxes (Tensor): ground truth boxes of shape (num_gt, 4).
            points (Tensor): points of all fpn levels of shape (num_points, 2).

        Returns:
            inside_gt_bbox_mask (Tensor): bool tensor of shape (num_points, num_gt).
        """
        num_points = points.size(0)
        num_gt = gt_boxes.size(0)
        if num_gt == 0:
            return points.new_zeros((num_points, num_gt), dtype=torch.bool)

        xs, ys = points[:, 0:1], points[:, 1:2]
        left = xs - gt_boxes[:, 0][None]
        right = gt_boxes[:, 2][None] - xs
        top = ys - gt_boxes[:, 1][None]
        bottom = gt_boxes[:, 3][None] - ys
        bbox_targets = torch.stack([left, top, right, bottom], dim=-1)
        return bbox_targets.min(dim=-1).values > 0

    def get_pos_loss_single(self, cls_score, objectness, reg_loss, gt_classes, center_prior_weights):
        """
        Calculate the positive loss of all points in gt_bboxes.

        Args:
            cls_score (Tensor): category scores of shape (num_points, num_class).
            objectness (Tensor): foreground probability of shape (num_points, 1).
            reg_loss (Tensor): regression loss of shape (num_points, num_gt).
            gt_classes (Tensor): ground truth classes of shape (num_gt, ).
            center_prior_weights (Tensor): center weighting coefficient of shape (num_points, num_gt).

        Returns:
            pos_loss (Tensor): the positive loss of all points in the gt_bboxes.
        """
        # p_loc: localization confidence
        p_loc = torch.exp(-reg_loss)
        # p_cls: classification confidence
        p_cls = (cls_score * objectness)[:, gt_classes]
        # p_pos: joint confidence indicator
        p_pos = p_cls * p_loc

        # 3 is a hyper-parameter to control the contributions of high and low
        # confidence locations towards positive losses.
        confidence_weight = torch.exp(p_pos * 3)
        p_pos_weight = (confidence_weight * center_prior_weights) / ((confidence_weight * center_prior_weights).sum(0, keepdim=True)).clamp(min=EPS)
        reweighted_p_pos = (p_pos * p_pos_weight).sum(0)
        pos_loss = F.binary_cross_entropy(reweighted_p_pos, torch.ones_like(reweighted_p_pos), reduction="none")
        return pos_loss.sum() * self.pos_loss_weight

    def get_neg_loss_single(self, cls_score, objectness, gt_classes, ious, inside_gt_bbox_mask):
        """
        Calculate the negative loss of all points in feature map.

        Args:
            cls_score (Tensor): category scores of shape (num_points, num_class).
            objectness (Tensor): foreground probability of shape (num_points, 1).
            gt_classes (Tensor): ground truth classes of shape (num_gt, ).
            ious (Tensor): iou of pred_bbox and gt_bboxes of shape (num_points, num_gt).
            inside_gt_bbox_mask (Tensor): bool tensor of shape (num_points, num_gt).

        Returns:
            neg_loss (Tensor): the negative loss of all points in the feature map.
        """
        num_gt = len(gt_classes)
        joint_conf = cls_score * objectness
        p_neg_weight = torch.ones_like(joint_conf)
        if num_gt > 0:
            # the order of dimension would affect the value of p_neg_weight, we
            # strictly follow the original implementation.
            inside_gt_bbox_mask = inside_gt_bbox_mask.permute(1, 0)
            ious = ious.permute(1, 0)

            foreground_idxs = torch.nonzero(inside_gt_bbox_mask, as_tuple=True)
            temp_weight = 1 / (1 - ious[foreground_idxs]).clamp_(EPS)

            def normalize(x):
                return (x - x.min() + EPS) / (x.max() - x.min() + EPS)

            for instance_idx in range(num_gt):
                idxs = foreground_idxs[0] == instance_idx
                if idxs.any():
                    temp_weight[idxs] = normalize(temp_weight[idxs])

            p_neg_weight[foreground_idxs[1], gt_classes[foreground_idxs[0]]] = 1 - temp_weight

        logits = joint_conf * p_neg_weight
        neg_loss = logits**2 * F.binary_cross_entropy(logits, torch.zeros_like(logits), reduction="none")
        return neg_loss.sum() * self.neg_loss_weight

    def inference(self, cls_scores, bbox_preds, objectnesses, image_sizes):
        all_level_points = self.point_generator(cls_scores)

        cls_scores = [permute_to_N_HWA_K(x, self.num_classes) for x in cls_scores]
        bbox_preds = [permute_to_N_HWA_K(x, 4) for x in bbox_preds]
        objectnesses = [permute_to_N_HWA_K(x, 1) for x in objectnesses]

        results = []
        for image_idx, image_size in enumerate(image_sizes):
            cls_scores_per_image = [x[image_idx] for x in cls_scores]
            bbox_preds_per_image = [x[image_idx] for x in bbox_preds]
            objectnesses_per_image = [x[image_idx] for x in objectnesses]
            results_per_image = self.inference_single_image(
                all_level_points,
                cls_scores_per_image,
                bbox_preds_per_image,
                objectnesses_per_image,
                image_size,
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, points, cls_scores, bbox_preds, objectnesses, image_size: tuple[int, int]):
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        for cls_score_i, bbox_pred_i, objectness_i, points_i, stride in zip(cls_scores, bbox_preds, objectnesses, points, self.strides, strict=True):
            # (HxW, K) joint confidence
            predicted_prob = (cls_score_i.sigmoid() * objectness_i.sigmoid()).flatten()

            # 1. keep boxes with score higher than threshold
            keep_idxs = predicted_prob > self.test_score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = keep_idxs.nonzero(as_tuple=True)[0]

            # 2. keep top k scoring boxes only
            num_topk = min(self.test_topk_candidates, topk_idxs.size(0))
            predicted_prob, idxs = predicted_prob.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[idxs[:num_topk]]

            point_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            bbox_pred_i = bbox_pred_i[point_idxs]
            points_i = points_i[point_idxs]
            strides_i = points_i.new_full(points_i.shape, stride)
            predicted_boxes = self.box_coder.apply_deltas(bbox_pred_i, points_i, strides_i)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [cat(x) for x in [boxes_all, scores_all, class_idxs_all]]

        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.test_nms_threshold)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result
