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
from overdrive.utils.comm import reduce_mean

from .base import DenseHead

__all__ = ["FCOSHead"]


INF = 1e8
EPS = 1e-6


class FCOSHead(DenseHead):
    """
    Anchor-free dense head proposed in FCOS.

    The classification and regression towers are shared by all feature levels,
    while each level has its own learnable regression scale.  Regression
    values are represented as ``(left, top, right, bottom)`` distances from a
    point to the four sides of its assigned box.
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
        loss_cls: nn.Module,
        loss_reg: nn.Module,
        loss_centerness: nn.Module,
        regress_ranges: tuple[tuple[float, float], ...] = (
            (-1, 64),
            (64, 128),
            (128, 256),
            (256, 512),
            (512, INF),
        ),
        center_sampling: bool = False,
        center_sample_radius: float = 1.5,
        norm_on_bbox: bool = False,
        centerness_on_reg: bool = False,
        test_score_threshold: float = 0.05,
        test_topk_candidates: int = 1000,
        test_nms_threshold: float = 0.6,
        max_detections_per_image: int = 100,
        prior_prob: float = 0.01,
    ):
        """
        Args:
            input_shape (Dict[str, ShapeSpec]): input shape of FPN features.
            in_features (List[str]): input feature names.
            num_classes (int): number of foreground classes.
            conv_dims (List[int]): dimensions of the tower convolutions.
            norm (str or Dict[str, str]): normalization for tower layers.
            activation (str or Dict[str, str]): activation for tower layers.
            box_coder (DistancePointBoxCoder): point-to-box coder.
            loss_cls (nn.Module): classification loss, normally focal loss.
            loss_reg (nn.Module): regression loss, normally GIoU loss with
                ``reduction="none"``.
            loss_centerness (nn.Module): centerness binary classification loss.
            regress_ranges (Tuple[Tuple[float, float]]): valid object-size
                range for each feature level.
            center_sampling (bool): whether to use center sampling.
            center_sample_radius (float): center sampling radius in strides.
            norm_on_bbox (bool): whether regression distances are stride
                normalized.
            centerness_on_reg (bool): whether the centerness branch uses the
                regression tower feature.
            test_score_threshold (float): classification score threshold.
            test_topk_candidates (int): candidates kept per feature level.
            test_nms_threshold (float): class-aware NMS threshold.
            max_detections_per_image (int): maximum final detections.
            prior_prob (float): prior foreground probability for cls bias.
        """
        super().__init__()

        if len(set(input_shape[f].channels for f in in_features)) != 1:
            raise ValueError("Using different channels of input between levels is not currently supported!")

        if not in_features:
            raise ValueError("in_features must contain at least one feature level")

        if not conv_dims:
            raise ValueError("conv_dims must contain at least one convolution dimension")

        if len(regress_ranges) != len(in_features):
            raise ValueError("regress_ranges must have one range for each input feature")

        if not 0.0 < prior_prob < 1.0:
            raise ValueError(f"prior_prob must be in (0, 1), got {prior_prob}")

        in_channels = input_shape[in_features[0]].channels
        strides = [input_shape[f].stride for f in in_features]
        if any(stride is None for stride in strides):
            raise ValueError("Every FCOS input feature must have a stride")

        self.in_features = in_features
        self.strides = strides
        self.num_classes = num_classes
        self.regress_ranges = tuple(tuple(item) for item in regress_ranges)
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.test_score_threshold = test_score_threshold
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_threshold = test_nms_threshold
        self.max_detections_per_image = max_detections_per_image

        cls_subnet = []
        bbox_subnet = []
        for in_dim, out_dim in zip([in_channels] + list(conv_dims[:-1]), conv_dims, strict=True):
            cls_subnet.append(Conv2d(in_dim, out_dim, kernel_size=3, stride=1, norm=norm, activation=activation))
            bbox_subnet.append(Conv2d(in_dim, out_dim, kernel_size=3, stride=1, norm=norm, activation=activation))

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = Conv2d(conv_dims[-1], num_classes, kernel_size=3, stride=1)
        self.bbox_pred = Conv2d(conv_dims[-1], 4, kernel_size=3, stride=1)
        self.centerness = Conv2d(conv_dims[-1], 1, kernel_size=3, stride=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in strides])
        self.point_generator = PointGenerator(strides, offset=0.5)
        self.box_coder = box_coder
        self.loss_cls = loss_cls
        self.loss_reg = loss_reg
        self.loss_centerness = loss_centerness

        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred, self.centerness]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    weight_init.normal_fill(layer, mean=0, std=0.01)

        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        weight_init.bias_fill(self.cls_score, bias_value)

    def forward(self, features: dict[str, torch.Tensor]):
        """
        Args:
            features (Dict[str, Tensor]): FPN feature maps.

        Returns:
            tuple: Classification scores, bbox distances and centerness
                logits.  At level ``l`` their shapes are respectively
                ``(N, K, H_l, W_l)``, ``(N, 4, H_l, W_l)`` and
                ``(N, 1, H_l, W_l)``.
        """
        features = [features[f] for f in self.in_features]

        cls_scores = []
        bbox_preds = []
        centernesses = []
        for feature, scale in zip(features, self.scales, strict=True):
            cls_feat = self.cls_subnet(feature)
            reg_feat = self.bbox_subnet(feature)

            cls_scores.append(self.cls_score(cls_feat))
            if self.centerness_on_reg:
                centernesses.append(self.centerness(reg_feat))
            else:
                centernesses.append(self.centerness(cls_feat))

            # Cast before clamp/exp to keep the regression branch safe under
            # AMP.  The normalized form is decoded with the actual stride;
            # the unnormalized form is decoded with unit stride.
            bbox_pred = scale(self.bbox_pred(reg_feat)).float()
            if self.norm_on_bbox:
                bbox_preds.append(bbox_pred.clamp(min=0))
            else:
                bbox_preds.append(bbox_pred.exp())

        return cls_scores, bbox_preds, centernesses

    def loss(self, cls_scores, bbox_preds, centernesses, gt_instances):
        """
        Compute FCOS classification, regression and centerness losses.
        """
        num_images = len(gt_instances)
        if num_images == 0:
            raise ValueError("FCOS loss requires at least one image")

        points = self.point_generator(cls_scores)
        point_strides = []
        for level_points, stride in zip(points, self.strides, strict=True):
            point_strides.append(level_points.new_full((level_points.size(0), 2), stride))
        strides = cat(point_strides, dim=0)

        gt_classes, gt_bbox_targets = self.get_ground_truth(points, gt_instances)
        points = cat(points, dim=0)
        gt_classes = torch.stack(gt_classes, dim=0)
        gt_bbox_targets = torch.stack(gt_bbox_targets, dim=0)

        cls_scores = cat([permute_to_N_HWA_K(score, self.num_classes) for score in cls_scores], dim=1)
        bbox_preds = cat([permute_to_N_HWA_K(pred, 4) for pred in bbox_preds], dim=1)
        centernesses = cat([permute_to_N_HWA_K(pred, 1) for pred in centernesses], dim=1)

        # Background is encoded as num_classes.  One-hot encoding with an
        # extra background column makes all background targets zero after the
        # final column is removed, as required by sigmoid focal loss.
        cls_targets = F.one_hot(gt_classes.to(torch.int64), num_classes=self.num_classes + 1)[..., : self.num_classes]
        cls_targets = cls_targets.to(dtype=cls_scores.dtype)
        loss_cls = self.loss_cls(
            cls_scores.reshape(-1, self.num_classes),
            cls_targets.reshape(-1, self.num_classes),
        )
        if loss_cls.numel() > 1:
            loss_cls = loss_cls.sum()

        pos_mask = (gt_classes >= 0) & (gt_classes < self.num_classes)
        num_pos = reduce_mean(pos_mask.sum().to(dtype=torch.float32)).clamp(min=1.0)
        loss_cls = loss_cls / num_pos

        pos_mask = pos_mask.reshape(-1)
        pos_bbox_preds = bbox_preds.reshape(-1, 4)[pos_mask]
        pos_centernesses = centernesses.reshape(-1)[pos_mask]
        pos_bbox_targets = gt_bbox_targets.reshape(-1, 4)[pos_mask]
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = torch.sqrt((left_right.min(dim=-1).values / left_right.max(dim=-1).values * top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values).clamp(min=0))
        # All ranks must execute the same reductions in the same order,
        # including ranks with no local positives (whose centerness sum is zero).
        centerness_denorm = reduce_mean(centerness_targets.sum().detach()).clamp(min=EPS)

        if pos_bbox_targets.size(0) > 0:
            points = points.unsqueeze(0).expand(num_images, -1, -1).reshape(-1, 2)
            strides = strides.unsqueeze(0).expand(num_images, -1, -1).reshape(-1, 2)
            pos_points = points[pos_mask]
            pos_strides = strides[pos_mask]

            if self.norm_on_bbox:
                decode_strides = pos_strides
            else:
                decode_strides = pos_strides.new_ones(pos_strides.shape)

            pos_pred_boxes = self.box_coder.apply_deltas(pos_bbox_preds, pos_points, decode_strides)
            pos_target_boxes = self.box_coder.apply_deltas(pos_bbox_targets, pos_points, decode_strides)

            loss_reg = self.loss_reg(pos_pred_boxes, pos_target_boxes)
            loss_reg = (loss_reg * centerness_targets).sum() / centerness_denorm

            loss_centerness = self.loss_centerness(pos_centernesses, centerness_targets).sum() / num_pos
        else:
            # Empty sums keep the graph connected without reading non-positive
            # predictions, which could contain non-finite values.
            loss_reg = pos_bbox_preds.sum()
            loss_centerness = pos_centernesses.sum()

        return {
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
            "loss_centerness": loss_centerness,
        }

    @torch.no_grad()
    def get_ground_truth(self, points: list[torch.Tensor], gt_instances: list[Instances]):
        """
        Assign every point in every image to at most one ground-truth box.

        This is intentionally the only FCOS target-assignment method.  All
        images, feature levels, center-sampling checks and regress-range checks
        are handled directly by the loops below.

        Returns:
            gt_classes: list of ``(P,)`` tensors, one per image.  Background is
                represented by ``num_classes``.
            gt_bbox_targets: list of ``(P, 4)`` tensors, one per image.  The
                order is ``(left, top, right, bottom)``.
        """
        if len(points) != len(self.regress_ranges):
            raise ValueError("The number of point levels and regress_ranges must match")

        num_points_per_level = []
        regress_ranges = []
        point_strides = []
        for level_points, regress_range, stride in zip(points, self.regress_ranges, self.strides, strict=True):
            num_points_per_level.append(level_points.size(0))
            regress_ranges.append(level_points.new_tensor(regress_range).view(1, 2).expand(level_points.size(0), 2))
            point_strides.append(level_points.new_full((level_points.size(0),), stride))

        points = cat(points, dim=0)
        regress_ranges = cat(regress_ranges, dim=0)
        point_strides = cat(point_strides, dim=0)
        num_points = points.size(0)
        point_indices = torch.arange(num_points, device=points.device)

        gt_classes = []
        gt_bbox_targets = []
        for gt_per_image in gt_instances:
            gt_boxes = gt_per_image.gt_boxes.tensor.to(device=points.device)
            gt_classes_per_image = gt_per_image.gt_classes.to(device=points.device, dtype=torch.int64)
            num_gt = gt_boxes.size(0)

            if num_gt == 0:
                gt_classes.append(gt_classes_per_image.new_full((num_points,), self.num_classes))
                gt_bbox_targets.append(gt_boxes.new_zeros((num_points, 4)))
                continue

            xs = points[:, 0, None]
            ys = points[:, 1, None]
            left = xs - gt_boxes[None, :, 0]
            top = ys - gt_boxes[None, :, 1]
            right = gt_boxes[None, :, 2] - xs
            bottom = gt_boxes[None, :, 3] - ys
            bbox_targets = torch.stack((left, top, right, bottom), dim=-1)

            inside_gt_bbox = bbox_targets.min(dim=-1).values > 0
            if self.center_sampling:
                gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
                center_radius = self.center_sample_radius * point_strides[:, None]
                center_x_min = torch.maximum(gt_boxes[None, :, 0], gt_centers[None, :, 0] - center_radius)
                center_y_min = torch.maximum(gt_boxes[None, :, 1], gt_centers[None, :, 1] - center_radius)
                center_x_max = torch.minimum(gt_boxes[None, :, 2], gt_centers[None, :, 0] + center_radius)
                center_y_max = torch.minimum(gt_boxes[None, :, 3], gt_centers[None, :, 1] + center_radius)
                center_bbox_targets = torch.stack(
                    (
                        xs - center_x_min,
                        ys - center_y_min,
                        center_x_max - xs,
                        center_y_max - ys,
                    ),
                    dim=-1,
                )
                inside_gt_bbox = center_bbox_targets.min(dim=-1).values > 0

            max_regress_distance = bbox_targets.max(dim=-1).values
            inside_regress_range = (max_regress_distance >= regress_ranges[:, None, 0]) & (max_regress_distance <= regress_ranges[:, None, 1])

            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
            areas = gt_areas[None].expand(num_points, num_gt).clone()
            areas[~(inside_gt_bbox & inside_regress_range)] = INF
            min_area, min_area_indices = areas.min(dim=1)

            labels = gt_classes_per_image[min_area_indices].clone()
            labels[min_area == INF] = self.num_classes
            selected_bbox_targets = bbox_targets[point_indices, min_area_indices]
            if self.norm_on_bbox:
                selected_bbox_targets = selected_bbox_targets / point_strides[:, None]

            gt_classes.append(labels)
            gt_bbox_targets.append(selected_bbox_targets)

        return gt_classes, gt_bbox_targets

    def inference(self, cls_scores, bbox_preds, centernesses, image_sizes):
        points = self.point_generator(cls_scores)

        cls_scores = [permute_to_N_HWA_K(score, self.num_classes) for score in cls_scores]
        bbox_preds = [permute_to_N_HWA_K(pred, 4) for pred in bbox_preds]
        centernesses = [permute_to_N_HWA_K(pred, 1) for pred in centernesses]

        results = []
        for image_idx, image_size in enumerate(image_sizes):
            cls_scores_per_image = [score[image_idx] for score in cls_scores]
            bbox_preds_per_image = [pred[image_idx] for pred in bbox_preds]
            centernesses_per_image = [pred[image_idx] for pred in centernesses]
            results_per_image = self.inference_single_image(
                points,
                cls_scores_per_image,
                bbox_preds_per_image,
                centernesses_per_image,
                image_size,
            )
            results.append(results_per_image)
        return results

    def inference_single_image(
        self,
        points: list[torch.Tensor],
        cls_scores: list[torch.Tensor],
        bbox_preds: list[torch.Tensor],
        centernesses: list[torch.Tensor],
        image_size: tuple[int, int],
    ):
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        for cls_score_i, bbox_pred_i, centerness_i, points_i, stride in zip(
            cls_scores,
            bbox_preds,
            centernesses,
            points,
            self.strides,
            strict=True,
        ):
            predicted_prob = cls_score_i.sigmoid().flatten()
            keep_idxs = predicted_prob > self.test_score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = keep_idxs.nonzero(as_tuple=True)[0]

            num_topk = min(self.test_topk_candidates, topk_idxs.size(0))
            if num_topk:
                predicted_prob, sort_idxs = predicted_prob.sort(descending=True)
                predicted_prob = predicted_prob[:num_topk]
                topk_idxs = topk_idxs[sort_idxs[:num_topk]]

                point_idxs = topk_idxs // self.num_classes
                class_idxs = topk_idxs % self.num_classes
                predicted_prob = predicted_prob * centerness_i.view(-1).sigmoid()[point_idxs]
                bbox_pred_i = bbox_pred_i[point_idxs]
                points_i = points_i[point_idxs]
                if self.norm_on_bbox:
                    strides_i = points_i.new_full(points_i.shape, stride)
                else:
                    strides_i = points_i.new_ones(points_i.shape)
                predicted_boxes = self.box_coder.apply_deltas(bbox_pred_i, points_i, strides_i)
                predicted_boxes = torch.stack(
                    (
                        predicted_boxes[:, 0].clamp(min=0, max=image_size[1]),
                        predicted_boxes[:, 1].clamp(min=0, max=image_size[0]),
                        predicted_boxes[:, 2].clamp(min=0, max=image_size[1]),
                        predicted_boxes[:, 3].clamp(min=0, max=image_size[0]),
                    ),
                    dim=-1,
                )

                boxes_all.append(predicted_boxes)
                scores_all.append(predicted_prob)
                class_idxs_all.append(class_idxs)

        result = Instances(image_size)
        if not boxes_all:
            result.pred_boxes = Boxes(points[0].new_zeros((0, 4)))
            result.scores = points[0].new_zeros((0,))
            result.pred_classes = torch.zeros((0,), dtype=torch.int64, device=points[0].device)
            return result

        boxes_all = cat(boxes_all, dim=0)
        scores_all = cat(scores_all, dim=0)
        class_idxs_all = cat(class_idxs_all, dim=0)

        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.test_nms_threshold)
        keep = keep[: self.max_detections_per_image]

        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result
