import numpy as np
import torch
import torch.nn as nn
from torch.autograd.function import Function

from overdrive.modeling.matcher import Matcher
from overdrive.modeling.pooler import ROIPooler
from overdrive.modeling.sampler import Sampler
from overdrive.structures import Boxes
from overdrive.structures import ImageList
from overdrive.structures import Instances
from overdrive.structures import pairwise_iou
from overdrive.utils.events import get_event_storage

from .base import ROIHead
from .box_head.conv_fc_box_head import rcnn_inference
from .standard_roi_head import add_ground_truth_to_proposals

__all__ = ["CascadeROIHead"]


class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class CascadeROIHead(ROIHead):
    def __init__(
        self,
        *,
        box_in_features: list[str],
        box_pooler: ROIPooler,
        box_heads: list[nn.Module],
        mask_in_features: list[str] | None = None,
        mask_pooler: ROIPooler | None = None,
        mask_head: nn.Module | None = None,
        keypoint_in_features: list[str] | None = None,
        keypoint_pooler: ROIPooler | None = None,
        keypoint_head: nn.Module | None = None,
        train_on_pred_boxes: bool = False,
        num_classes: int,
        matchers: list[Matcher],
        sampler: Sampler,
        proposal_append_gt: bool = True,
    ):
        super().__init__()

        self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_heads = nn.ModuleList(box_heads)

        self.mask_in_features = mask_in_features
        self.mask_pooler = mask_pooler
        self.mask_head = mask_head

        self.keypoint_in_features = keypoint_in_features
        self.keypoint_pooler = keypoint_pooler
        self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes
        self.num_classes = num_classes
        self.proposal_append_gt = proposal_append_gt

        self.matchers = matchers
        self.sampler = sampler

        num_stage = len(self.box_heads)
        if num_stage == 0:
            raise ValueError("CascadeROIHead requires at least one box head")
        if len(self.matchers) != num_stage:
            raise ValueError(f"Expected {num_stage} matchers to match the box heads, but got {len(self.matchers)}")

        self.num_cascade_stages = num_stage

    @property
    def with_box(self) -> bool:
        return len(self.box_heads) > 0

    @torch.no_grad()
    def label_and_sample_propsals(
        self,
        proposals: list[Instances],
        targets: list[Instances],
    ):
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets, strict=True):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes,
                proposals_per_image.proposal_boxes,
            )
            matched_idxs, matched_labels = self.matchers[0].match(match_quality_matrix)
            matched_labels = self.sampler.sample(matched_labels)

            gt_classes = targets_per_image.gt_classes
            has_gt = gt_classes.numel() > 0
            # Get the corresponding GT for each proposal
            valid_mask = matched_labels >= 0
            if has_gt:
                gt_classes = gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                gt_classes[matched_labels == 0] = self.num_classes
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

            gt_classes = gt_classes[valid_mask]
            proposals_per_image = proposals_per_image[valid_mask]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[valid_mask]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for trg_name, trg_value in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(
        self,
        images: ImageList,
        features: dict[str, torch.Tensor],
        proposals: list[Instances],
        gt_instances: list[Instances] | None = None,
    ) -> tuple[list[Instances], dict[str, torch.Tensor]]:
        del images

        if self.training:
            if gt_instances is None or len(gt_instances) == 0:
                raise ValueError("`gt_instances` is required during training")

            proposals = self.label_and_sample_propsals(proposals, gt_instances)

        if self.training:
            losses = self._forward_box(features, proposals, gt_instances)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features: dict[str, torch.Tensor], instances: list[Instances]) -> list[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            list[Instances]:
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        if self.training:
            raise RuntimeError("Expected not self.training")
        if not instances[0].has("pred_boxes") or not instances[0].has("pred_classes"):
            raise ValueError("Expected instances[0].has('pred_boxes') and instances[0].has('pred_classes')")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(self, features: dict[str, torch.Tensor], proposals: list[Instances], gt_instances: list[Instances] | None = None):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        head_outputs = []
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]
        for k in range(self.num_cascade_stages):
            if k > 0:
                # The output boxes of the previous stage are used to create the input
                # proposals of the next stage.
                proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)
                if self.training:
                    proposals = self._match_and_label_boxes(proposals, k, gt_instances)
            pred_scores, pred_deltas = self._run_stage(features, proposals, k)
            prev_pred_boxes = self.box_heads[k].predict_boxes(pred_deltas, proposals)
            head_outputs.append((self.box_heads[k], pred_scores, pred_deltas, proposals))

        if self.training:
            losses = {}
            storage = get_event_storage()
            for stage, (box_head, pred_scores, pred_deltas, proposals) in enumerate(head_outputs):
                with storage.name_scope(f"stage{stage}"):
                    stage_losses = box_head.loss(pred_scores, pred_deltas, proposals)
                losses.update({f"{k}_stage{stage}": v for k, v in stage_losses.items()})
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h[0].predict_probs(h[1], h[3]) for h in head_outputs]

            # Average the scores across heads
            scores = [sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages) for scores_per_image in zip(*scores_per_stage, strict=True)]
            # Use the boxes of the last head
            box_head, pred_scores, pred_deltas, proposals = head_outputs[-1]
            boxes = box_head.predict_boxes(pred_deltas, proposals)
            pred_instances, _ = rcnn_inference(
                boxes,
                scores,
                image_sizes,
                box_head.test_score_threshold,
                box_head.test_nms_threshold,
                box_head.max_detections_per_image,
            )
            return pred_instances

    def _forward_mask(self, features: dict[str, torch.Tensor], instances: list[Instances]):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.with_mask:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = self.select_foreground_proposals(instances)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}

        pred_mask_logits = self.mask_head(features)

        if self.training:
            return self.mask_head.loss(pred_mask_logits, instances)
        else:
            return self.mask_head.inference(pred_mask_logits, instances)

    def _forward_keypoint(self, features: dict[str, torch.Tensor], instances: list[Instances]):
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.with_keypoint:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals with >=1 visible keypoints.
            instances, _ = self.select_foreground_proposals(instances)
            instances = self.select_proposals_with_visible_keypoints(instances)

        if self.keypoint_pooler is not None:
            features = [features[f] for f in self.keypoint_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.keypoint_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.keypoint_in_features}

        pred_keypoint_logits = self.keypoint_head(features)

        if self.training:
            return self.keypoint_head.loss(pred_keypoint_logits, instances)
        else:
            return self.keypoint_head.inference(pred_keypoint_logits, instances)

    @torch.no_grad()
    def _match_and_label_boxes(self, proposals, stage, gt_instances):
        """
        Match proposals with groundtruth using the matcher at the given stage.
        Remove ignored proposals and label the rest as foreground or background.

        Args:
            proposals (list[Instances]): One Instances for each image, with
                the field "proposal_boxes".
            stage (int): the current stage
            gt_instances (list[Instances]): the ground truth instances

        Returns:
            list[Instances]: proposals excluding ignored matches, with fields
                "gt_classes" and "gt_boxes".
        """
        proposals_with_gt = []
        num_fg_samples, num_bg_samples = [], []
        for proposals_per_image, gt_instances_per_image in zip(proposals, gt_instances, strict=True):
            match_quality_matrix = pairwise_iou(gt_instances_per_image.gt_boxes, proposals_per_image.proposal_boxes)
            matched_idxs, proposal_labels = self.matchers[stage].match(match_quality_matrix)
            valid_mask = proposal_labels >= 0
            proposals_per_image = proposals_per_image[valid_mask]
            matched_idxs = matched_idxs[valid_mask]
            proposal_labels = proposal_labels[valid_mask]
            if len(gt_instances_per_image) > 0:
                gt_classes = gt_instances_per_image.gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                gt_classes[proposal_labels == 0] = self.num_classes
                gt_boxes = gt_instances_per_image.gt_boxes[matched_idxs]
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                gt_boxes = Boxes(gt_instances_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4)))
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_boxes = gt_boxes
            proposals_with_gt.append(proposals_per_image)

            num_fg_samples.append((proposal_labels == 1).sum().item())
            num_bg_samples.append((proposal_labels == 0).sum().item())

        # Log the number of fg/bg samples in each stage
        storage = get_event_storage()
        storage.put_scalar(
            f"stage{stage}/roi_head/num_fg_samples",
            sum(num_fg_samples) / len(num_fg_samples),
        )
        storage.put_scalar(
            f"stage{stage}/roi_head/num_bg_samples",
            sum(num_bg_samples) / len(num_bg_samples),
        )
        return proposals_with_gt

    def _run_stage(self, features, proposals, stage):
        """
        Args:
            features (list[Tensor]): #lvl input features to ROIHeads
            proposals (list[Instances]): #image Instances, with the field "proposal_boxes"
            stage (int): the current stage

        Returns:
            Same output as `FastRCNNOutputLayers.forward()`.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        # The original implementation averages the losses among heads,
        # but scale up the parameter gradients of the heads.
        # This is equivalent to adding the losses among heads,
        # but scale down the gradients on features.
        if self.training:
            box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        box_features = self.box_heads[stage](box_features)
        return box_features

    def _create_proposals_from_boxes(self, boxes, image_sizes):
        """
        Args:
            boxes (list[Tensor]): per-image predicted boxes, each of shape Ri x 4
            image_sizes (list[tuple]): list of image shapes in (h, w)

        Returns:
            list[Instances]: per-image proposals with the given boxes.
        """
        # Just like RPN, the proposals should not have gradients
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size in zip(boxes, image_sizes, strict=True):
            boxes_per_image.clip(image_size)
            if self.training:
                # do not filter empty boxes at inference time,
                # because the scores from each stage need to be aligned and added later
                boxes_per_image = boxes_per_image[boxes_per_image.nonempty()]
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            proposals.append(prop)
        return proposals

    def select_foreground_proposals(self, proposals: list[Instances]):
        if not isinstance(proposals, (list, tuple)):
            raise TypeError("Expected isinstance(proposals, (list, tuple))")
        if not isinstance(proposals[0], Instances):
            raise TypeError("Expected isinstance(proposals[0], Instances)")
        if not proposals[0].has("gt_classes"):
            raise ValueError("Expected proposals[0].has('gt_classes')")

        fg_proposals = []
        fg_selection_masks = []
        for proposals_per_image in proposals:
            gt_classes = proposals_per_image.gt_classes
            fg_selection_mask = (gt_classes != -1) & (gt_classes != self.num_classes)
            fg_idxs = fg_selection_mask.nonzero().squeeze(1)
            fg_proposals.append(proposals_per_image[fg_idxs])
            fg_selection_masks.append(fg_selection_mask)
        return fg_proposals, fg_selection_masks

    def select_proposals_with_visible_keypoints(self, proposals: list[Instances]):
        ret = []
        all_num_fg = []
        for proposals_per_image in proposals:
            # If empty/unannotated image (hard negatives), skip filtering for train
            if len(proposals_per_image) == 0:
                ret.append(proposals_per_image)
                continue

            gt_keypoints = proposals_per_image.gt_keypoints.tensor
            # #fg x K x 3
            vis_mask = gt_keypoints[:, :, 2] >= 1
            xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]
            proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(dim=1)  # #fg x 1 x 4
            kp_in_box = (xs >= proposal_boxes[:, :, 0]) & (xs <= proposal_boxes[:, :, 2]) & (ys >= proposal_boxes[:, :, 1]) & (ys <= proposal_boxes[:, :, 3])
            selection = (kp_in_box & vis_mask).any(dim=1)
            selection_idxs = selection.nonzero(as_tuple=True)[0]
            all_num_fg.append(selection_idxs.numel())
            ret.append(proposals_per_image[selection_idxs])

        storage = get_event_storage()
        storage.put_scalar("keypoint_head/num_fg_samples", np.mean(all_num_fg))

        return ret
