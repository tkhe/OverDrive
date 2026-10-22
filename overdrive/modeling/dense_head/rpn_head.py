import torch
import torch.nn as nn

from overdrive.layers import Conv2d
from overdrive.layers import ShapeSpec
from overdrive.layers import batched_nms
from overdrive.layers import cat
from overdrive.layers import weight_init
from overdrive.modeling.anchor_generator import AnchorGenerator
from overdrive.modeling.box_coder import BoxCoder
from overdrive.modeling.matcher import Matcher
from overdrive.modeling.sampler import Sampler
from overdrive.structures import Boxes
from overdrive.structures import ImageList
from overdrive.structures import Instances
from overdrive.structures import pairwise_iou
from overdrive.utils.events import get_event_storage
from overdrive.utils.memory import retry_if_cuda_oom

from .base import DenseHead

__all__ = ["RPNHead"]


class RPNHead(DenseHead):
    """
    support RPN head proposed in Faster R-CNN (https://arxiv.org/abs/1506.01497)
    """

    def __init__(
        self,
        input_shape: dict[str, ShapeSpec],
        in_features: list[str],
        *,
        conv_dims: list[int],
        norm: str | dict | nn.Module | None = "",
        activation: str | dict | nn.Module | None = {"name": "ReLU", "inplace": False},
        anchor_generator: AnchorGenerator,
        box_coder: BoxCoder,
        matcher: Matcher,
        sampler: Sampler,
        loss_cls: nn.Module,
        loss_reg: nn.Module,
        reg_decoded_box: bool = False,
        pre_nms_topk: tuple[int, int],
        post_nms_topk: tuple[int, int],
        nms_threshold: float = 0.7,
        min_box_size: float = 0,
    ):
        """
        Args:
            input_shape: Mapping from feature names to their shapes. Selected features
                must have the same number of channels.
            in_features: Names of input features, ordered to match the anchor generator.
            conv_dims: Output channels of each hidden convolution. A value of -1 keeps
                that convolution's input channels. An empty list skips hidden convolutions.
            norm: Normalization for hidden convolutions. Accepts a registered name,
                config dict, module, or None. Defaults to "", disabling normalization.
            activation: Activation for hidden convolutions. Defaults to ReLU(inplace=False).
                Accepts a registered name, config dict, module, or None. Set to None
                or an empty string to disable activation.
            anchor_generator: Generates anchors for each feature level. All levels must
                have the same number of anchors per spatial position.
            box_coder: Encodes ground-truth boxes as anchor deltas and decodes predicted
                deltas into proposal boxes.
            matcher: Matches anchors to ground-truth boxes using pairwise IoU and assigns
                foreground, background, or ignore labels.
            sampler: Subsamples matched anchor labels for training. Its num_samples
                determines the per-image loss normalization factor.
            loss_cls: Objectness loss applied to sampled foreground and background anchors.
            loss_reg: Box regression loss applied to sampled foreground anchors.
            reg_decoded_box: Whether to compute regression loss on decoded boxes instead
                of anchor deltas. Defaults to False.
            pre_nms_topk: Maximum proposals selected per feature level and image before
                NMS, as a (training, evaluation) tuple.
            post_nms_topk: Maximum proposals retained per image across all feature levels
                after NMS, as a (training, evaluation) tuple.
            nms_threshold: IoU threshold for NMS, applied independently within each feature
                level. Defaults to 0.7.
            min_box_size: Minimum proposal side length in pixels after clipping to the
                image. Boxes with either side at or below this value are removed. Defaults to 0.
        """
        super().__init__()

        in_channels = [input_shape[f].channels for f in in_features]
        num_anchors = anchor_generator.num_anchors

        if len(set(in_channels)) != 1:
            raise ValueError(f"Expected all feature levels to have the same number of input channels, but got {in_channels}")
        if len(set(num_anchors)) != 1:
            raise ValueError(f"Expected all feature levels to have the same number of anchors per spatial position, but got {num_anchors}")

        in_channels = in_channels[0]
        num_anchors = num_anchors[0]

        self.in_features = in_features

        cur_channels = in_channels
        if len(conv_dims) == 1:
            out_channels = cur_channels if conv_dims[0] == -1 else conv_dims[0]
            self.rpn_conv = Conv2d(cur_channels, out_channels, kernel_size=3, stride=1, norm=norm, activation=activation)
            cur_channels = out_channels
        else:
            self.rpn_conv = nn.Sequential()
            for k, conv_dim in enumerate(conv_dims, start=1):
                out_channels = cur_channels if conv_dim == -1 else conv_dim
                if out_channels <= 0:
                    raise ValueError(f"Conv output channels should be greater than 0. Got {out_channels}")
                conv = Conv2d(cur_channels, out_channels, kernel_size=3, stride=1, norm=norm, activation=activation)
                self.rpn_conv.add_module(f"conv{k}", conv)
                cur_channels = out_channels

        self.objectness_logits = nn.Conv2d(cur_channels, num_anchors, kernel_size=1)
        self.anchor_deltas = nn.Conv2d(cur_channels, num_anchors * 4, kernel_size=1)

        self.anchor_generator = anchor_generator
        self.box_coder = box_coder
        self.matcher = matcher
        self.sampler = sampler

        self.loss_cls = loss_cls
        self.loss_reg = loss_reg
        self.reg_decoded_box = reg_decoded_box

        self.pre_nms_topk = {True: pre_nms_topk[0], False: pre_nms_topk[1]}
        self.post_nms_topk = {True: post_nms_topk[0], False: post_nms_topk[1]}
        self.nms_threshold = nms_threshold
        self.min_box_size = min_box_size

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                weight_init.normal_fill(layer, std=0.01)

    def forward(self, features: dict[str, torch.Tensor]):
        features = [features[f] for f in self.in_features]

        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = self.rpn_conv(x)
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas

    def loss_and_inference(
        self,
        images: ImageList,
        pred_objectness_logits: list[torch.Tensor],
        pred_anchor_deltas: list[torch.Tensor],
        gt_instances: list[Instances] | None = None,
    ):
        anchors = self.anchor_generator(pred_objectness_logits)

        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (H, Hi, Wi, A) -> (N, Hi*Wi*A)
            x.permute(0, 2, 3, 1).flatten(1)
            for x in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, 4, x.shape[-2], x.shape[-1]).permute(0, 3, 4, 1, 2).flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training:
            if gt_instances is None:
                raise ValueError("RPN requires gt_instances in training!")

            gt_labels, gt_boxes = self.get_ground_truth(anchors, gt_instances)
            losses = self.loss(anchors, pred_objectness_logits, pred_anchor_deltas, gt_boxes, gt_labels)
        else:
            losses = {}

        proposals = self.inference(anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes)

        return proposals, losses

    @torch.no_grad()
    def get_ground_truth(self, anchors, gt_instances):
        anchors = Boxes.cat(anchors)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        for gt_boxes_i in gt_boxes:
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.matcher.match)(match_quality_matrix)
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            gt_labels_i = self.sampler.sample(gt_labels_i)

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor
            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels, matched_gt_boxes

    def loss(
        self,
        anchors: list[Boxes],
        pred_objectness_logits: list[torch.Tensor],
        pred_anchor_deltas: list[torch.Tensor],
        gt_boxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
    ):
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()

        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        valid_mask = gt_labels >= 0
        loss_cls = self.loss_cls(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
        )

        anchors = Boxes.cat(anchors).tensor
        if self.reg_decoded_box:
            pred_boxes = [self.box_coder.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)]
            loss_reg = self.loss_reg(
                torch.stack(pred_boxes)[pos_mask],
                torch.stack(gt_boxes)[pos_mask],
            )
        else:
            gt_anchor_detlas = [self.box_coder.get_deltas(anchors, k) for k in gt_boxes]
            gt_anchor_detlas = torch.stack(gt_anchor_detlas)
            loss_reg = self.loss_reg(
                cat(pred_anchor_deltas, dim=1)[pos_mask],
                gt_anchor_detlas[pos_mask],
            )

        normalizer = self.sampler.num_samples * num_images

        return {
            "loss_rpn_cls": loss_cls / normalizer,
            "loss_rpn_reg": loss_reg / normalizer,
        }

    def inference(
        self,
        anchors: list[Boxes],
        pred_objectness_logits: list[torch.Tensor],
        pred_anchor_deltas: list[torch.Tensor],
        image_sizes: list[tuple[int, int]],
    ):
        """
        Decode all the predicted box regression deltas to proposals. Find the top proposals
        by applying NMS and removing boxes that are too small.

        Returns:
            proposals (list[Instances]): list of N Instances. The i-th Instances
                stores post_nms_topk object proposals for image i, sorted by their
                objectness score in descending order.
        """
        # The proposals are treated as fixed for joint training with roi head.
        # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
        # are also network responses.
        with torch.no_grad():
            N = pred_anchor_deltas[0].shape[0]
            pred_proposals = []
            # For each feature map
            for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas, strict=True):
                B = anchors_i.tensor.size(1)
                pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
                # Expand anchors to shape (N*Hi*Wi*A, B)
                anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
                proposals_i = self.box_coder.apply_deltas(pred_anchor_deltas_i, anchors_i)
                # Append feature map proposals with shape (N, Hi*Wi*A, B)
                pred_proposals.append(proposals_i.view(N, -1, B))

            return find_top_rpn_proposals(
                pred_proposals,
                pred_objectness_logits,
                image_sizes,
                self.nms_threshold,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_size,
                self.training,
            )


def find_top_rpn_proposals(
    proposals: list[torch.Tensor],
    pred_objectness_logits: list[torch.Tensor],
    image_sizes: list[tuple[int, int]],
    nms_thresh: float,
    pre_nms_topk: int,
    post_nms_topk: int,
    min_box_size: float,
    training: bool,
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps for each image.

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        image_sizes (list[tuple]): sizes (h, w) for each image
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_size (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.

    Returns:
        list[Instances]: list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i, sorted by their
            objectness score in descending order.
    """
    num_images = len(image_sizes)
    device = proposals[0].device

    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = torch.arange(num_images, device=device)
    for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits, strict=True)):
        Hi_Wi_A = logits_i.shape[1]
        if isinstance(Hi_Wi_A, torch.Tensor):
            num_proposals_i = torch.clamp(Hi_Wi_A, max=pre_nms_topk)
        else:
            num_proposals_i = min(Hi_Wi_A, pre_nms_topk)

        topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)

        # each is N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))

    # 2. Concat all levels together
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)

    # 3. For each image, run a per-level NMS, and choose topk results.
    results: list[Instances] = []
    for n, image_size in enumerate(image_sizes):
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        lvl = level_ids

        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            if training:
                raise FloatingPointError("Predicted boxes or scores contain Inf/NaN. Training has diverged.")
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
            lvl = lvl[valid_mask]
        boxes.clip(image_size)

        # filter empty boxes
        keep = boxes.nonempty(threshold=min_box_size)
        if keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]

        keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
        keep = keep[:post_nms_topk]  # keep is already sorted

        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        results.append(res)
    return results
