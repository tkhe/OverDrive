import torch
import torch.nn as nn

from overdrive.modeling.backbone import Backbone
from overdrive.modeling.dense_head import DenseHead
from overdrive.modeling.neck import Neck
from overdrive.modeling.postprocessing import detector_postprocess
from overdrive.modeling.postprocessing import sem_seg_postprocess
from overdrive.modeling.roi_head import ROIHead
from overdrive.modeling.sem_seg_head import SemSegHead
from overdrive.structures import ImageList

__all__ = ["PanopticFPN"]


class PanopticFPN(nn.Module):
    """
    support PanopticFPN (https://arxiv.org/abs/1901.02446).
    """

    def __init__(
        self,
        *,
        backbone: Backbone,
        neck: Neck | None = None,
        rpn_head: DenseHead,
        roi_head: ROIHead,
        sem_seg_head: SemSegHead,
        pixel_mean: list[float],
        pixel_std: list[float],
        combine_overlap_thresh: float = 0.5,
        combine_stuff_area_thresh: int = 4096,
        combine_instances_score_thresh: float = 0.5,
    ):
        super().__init__()

        if not isinstance(backbone, Backbone):
            raise TypeError(f"Expected backbone to be a Backbone instance, but got {type(backbone).__name__}.")

        if neck is not None and not isinstance(neck, Neck):
            raise TypeError(f"Expected neck to be a Neck instance or None, but got {type(neck).__name__}.")

        if not isinstance(rpn_head, DenseHead):
            raise TypeError(f"Expected rpn_head to be a DenseHead instance, but got {type(rpn_head).__name__}.")

        if not isinstance(roi_head, ROIHead):
            raise TypeError(f"Expected roi_head to be a ROIHead instance, but got {type(roi_head).__name__}.")

        if not isinstance(sem_seg_head, SemSegHead):
            raise TypeError(f"Expected sem_seg_head to be a SemSegHead instance, but got {type(sem_seg_head).__name__}.")

        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head
        self.roi_head = roi_head
        self.sem_seg_head = sem_seg_head

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

        self.combine_overlap_thresh = combine_overlap_thresh
        self.combine_stuff_area_thresh = combine_stuff_area_thresh
        self.combine_instances_score_thresh = combine_instances_score_thresh

    @property
    def with_neck(self) -> bool:
        return self.neck is not None

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    def forward(self, batched_inputs: list[dict[str, torch.Tensor]]):
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        if self.with_neck:
            features = self.neck(features)
        sem_seg_predictions = self.sem_seg_head(features)
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)

        if self.training:
            if any("sem_seg" not in x or "instances" not in x for x in batched_inputs):
                raise ValueError("PanopticFPN training requires sem_seg and instances for every image")

            gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
            padding_constraints = {}
            padding_constraints.update(self.backbone.padding_constraints)
            if self.with_neck:
                padding_constraints.update(self.neck.padding_constraints)
            gt_sem_seg = ImageList.from_tensors(
                gt_sem_seg,
                padding_constraints,
                pad_value=self.sem_seg_head.ignore_index,
            ).tensor
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        proposals, rpn_losses = self.rpn_head.loss_and_inference(
            images,
            pred_objectness_logits,
            pred_anchor_deltas,
            gt_instances,
        )
        results, roi_losses = self.roi_head(images, features, proposals, gt_instances)

        if self.training:
            losses = {}
            losses.update(rpn_losses)
            losses.update(roi_losses)
            losses.update(self.sem_seg_head.loss(sem_seg_predictions, gt_sem_seg))
            return losses
        else:
            sem_seg_results = self.sem_seg_head.inference(sem_seg_predictions, images.tensor.shape[-2:])

            processed_results = []
            for results_per_image, sem_seg_per_image, input_per_image, image_size in zip(results, sem_seg_results, batched_inputs, images.image_sizes, strict=True):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                transforms = input_per_image.get("transforms", None)
                sem_seg_per_image = sem_seg_postprocess(
                    sem_seg_per_image,
                    image_size,
                    height,
                    width,
                    transforms=transforms,
                    align_corners=self.sem_seg_head.align_corners,
                )
                results_per_image = detector_postprocess(results_per_image, height, width, transforms=transforms)
                panoptic_seg = combine_semantic_and_instance_outputs(
                    results_per_image,
                    sem_seg_per_image.argmax(dim=0),
                    self.combine_overlap_thresh,
                    self.combine_stuff_area_thresh,
                    self.combine_instances_score_thresh,
                )
                processed_results.append({"instances": results_per_image, "sem_seg": sem_seg_per_image, "panoptic_seg": panoptic_seg})
            return processed_results

    def preprocess_image(self, batched_inputs: list[dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        padding_constraints = {}
        padding_constraints.update(self.backbone.padding_constraints)
        if self.with_neck:
            padding_constraints.update(self.neck.padding_constraints)
        images = ImageList.from_tensors(images, padding_constraints)
        return images


def combine_semantic_and_instance_outputs(
    instance_results,
    semantic_results,
    overlap_threshold,
    stuff_area_thresh,
    instances_score_thresh,
):
    """
    Implement a simple combining logic following
    "combine_semantic_and_instance_predictions.py" in panopticapi
    to produce panoptic segmentation outputs.

    Args:
        instance_results: output of :func:`detector_postprocess`.
        semantic_results: an (H, W) tensor, each element is the contiguous semantic
            category id

    Returns:
        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
            Each dict contains keys "id", "category_id", "isthing".
    """
    panoptic_seg = torch.zeros_like(semantic_results, dtype=torch.int32)

    # sort instance outputs by scores
    sorted_inds = torch.argsort(-instance_results.scores)

    current_segment_id = 0
    segments_info = []

    instance_masks = instance_results.pred_masks.to(dtype=torch.bool, device=panoptic_seg.device)

    # Add instances one-by-one, check for overlaps with existing ones
    for inst_id in sorted_inds:
        score = instance_results.scores[inst_id].item()
        if score < instances_score_thresh:
            break
        mask = instance_masks[inst_id]  # H,W
        mask_area = mask.sum().item()

        if mask_area == 0:
            continue

        intersect = (mask > 0) & (panoptic_seg > 0)
        intersect_area = intersect.sum().item()

        if intersect_area * 1.0 / mask_area > overlap_threshold:
            continue

        if intersect_area > 0:
            mask = mask & (panoptic_seg == 0)

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": True,
                "score": score,
                "category_id": instance_results.pred_classes[inst_id].item(),
                "instance_id": inst_id.item(),
            }
        )

    # Add semantic results to remaining empty areas
    semantic_labels = torch.unique(semantic_results).cpu().tolist()
    for semantic_label in semantic_labels:
        if semantic_label == 0:  # 0 is a special "thing" class
            continue
        mask = (semantic_results == semantic_label) & (panoptic_seg == 0)
        mask_area = mask.sum().item()
        if mask_area < stuff_area_thresh:
            continue

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": False,
                "category_id": semantic_label,
                "area": mask_area,
            }
        )

    return panoptic_seg, segments_info
