# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from MMDetection for OverDrive; see LICENSES/mmdetection.txt.
import torch

__all__ = ["fast_nms"]


def fast_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    coefficients: torch.Tensor,
    score_threshold: float = 0.05,
    iou_threshold: float = 0.5,
    top_k: int = 200,
    max_detections: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """YOLACT class-wise Fast NMS, carrying mask coefficients with each box.

    Args:
        boxes: Class-agnostic boxes (N, 4), in XYXY coordinates.
        scores: Softmax scores (N, C + 1), with background in the last column.
        coefficients: Mask coefficients (N, K).
        top_k: Maximum candidates per class before suppression (must be > 0).
        max_detections: Maximum survivors across classes; <= 0 keeps all.

    Returns:
        Detections (M, 5), foreground labels (M,), coefficients (M, K).
        Unlike greedy NMS, even a suppressed box can suppress a lower-scoring
        box. All pairwise comparisons within a class are evaluated in parallel.
    """
    boxes, scores = boxes.float(), scores.float()
    if not len(boxes):
        return boxes.new_empty((0, 5)), torch.empty(0, dtype=torch.long, device=boxes.device), coefficients[:0]

    scores, indices = scores[:, :-1].t().sort(dim=1, descending=True)
    scores, indices = scores[:, :top_k], indices[:, :top_k]
    boxes = boxes[indices]
    coefficients = coefficients[indices]
    areas = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0).prod(dim=-1)
    left_top = torch.maximum(boxes[:, :, None, :2], boxes[:, None, :, :2])
    right_bottom = torch.minimum(boxes[:, :, None, 2:], boxes[:, None, :, 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
    ious = intersection / (areas[:, :, None] + areas[:, None, :] - intersection).clamp_min(1e-6)
    keep = (ious.triu(diagonal=1).max(dim=1).values <= iou_threshold) & (scores > score_threshold)
    labels = torch.arange(scores.shape[0], device=scores.device)[:, None].expand_as(scores)[keep]
    boxes, scores, coefficients = boxes[keep], scores[keep], coefficients[keep]
    order = scores.argsort(descending=True)
    if max_detections > 0:
        order = order[:max_detections]
    return torch.cat((boxes[order], scores[order, None]), dim=1), labels[order], coefficients[order]
