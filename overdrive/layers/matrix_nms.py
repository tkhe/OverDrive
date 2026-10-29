# Copyright (c) OpenMMLab. All rights reserved.
# Adapted for OverDrive from MMDetection (Apache-2.0); see LICENSES/mmdetection.txt.
import torch

__all__ = ["mask_matrix_nms"]


def mask_matrix_nms(
    masks: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    filter_thr: float = -1,
    nms_pre: int = -1,
    max_num: int = -1,
    kernel: str = "gaussian",
    sigma: float = 2.0,
    mask_area: torch.Tensor | None = None,
):
    """Class-aware Matrix NMS. Return scores, labels, masks and original indices.

    Masks have shape (N, H, W); labels and scores have shape (N,).
    Non-positive limits disable truncation. Empty masks are supported.
    """
    if kernel not in ("gaussian", "linear"):
        raise ValueError("kernel must be 'gaussian' or 'linear'")
    if not len(masks) == len(labels) == len(scores):
        raise ValueError("masks, labels and scores must have equal lengths")
    if mask_area is not None and len(mask_area) != len(masks):
        raise ValueError("mask_area must have one value per mask")
    keep = scores.argsort(descending=True)
    if nms_pre > 0:
        keep = keep[:nms_pre]
    masks, labels, scores = masks[keep], labels[keep], scores[keep].float()
    if len(keep) == 0:
        return scores, labels, masks, keep

    areas = masks.sum((1, 2)).float() if mask_area is None else mask_area[keep].float()
    flattened = masks.flatten(1).float()
    # Keep the intersection calculation in float32 even under AMP.
    with torch.autocast(device_type=masks.device.type, enabled=False):
        intersections = flattened @ flattened.t()
    unions = areas[:, None] + areas[None, :] - intersections
    ious = (intersections / unions.clamp_min(1e-6)).triu(diagonal=1)
    decay_ious = ious * (labels[:, None] == labels[None, :])
    compensation = decay_ious.max(dim=0).values[:, None]
    if kernel == "gaussian":
        decay = torch.exp(-sigma * (decay_ious.square() - compensation.square()))
    else:
        decay = (1 - decay_ious) / (1 - compensation).clamp_min(1e-6)
    scores = scores * decay.min(dim=0).values

    valid = torch.where(scores >= filter_thr)[0] if filter_thr > 0 else torch.arange(len(scores), device=scores.device)
    order = valid[scores[valid].argsort(descending=True)]
    if max_num > 0:
        order = order[:max_num]
    return scores[order], labels[order], masks[order], keep[order]
