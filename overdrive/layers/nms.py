import torch
from torchvision.ops import boxes as box_ops

__all__ = ["batched_nms"]


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    idxs: torch.Tensor,
    iou_threshold: float,
    class_agnostic: bool = False,
) -> torch.Tensor:
    """
    Apply per-category or class-agnostic NMS with float32 inputs.

    Args:
        boxes (Tensor[N, 4]): boxes where NMS will be performed.
        scores (Tensor[N]): scores for each box.
        idxs (Tensor[N]): category indices, ignored when class_agnostic is True.
        iou_threshold (float): IoU threshold used to suppress boxes.
        class_agnostic (bool): if True, suppress boxes across all categories.
            If False, keep the default per-category NMS behavior.
    """
    if boxes.shape[-1] != 4:
        raise ValueError(f"Expected boxes to have shape (N, 4), got {tuple(boxes.shape)}")

    # FP16 lacks range for category offsets; NMS requires matching input dtypes.
    boxes = boxes.float()
    scores = scores.float()
    if class_agnostic:
        return box_ops.nms(boxes, scores, iou_threshold)
    return box_ops.batched_nms(boxes, scores, idxs, iou_threshold)
