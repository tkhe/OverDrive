import torch

from .base import BoxCoder

__all__ = ["DistancePointBoxCoder"]


class DistancePointBoxCoder(BoxCoder):
    def get_deltas(self, points: torch.Tensor, target_boxes: torch.Tensor, strides: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points (Tensor): of shape (N, 2)
            target_boxes (Tensor): of shape (N, 4)
            strides (Tensor): of shape (N, 2)
        """
        left = (points[:, 0] - target_boxes[:, 0]) / strides[:, 0]
        top = (points[:, 1] - target_boxes[:, 1]) / strides[:, 1]
        right = (target_boxes[:, 2] - points[:, 0]) / strides[:, 0]
        bottom = (target_boxes[:, 3] - points[:, 1]) / strides[:, 1]
        return torch.stack([left, top, right, bottom], dim=-1)

    def apply_deltas(self, deltas: torch.Tensor, points: torch.Tensor, strides: torch.Tensor) -> torch.Tensor:
        x1 = points[:, 0] - deltas[:, 0] * strides[:, 0]
        y1 = points[:, 1] - deltas[:, 1] * strides[:, 1]
        x2 = points[:, 0] + deltas[:, 2] * strides[:, 0]
        y2 = points[:, 1] + deltas[:, 3] * strides[:, 1]
        return torch.stack([x1, y1, x2, y2], dim=-1)
