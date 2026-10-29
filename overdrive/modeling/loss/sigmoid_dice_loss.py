import torch
import torch.nn as nn

__all__ = ["SigmoidDiceLoss"]


class SigmoidDiceLoss(nn.Module):
    """Per-instance sigmoid Dice loss with the squared (V-Net) denominator.

    SOLO uses ``2 * sum(p * y) / (sum(p**2) + sum(y**2) + 2 * eps)``.
    Inputs and targets have identical shape (N, ...). Unlike semantic DiceLoss,
    instances are independent and there is no softmax across channels.
    """

    def __init__(self, reduction: str = "mean", loss_weight: float = 1.0, eps: float = 1e-3):
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be 'none', 'mean' or 'sum'")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.shape != targets.shape or inputs.ndim < 2:
            raise ValueError("inputs and targets must have identical shape (N, ...)")
        probs = inputs.float().sigmoid().flatten(1)
        targets = targets.float().flatten(1)
        numerator = 2 * (probs * targets).sum(1)
        denominator = probs.square().sum(1) + targets.square().sum(1) + 2 * self.eps
        loss = (1 - numerator / denominator) * self.loss_weight
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.sum() / max(len(loss), 1)
        return loss
