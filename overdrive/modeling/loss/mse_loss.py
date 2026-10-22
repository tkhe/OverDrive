import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MSELoss"]


class MSELoss(nn.Module):
    def __init__(self, reduction: str = "mean", loss_weight: float = 1.0):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        return self.loss_weight * F.mse_loss(inputs, targets, reduction=self.reduction)
