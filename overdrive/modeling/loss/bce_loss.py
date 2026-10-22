import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BCELoss",
    "BCEWithLogitsLoss",
]


class BCELoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        return self.loss_weight * F.binary_cross_entropy(inputs, targets, reduction=self.reduction)


class BCEWithLogitsLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        return self.loss_weight * F.binary_cross_entropy_with_logits(inputs, targets, reduction=self.reduction)
