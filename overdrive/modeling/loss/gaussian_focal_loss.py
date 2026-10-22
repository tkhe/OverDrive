import torch
import torch.nn as nn

__all__ = ["GaussianFocalLoss"]


def gaussian_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 2,
    gamma: float = 4,
    pos_weight: float = 1,
    neg_weight: float = 1,
    eps: float = 1e-12,
    reduction: str = "mean",
):
    pos_weights = targets.eq(1)
    neg_weights = (1 - targets).pow(gamma)
    pos_loss = -(inputs + eps).log() * (1 - inputs).pow(alpha) * pos_weights
    neg_loss = -(1 - inputs + eps).log() * inputs.pow(alpha) * neg_weights
    loss = pos_weight * pos_loss + neg_weight * neg_loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class GaussianFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 2,
        gamma: float = 4,
        pos_weight: float = 1,
        neg_weight: float = 1,
        eps: float = 1e-12,
        reduction: str = "mean",
        loss_weight: float = 1,
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        return self.loss_weight * gaussian_focal_loss(
            inputs,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            pos_weight=self.pos_weight,
            neg_weight=self.neg_weight,
            eps=self.eps,
            reduction=self.reduction,
        )
