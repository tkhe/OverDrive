import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DiceLoss"]


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-3,
    reduction: str = "mean",
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Average spatial Dice losses over all classes for each image.

    ``inputs`` are probabilities and ``targets`` are encoded labels, both shaped
    (N, C, H, W). ``valid_mask`` is an optional boolean mask shaped (N, H, W).
    ``none`` returns (N,), ``sum`` sums image losses, and ``mean`` averages only
    images with valid pixels. Fully masked images and empty batches return zero.
    """
    if inputs.shape != targets.shape:
        raise ValueError("Expected inputs and targets to have the same shape")

    if valid_mask is None:
        valid_mask = torch.ones_like(inputs[:, 0], dtype=torch.bool)
    spatial_mask = valid_mask.unsqueeze(1)
    inputs = inputs.masked_fill(~spatial_mask, 0).flatten(2)
    targets = targets.to(dtype=inputs.dtype).masked_fill(~spatial_mask, 0).flatten(2)

    intersection = torch.sum(inputs * targets, dim=2)
    inputs_sum = torch.sum(inputs, dim=2)
    targets_sum = torch.sum(targets, dim=2)
    dice_score = (2 * intersection + eps) / (inputs_sum + targets_sum + eps)
    loss = (1 - dice_score).mean(dim=1)

    if reduction == "mean":
        valid_images = valid_mask.flatten(1).any(dim=1)
        loss = loss.sum() / valid_images.sum().clamp_min(1)
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class DiceLoss(nn.Module):
    """
    Softmax Dice loss for mutually exclusive multiclass semantic segmentation.

    Logits have shape (N, C, H, W), C >= 2. Targets are integer labels (N, H, W)
    or one-hot/soft targets (N, C, H, W). ``ignore_index`` masks integer label
    positions; encoded targets treat all spatial positions as valid.
    All classes, including class 0 and absent classes, participate in the average.
    Floating-point precision follows the caller's autocast context.
    """

    def __init__(
        self,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        ignore_index: int | None = 255,
        eps: float = 1e-3,
    ):
        super().__init__()
        if eps <= 0:
            raise ValueError("Expected eps > 0")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("Expected reduction in ['none', 'mean', 'sum']")

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1] < 2:
            raise ValueError("Multiclass Dice loss requires at least two class channels")

        valid_mask = None
        if inputs.shape != targets.shape:
            targets = targets.long()
            if self.ignore_index is not None:
                valid_mask = targets != self.ignore_index
            else:
                valid_mask = torch.ones_like(targets, dtype=torch.bool)
            targets = F.one_hot(targets.masked_fill(~valid_mask, 0), inputs.shape[1]).permute(0, 3, 1, 2)

        return self.loss_weight * dice_loss(
            F.softmax(inputs, dim=1),
            targets,
            eps=self.eps,
            reduction=self.reduction,
            valid_mask=valid_mask,
        )
