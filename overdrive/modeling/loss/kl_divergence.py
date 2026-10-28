import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["KLDivergence"]


class KLDivergence(nn.Module):
    """
    KL(teacher || student) along the class axis (dim=1), scaled by tau².
    """

    def __init__(
        self,
        tau: float = 1.0,
        reduction: str = "batchmean",
        loss_weight: float = 1.0,
    ):
        super().__init__()

        if tau <= 0:
            raise ValueError(f"`tau` must be positive, but got {tau}")

        self.tau = tau
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, student: torch.Tensor, teacher: torch.Tensor):
        student = student / self.tau
        teacher = teacher / self.tau
        loss = F.kl_div(
            F.log_softmax(student, dim=1),
            F.softmax(teacher, dim=1),
            reduction=self.reduction,
        )
        return self.loss_weight * self.tau**2 * loss
