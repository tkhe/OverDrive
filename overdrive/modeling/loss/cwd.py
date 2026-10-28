import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ChannelWiseDivergence"]


class ChannelWiseDivergence(nn.Module):
    """
    Spatial KL for NCHW feature maps.
    """

    def __init__(self, tau: float = 1.0, loss_weight: float = 1.0):
        super().__init__()

        if tau <= 0:
            raise ValueError(f"`tau` must be positive, but got {tau}")

        self.tau = tau
        self.loss_weight = loss_weight

    def forward(self, student, teacher):
        if student.shape != teacher.shape:
            raise ValueError(f"CWD requires matching feature shapes, but got student shape {tuple(student.shape)} and teacher shape {tuple(teacher.shape)}.")

        student = student.flatten(2) / self.tau
        teacher = teacher.flatten(2) / self.tau

        loss = F.kl_div(
            F.log_softmax(student, dim=-1),
            F.softmax(teacher, dim=-1),
            reduction="sum",
        )
        return loss * self.loss_weight * self.tau**2 / (student.shape[0] * student.shape[1])
