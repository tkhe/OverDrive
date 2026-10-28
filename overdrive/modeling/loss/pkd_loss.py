import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PKDLoss"]


class PKDLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()

        self.loss_weight = loss_weight

    def norm(self, feat):
        flat = feat.permute(1, 0, 2, 3).reshape(feat.shape[1], -1)
        std = flat.std(dim=1, keepdim=True, correction=int(flat.shape[1] > 1))
        normalized = (flat - flat.mean(dim=1, keepdim=True)) / (std + 1e-6)
        return normalized.reshape(feat.shape[1], feat.shape[0], *feat.shape[2:]).permute(1, 0, 2, 3)

    def forward(self, student: torch.Tensor, teacher: torch.Tensor):
        if student.shape[2:] != teacher.shape[2:]:
            student = F.interpolate(student, size=teacher.shape[2:], mode="bilinear", align_corners=False)

        if student.shape != teacher.shape:
            raise ValueError(f"PKD requires matching batch and channel dimensions, but got student shape {tuple(student.shape)} and teacher shape {tuple(teacher.shape)}")

        loss = F.mse_loss(self.norm(student), self.norm(teacher)) / 2
        return loss * self.loss_weight
