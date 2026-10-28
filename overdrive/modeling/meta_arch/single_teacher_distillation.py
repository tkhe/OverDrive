import torch
import torch.nn as nn

__all__ = ["SingleTeacherDistillation"]


class SingleTeacherDistillation(nn.Module):
    """
    Distill a student model from a single frozen teacher.

    The teacher's parameters are frozen, its forward pass runs under
    ``torch.no_grad()``, and it remains in evaluation mode even when this
    module is trained.

    Training returns the student's task losses together with distillation
    losses computed by ``distiller``. Evaluation uses only the student model.
    """

    def __init__(self, student, teacher, distiller):
        super().__init__()

        self.student = student
        self.teacher = teacher
        self.distiller = distiller
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.distiller.prepare_from_student(self.student)
        self.distiller.prepare_from_teacher(self.teacher)

    def train(self, mode: bool = True):
        super().train(mode)

        self.teacher.eval()
        return self

    def forward(self, batched_inputs):
        if not self.training:
            return self.student(batched_inputs)

        with self.distiller:
            with self.distiller.teacher_recorders, torch.no_grad():
                self.teacher(batched_inputs)
            with self.distiller.student_recorders:
                student_losses = self.student(batched_inputs)
            losses = {f"student.{name}": value for name, value in student_losses.items()}
            losses.update({f"distill.{name}": value for name, value in self.distiller.compute_distill_losses().items()})
        return losses
