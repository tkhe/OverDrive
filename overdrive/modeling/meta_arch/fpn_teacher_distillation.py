import torch

from .single_teacher_distillation import SingleTeacherDistillation

__all__ = ["FPNTeacherDistillation"]


class FPNTeacherDistillation(SingleTeacherDistillation):
    """
    Run only teacher preprocessing, backbone and optional neck in training.

    The teacher must provide preprocess_image() returning an ImageList and a
    backbone. Teacher recorders must target executed backbone/neck modules;
    heads and the teacher's root forward are not called. Evaluation runs only
    the student, as in SingleTeacherDistill.
    """

    def forward(self, batched_inputs):
        if not self.training:
            return self.student(batched_inputs)

        with self.distiller:
            with self.distiller.teacher_recorders, torch.no_grad():
                images = self.teacher.preprocess_image(batched_inputs)
                features = self.teacher.backbone(images.tensor)
                if hasattr(self.teacher, "neck") and self.teacher.neck is not None:
                    features = self.teacher.neck(features)
            with self.distiller.student_recorders:
                student_losses = self.student(batched_inputs)
            losses = {f"student.{name}": value for name, value in student_losses.items()}
            losses.update({f"distill.{name}": value for name, value in self.distiller.compute_distill_losses().items()})
        return losses
