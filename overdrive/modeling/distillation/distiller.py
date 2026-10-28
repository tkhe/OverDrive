from torch import nn

from .recorder import RecorderManager

__all__ = ["ConfigurableDistiller"]


class ConfigurableDistiller(nn.Module):
    """
    Configure recorders, connectors and named loss argument mappings.

    Each mapping argument contains ``recorder``, ``from_student``, and optional
    ``record_idx``, ``data_idx``, ``connector`` and ``connector_idx``. Dictionaries
    of losses/connectors contain instantiated modules. Connectors are ordinary modules,
    e.g. Conv2d, Upsample or Sequential.
    """

    def __init__(
        self,
        student_recorders,
        teacher_recorders,
        distill_losses,
        loss_forward_mappings,
        connectors=None
    ):
        super().__init__()

        self.student_recorders = RecorderManager(student_recorders)
        self.teacher_recorders = RecorderManager(teacher_recorders)
        self.distill_losses = nn.ModuleDict(distill_losses)
        self.connectors = nn.ModuleDict(connectors or {})
        self.loss_forward_mappings = loss_forward_mappings
        if self.distill_losses.keys() != loss_forward_mappings.keys():
            raise ValueError("distill_losses and loss_forward_mappings must have the same keys")

    def prepare_from_student(self, model):
        self.student_recorders.initialize(model)

    def prepare_from_teacher(self, model):
        self.teacher_recorders.initialize(model)

    def get_record(self, recorder, from_student, record_idx=0, data_idx=None, connector=None, connector_idx=None):
        manager = self.student_recorders if from_student else self.teacher_recorders
        data = manager.get_recorder(recorder).get_record_data(record_idx, data_idx)
        if connector is not None:
            data = self.connectors[connector](data)
        return data if connector_idx is None else data[connector_idx]

    def compute_distill_losses(self):
        return {
            name: self.distill_losses[name](**{argument: self.get_record(**record) for argument, record in mapping.items()})
            for name, mapping in self.loss_forward_mappings.items()
        }

    def clear(self):
        self.student_recorders.clear()
        self.teacher_recorders.clear()

    def close(self):
        self.student_recorders.close()
        self.teacher_recorders.close()

    def __enter__(self):
        self.clear()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Also release saved graphs if a model or loss raises.
        self.clear()
