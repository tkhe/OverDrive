from torch import nn

__all__ = ["DML"]


class DML(nn.Module):
    """Mutual learning among independently optimized, supervised models.

    All models see the same batch and run once before any distillation loss is
    computed. Training returns a flat loss dictionary; evaluation returns
    predictions keyed by model name. Individual inference is also available
    directly through ``models[name]``.
    """

    def __init__(self, models, distiller):
        super().__init__()
        if len(models) < 2:
            raise ValueError("DML requires at least two students")
        self.models = nn.ModuleDict(models)
        self.distiller = distiller
        self.distiller.prepare(self.models)

    def student_modules(self, name):
        """Expose one student's model, losses and connectors for parameter grouping."""
        return nn.ModuleDict({"model": self.models[name], "distiller": self.distiller.learner_modules(name)})

    def forward(self, batched_inputs):
        if not self.training:
            return {name: model(batched_inputs) for name, model in self.models.items()}

        with self.distiller:
            losses = {}
            for name, model in self.models.items():
                with self.distiller.recorders[name]:
                    task_losses = model(batched_inputs)
                losses.update({f"{name}/gt/{key}": value for key, value in task_losses.items()})
            losses.update(self.distiller.compute_distill_losses())
        return losses
