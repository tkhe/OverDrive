import logging
import math
import operator

from overdrive.checkpoint.dml_checkpoint import DMLBestCheckpointer
from overdrive.utils import comm

from .base import Hook
from .priority import HookPriority

__all__ = ["DMLBestCheckpointHook"]


class DMLBestCheckpointHook(Hook):
    """Track each selected student's best metric in one checkpointable hook."""

    def __init__(self, students, eval_period, val_metric, mode="max", file_prefix="model_best"):
        self.students = list(students)
        self.period = eval_period
        self.val_metric = val_metric
        self.compare = {"max": operator.gt, "min": operator.lt}[mode]
        self.file_prefix = file_prefix
        self.best = {}

    def before_train(self):
        checkpointer = self.loop.checkpointer
        self.checkpointer = DMLBestCheckpointer(checkpointer.model, checkpointer.save_dir, **checkpointer.checkpointables)

    def _save_best(self):
        if not comm.is_main_process():
            return
        improved = []
        step = min(self.loop.epoch, self.loop.max_epoch - 1) if self.loop.mode == "epoch" else min(self.loop.iter, self.loop.max_iter - 1)
        for name in self.students:
            metric = f"{name}/{self.val_metric}"
            latest = self.loop.storage.latest().get(metric)
            if latest is None:
                logging.getLogger(__name__).warning("Evaluation metric %s was not produced", metric)
                continue
            value = latest[0]
            if math.isfinite(value) and (name not in self.best or self.compare(value, self.best[name]["metric"])):
                self.best[name] = {"metric": value, "step": step}
                improved.append(name)
        # Update all records before writing any complete snapshot.
        for name in improved:
            self.checkpointer.save(f"{self.file_prefix}_{name}")

    def after_step(self):
        if self.loop.mode == "iteration" and self.period > 0:
            step = self.loop.iter + 1
            if step % self.period == 0 and step != self.loop.max_iter:
                self._save_best()

    def after_epoch(self):
        if self.period > 0:
            epoch = self.loop.epoch + 1
            if epoch % self.period == 0 and epoch != self.loop.max_epoch:
                self._save_best()

    def after_train(self):
        completed = self.loop.epoch == self.loop.max_epoch if self.loop.mode == "epoch" else self.loop.iter == self.loop.max_iter
        if completed:
            self._save_best()

    def state_dict(self):
        return {"best": self.best}

    def load_state_dict(self, state_dict):
        self.best = state_dict["best"]

    @property
    def priority(self):
        return HookPriority.LOW
