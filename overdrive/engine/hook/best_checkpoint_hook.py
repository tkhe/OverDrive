import logging
import math
import operator

from overdrive.checkpoint import Checkpointer

from .base import Hook
from .priority import HookPriority

__all__ = ["BestCheckpointHook"]


class BestCheckpointHook(Hook):
    def __init__(
        self,
        eval_period: int,
        val_metric: str,
        mode: str = "max",
        file_prefix: str = "model_best",
        checkpointer: Checkpointer | None = None,
    ):
        if mode not in ("max", "min"):
            raise ValueError(f"Expected mode in ('max', 'min'), but got mode={mode}")

        self.logger = logging.getLogger(__name__)
        self.period = eval_period
        self.val_metric = val_metric

        if mode == "max":
            self.compare = operator.gt
        else:
            self.compare = operator.lt

        self._checkpointer = checkpointer
        self.file_prefix = file_prefix
        self.best_metric = None
        self.best_step = None

    @property
    def checkpointer(self):
        return self._checkpointer or self.loop.checkpointer

    def update_best(self, val, step):
        if math.isnan(val) or math.isinf(val):
            return False

        self.best_metric = val
        self.best_step = step
        return True

    def best_checking_by_epoch(self):
        metric_tuple = self.loop.storage.latest().get(self.val_metric)

        if metric_tuple is None:
            self.logger.warning(f"Given val metric {self.val_metric} does not seem to be computed/stored. Will not be checkpointing based on it.")
            return
        else:
            latest_metric, _ = metric_tuple

        if self.best_metric is None:
            if self.update_best(latest_metric, self.loop.epoch):
                self.checkpointer.save(self.file_prefix)
                self.logger.info(f"Saved first model at {self.best_metric:0.5f} @ {self.best_step + 1} epoch")
        elif self.compare(latest_metric, self.best_metric):
            self.checkpointer.save(self.file_prefix)
            self.logger.info(f"Saved best model as latest eval score for {self.val_metric} is {latest_metric:0.5f}, better than last best score {self.best_metric:0.5f} @ epoch {self.best_step + 1}.")
            self.update_best(latest_metric, self.loop.epoch)
        else:
            self.logger.info(f"Not saving as latest eval score for {self.val_metric} is {latest_metric:0.5f}, not better than best score {self.best_metric:0.5f} @ epoch {self.best_step + 1}.")

    def best_checking_by_iteration(self):
        metric_tuple = self.loop.storage.latest().get(self.val_metric)

        if metric_tuple is None:
            self.logger.warning(f"Given val metric {self.val_metric} does not seem to be computed/stored. Will not be checkpointing based on it.")
            return
        else:
            latest_metric, _ = metric_tuple

        if self.best_metric is None:
            if self.update_best(latest_metric, self.loop.iter):
                self.checkpointer.save(self.file_prefix)
                self.logger.info(f"Saved first model at {self.best_metric:0.5f} @ {self.best_step + 1} iteration")
        elif self.compare(latest_metric, self.best_metric):
            self.checkpointer.save(self.file_prefix)
            self.logger.info(
                f"Saved best model as latest eval score for {self.val_metric} is {latest_metric:0.5f}, better than last best score {self.best_metric:0.5f} @ iteration {self.best_step + 1}."
            )
            self.update_best(latest_metric, self.loop.iter)
        else:
            self.logger.info(f"Not saving as latest eval score for {self.val_metric} is {latest_metric:0.5f}, not better than best score {self.best_metric:0.5f} @ iteration {self.best_step + 1}.")

    def after_step(self):
        if self.loop.mode == "epoch":
            return

        next_step = self.loop.iter + 1
        if self.period > 0 and next_step % self.period == 0 and next_step != self.loop.max_iter:
            self.best_checking_by_iteration()

    def after_epoch(self):
        next_epoch = self.loop.epoch + 1
        if self.period > 0 and next_epoch % self.period == 0 and next_epoch != self.loop.max_epoch:
            self.best_checking_by_epoch()

    def after_train(self):
        if self.loop.mode == "epoch":
            if self.loop.epoch + 1 >= self.loop.max_epoch:
                self.best_checking_by_epoch()
        elif self.loop.mode == "iteration":
            if self.loop.iter + 1 >= self.loop.max_iter:
                self.best_checking_by_iteration()
        else:
            raise NotImplementedError(f"Unknown loop mode {self.loop.mode} for BestCheckpointHook.")

    @property
    def priority(self):
        """
        Use the same priority as EvalHook. EvalHook is registered by default,
        while BestCheckpointHook is registered later as a custom hook. Hooks
        with equal priority run in registration order, so BestCheckpointHook
        runs after EvalHook and uses the latest evaluation metrics when
        selecting the best checkpoint.
        """
        return HookPriority.LOW
