from collections.abc import Callable

from overdrive.evaluation.testing import flatten_results_dict
from overdrive.utils import comm

from .base import Hook
from .priority import HookPriority

__all__ = ["EvalHook"]


class EvalHook(Hook):
    """
    Periodically run an evaluation function and record its results.

    Evaluation runs are skipped at the final step or epoch because the final
    evaluation is performed by :meth:`after_train`. Nested result dictionaries
    are flattened before being written to the active event storage.
    """

    def __init__(self, period: int, eval_function: Callable):
        """
        Initialize the evaluation schedule.

        Args:
            period: Number of iterations or epochs between evaluations.
            eval_function: Callable returning a nested dictionary of numeric
                evaluation results.
        """
        self.period = period
        self.func = eval_function

    def _do_eval(self):
        """
        Run evaluation, validate its results, and record the flattened values.
        """
        results = self.func()

        if results:
            if not (isinstance(results, dict)):
                raise TypeError(f"Eval function must return a dict. Got {results} instead.")

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(f"[EvalHook] eval_function should return a nested dict of float. Got '{k}: {v}' instead.") from e

            self.loop.storage.put_scalars(**flattened_results, smoothing_hint=False)

        comm.synchronize()

    def after_step(self):
        """
        Run an evaluation when the configured iteration interval elapses.
        """
        if self.loop.mode == "iteration":
            next_step = self.loop.iter + 1
            if self.period > 0 and next_step % self.period == 0:
                # do last eval in after_train
                if next_step != self.loop.max_iter:
                    self._do_eval()

    def after_epoch(self):
        """
        Run an evaluation when the configured epoch interval elapses.
        """
        if self.loop.mode == "epoch":
            next_epoch = self.loop.epoch + 1
            if self.period > 0 and next_epoch % self.period == 0:
                # do last eval in after_train:
                if next_epoch != self.loop.max_epoch:
                    self._do_eval()

    def after_train(self):
        """
        Run the final evaluation after the training loop completes.
        """
        if self.loop.mode == "epoch" and self.loop.epoch == self.loop.max_epoch:
            self._do_eval()
        elif self.loop.mode == "iteration" and self.loop.iter == self.loop.max_iter:
            self._do_eval()

    @property
    def priority(self):
        """
        Run evaluation before lower-priority cleanup hooks.
        """
        return HookPriority.LOW
