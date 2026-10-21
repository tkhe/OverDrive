import torch

from .base import Hook
from .priority import HookPriority

__all__ = ["EmptyCacheHook"]


class EmptyCacheHook(Hook):
    """
    Periodically release unused memory held by the CUDA allocator.

    When ``by_epoch`` is false, the cache is emptied every ``period`` steps
    according to the active loop mode. When it is true, the cache is emptied
    every ``period`` epochs instead.
    """

    def __init__(self, period: int, by_epoch: bool = False):
        """
        Initialize the cache-clearing schedule.

        Args:
            period: Number of steps or epochs between cache-clearing calls.
            by_epoch: Whether to schedule clearing by epoch rather than by
                step or iteration.
        """
        self.period = period
        self.by_epoch = by_epoch

    def after_step(self):
        """
        Clear the CUDA cache when the configured step interval elapses.
        """
        if self.by_epoch:
            return

        if self.loop.mode == "epoch":
            if (self.loop.step_in_epoch + 1) % self.period == 0:
                torch.cuda.empty_cache()
        elif self.loop.mode == "iteration":
            if (self.loop.iter + 1) % self.period == 0:
                torch.cuda.empty_cache()

    def after_epoch(self):
        """
        Clear the CUDA cache when the configured epoch interval elapses.
        """
        if not self.by_epoch:
            return

        if (self.loop.epoch + 1) % self.period == 0:
            torch.cuda.empty_cache()

    @property
    def priority(self):
        """
        Run after other hooks so cleanup happens at the end of a phase.
        """
        return HookPriority.VERY_LOW
