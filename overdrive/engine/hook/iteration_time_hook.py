import datetime
import logging
import time

from overdrive.utils.timer import Timer

from .base import Hook
from .priority import HookPriority

__all__ = ["IterationTimeHook"]


class IterationTimeHook(Hook):
    """
    Measure iteration time and report training throughput.

    Warmup steps are excluded from per-step timing so timer initialization and
    early runtime effects do not skew the reported iteration duration.
    """

    def __init__(self, warmup_step: int = 3):
        """
        Initialize the iteration timers.

        Args:
            warmup_step: Number of initial steps to exclude from timing
                metrics.
        """
        super().__init__()

        self.warmup_step = warmup_step
        self.step_timer = Timer()
        self.start_time = time.perf_counter()
        self.total_timer = Timer()
        self.step = 0

    def before_train(self):
        """
        Reset the total training timer before the loop starts.
        """
        self.start_time = time.perf_counter()
        self.total_timer.reset()
        self.total_timer.pause()

    def before_step(self):
        """
        Start timing the current training step.
        """
        self.step_timer.reset()
        self.total_timer.resume()

    def after_step(self):
        """
        Record the step duration after warmup and pause total timing.
        """
        self.step += 1
        if self.step > self.warmup_step:
            sec = self.step_timer.seconds()
            self.loop.storage.put_scalars(time=sec)
        else:
            self.start_time = time.perf_counter()
            self.total_timer.reset()

        self.total_timer.pause()

    def after_train(self):
        """
        Log overall training time and the time spent in hooks.
        """
        logger = logging.getLogger(__name__)
        total_time = time.perf_counter() - self.start_time
        total_time_minus_hooks = self.total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        if self.step > 0 and total_time_minus_hooks > 0:
            logger.info(f"Overall training speed: {self.step} iterations in {str(datetime.timedelta(seconds=int(total_time_minus_hooks)))} ({total_time_minus_hooks / self.step:.4f} s / it)")

        logger.info(f"Total training time: {str(datetime.timedelta(seconds=int(total_time)))} ({str(datetime.timedelta(seconds=int(hook_time)))} on hooks)")

    @property
    def priority(self):
        """
        Run with normal hook priority.
        """
        return HookPriority.NORMAL
