import logging
from collections import Counter

from torch.optim.lr_scheduler import LRScheduler

from overdrive.solver import LRMultiplier
from overdrive.solver import ParamScheduler

from .base import Hook
from .priority import HookPriority

__all__ = ["LRSchedulerHook"]


class LRSchedulerHook(Hook):
    """
    A hook that steps the learning rate scheduler during training.

    It updates the learning rate after every training step and logs the
    current value to the loop's storage. When a :obj:`ParamScheduler` is
    provided, it is wrapped into a :obj:`LRMultiplier` at the start of
    training so it can be driven by the loop's step/iteration count.
    """

    def __init__(self, optimizer=None, scheduler=None):
        """
        Args:
            optimizer (torch.optim.Optimizer, optional): Optimizer whose
                learning rate is scheduled. If ``None``, the optimizer is
                resolved from ``self.loop.backend.optimizer`` in
                :meth:`before_train`.
            scheduler (LRScheduler or ParamScheduler, optional): Learning rate
                scheduler. If ``None``, the scheduler is resolved from
                ``self.loop.scheduler``. A :obj:`ParamScheduler` is wrapped
                into a :obj:`LRMultiplier` in :meth:`before_train`.
        """
        self._optimizer = optimizer
        self._scheduler = scheduler

    def before_train(self):
        """
        Resolve the optimizer and finalize the scheduler before training.

        Falls back to the loop's optimizer when none was given, wraps a
        :obj:`ParamScheduler` into a :obj:`LRMultiplier` using the loop's
        step counts, and records the param group whose learning rate is
        used for logging.
        """
        self._optimizer = self._optimizer or self.loop.backend.optimizer
        if isinstance(self.scheduler, ParamScheduler):
            max_step = self.loop.max_epoch * self.loop.max_step_in_epoch if self.loop.mode == "epoch" else self.loop.max_iter
            last_step = self.loop.epoch * self.loop.max_step_in_epoch + self.loop.step_in_epoch - 1 if self.loop.mode == "epoch" else self.loop.iter - 1
            # Iteration-based schedules do not need epoch conversion; one
            # iteration is the smallest valid conversion unit.
            steps_per_epoch = self.loop.max_step_in_epoch if self.loop.mode == "epoch" else 1
            self._scheduler = LRMultiplier(
                self._optimizer,
                self.scheduler,
                max_step,
                steps_per_epoch=steps_per_epoch,
                last_iter=last_step,
            )
        self._best_param_group_id = LRSchedulerHook.get_best_param_group_id(self._optimizer)

    @staticmethod
    def get_best_param_group_id(optimizer):
        """
        Pick the param group that best represents the model's learning rate.

        When every param group holds a single parameter, the most common
        learning rate across groups is chosen. Otherwise the group with the
        largest number of parameters is used.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to inspect.

        Returns:
            int: Index of the selected param group in
            ``optimizer.param_groups``.
        """
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    return i
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    return i

    def after_step(self):
        """
        Log the current learning rate and advance the scheduler by one step.
        """
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        self.loop.storage.put_scalar("lr", lr, smoothing_hint=False)
        self.scheduler.step()

    @property
    def scheduler(self):
        """
        LRScheduler or ParamScheduler: The active scheduler, falling back to
        ``self.loop.scheduler`` when none was configured on the hook.
        """
        return self._scheduler or self.loop.scheduler

    def state_dict(self):
        """
        Returns:
            dict: The scheduler's state when it is an :obj:`LRScheduler`,
            otherwise an empty dict.
        """
        if isinstance(self.scheduler, LRScheduler):
            return self.scheduler.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        """
        Restore the scheduler state from a checkpoint.

        Args:
            state_dict (dict): State produced by :meth:`state_dict`. Ignored
                when the scheduler is not an :obj:`LRScheduler`.
        """
        if isinstance(self.scheduler, LRScheduler):
            logger = logging.getLogger(__name__)
            logger.info("Loading scheduler from state_dict ...")
            self.scheduler.load_state_dict(state_dict)

    @property
    def priority(self):
        return HookPriority.HIGH
