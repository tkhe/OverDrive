import logging

from torch.optim.lr_scheduler import LRScheduler

from overdrive.solver import LRMultiplier
from overdrive.solver import ParamScheduler

from .base import Hook
from .lr_scheduler_hook import LRSchedulerHook
from .priority import HookPriority

__all__ = ["MultiLRSchedulerHook", "MultiLRLoggerHook"]


class MultiLRSchedulerHook(Hook):
    """One checkpointable hook for all independently scheduled optimizers."""

    def __init__(self, schedulers):
        self.schedulers = dict(schedulers)
        self._pending_state = None
        self._initialized = False

    def before_train(self):
        self.optimizers = self.loop.backend.optimizers
        steps_per_epoch = self.loop.max_step_in_epoch if self.loop.mode == "epoch" else 1
        max_step = self.loop.max_epoch * steps_per_epoch if self.loop.mode == "epoch" else self.loop.max_iter
        start_step = self.loop.epoch * steps_per_epoch if self.loop.mode == "epoch" else self.loop.iter
        restored_lrs = {name: [group["lr"] for group in optimizer.param_groups] for name, optimizer in self.optimizers.items()}
        for name, scheduler in self.schedulers.items():
            if isinstance(scheduler, ParamScheduler):
                self.schedulers[name] = LRMultiplier(
                    self.optimizers[name], scheduler, max_step, steps_per_epoch=steps_per_epoch, last_iter=start_step - 1
                )
        self._initialized = True
        if self._pending_state is not None:
            self.load_state_dict(self._pending_state)
            # LRScheduler's constructor performs an initial step. Preserve the
            # optimizer LR restored by the backend, including at schedule boundaries.
            for name, optimizer in self.optimizers.items():
                for group, lr in zip(optimizer.param_groups, restored_lrs[name], strict=True):
                    group["lr"] = lr
            self._pending_state = None
        self._param_groups = {name: LRSchedulerHook.get_best_param_group_id(optimizer) for name, optimizer in self.optimizers.items()}

    def after_step(self):
        for name, scheduler in self.schedulers.items():
            lr = self.optimizers[name].param_groups[self._param_groups[name]]["lr"]
            self.loop.storage.put_scalar(f"{name}/lr", lr, smoothing_hint=False)
            scheduler.step()

    def state_dict(self):
        if self._pending_state is not None:
            return self._pending_state
        return {name: scheduler.state_dict() for name, scheduler in self.schedulers.items() if isinstance(scheduler, LRScheduler)}

    def load_state_dict(self, state_dict):
        if not self._initialized:
            self._pending_state = state_dict
            return
        for name, state in state_dict.items():
            self.schedulers[name].load_state_dict(state)

    @property
    def priority(self):
        return HookPriority.HIGH


class MultiLRLoggerHook(Hook):
    """Print student learning rates alongside the existing loss/timing logger."""

    def __init__(self, period=10):
        self.period = period

    def after_step(self):
        step = self.loop.step_in_epoch + 1 if self.loop.mode == "epoch" else self.loop.iter + 1
        last_step = self.loop.max_step_in_epoch if self.loop.mode == "epoch" else self.loop.max_iter
        if step % self.period == 0 or step == last_step:
            storage = self.loop.storage
            rates = "  ".join(f"{name}/lr: {storage.history(f'{name}/lr').latest():.5g}" for name in self.loop.backend.optimizers)
            logging.getLogger("overdrive").info(rates)
