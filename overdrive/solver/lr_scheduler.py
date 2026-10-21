import torch
from torch.optim.lr_scheduler import LRScheduler

from .param_scheduler import ParamScheduler

__all__ = ["LRMultiplier"]


class LRMultiplier(LRScheduler):
    """
    Scale an optimizer's base learning rates with a parameter scheduler.

    At each step, every parameter group's base learning rate is multiplied by
    the scalar returned by ``multiplier``.

    Examples:
        >>> from overdrive.solver.param_scheduler import LinearParamScheduler
        >>> parameter = torch.nn.Parameter(torch.tensor(0.0))
        >>> optimizer = torch.optim.SGD([parameter], lr=0.1)
        >>> multiplier = LinearParamScheduler(1.0, 0.0, begin=0, end=1000)
        >>> scheduler = LRMultiplier(
        ...     optimizer,
        ...     multiplier,
        ...     max_iter=1000,
        ...     steps_per_epoch=100,
        ... )
        >>> scheduler.get_last_lr()
        [0.1]
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        multiplier: ParamScheduler,
        max_iter: int,
        steps_per_epoch: int,
        last_iter: int = -1,
    ):
        """
        Initialize a learning-rate multiplier.

        Args:
            optimizer: Optimizer whose learning rates are scheduled.
            multiplier: Parameter scheduler that produces the scalar applied
                to every base learning rate.
            max_iter: Total number of training iterations.
            steps_per_epoch: Number of iterations in one epoch.
            last_iter: Index of the last completed iteration. Use ``-1`` when
                starting a new schedule.

        Raises:
            ValueError: If ``multiplier`` is not a :class:`ParamScheduler`.
        """
        if not isinstance(multiplier, ParamScheduler):
            raise ValueError(f"multiplier must be an instance of ParamScheduler. Got {multiplier} instead.")

        self._multiplier = multiplier
        self._max_iter = max_iter
        self._multiplier.setup(steps_per_epoch=steps_per_epoch, expected_duration=max_iter)

        if self._multiplier.begin != 0:
            raise ValueError(f"multiplier must cover the whole run and therefore begin at iteration 0, got begin={self._multiplier.begin}")

        super().__init__(optimizer, last_epoch=last_iter)

    def state_dict(self):
        """
        Return the scheduler state, excluding the optimizer and the schedule itself.

        Mirrors :class:`~torch.optim.lr_scheduler.LRScheduler` by copying everything
        but the optimizer, which is checkpointed separately. ``_multiplier`` is left
        out as well: it is a schedule definition rebuilt from config, not run state.
        A fixed field list was tried first and silently dropped internal bookkeeping
        that torch maintains, so state is excluded by name rather than included.
        """
        return {key: value for key, value in self.__dict__.items() if key not in ("optimizer", "_multiplier")}

    def get_lr(self) -> list[float]:
        # torch's LRScheduler steps once during __init__, so a loop running max_iter
        # iterations ends with last_epoch == max_iter -- one past the schedule's last
        # valid step. Hold the final value for that one step instead of raising.
        if self.last_epoch > self._max_iter:
            raise ValueError(f"schedule covers {self._max_iter} iteration(s), but the scheduler has been stepped to iteration {self.last_epoch}")

        step = min(self.last_epoch, self._max_iter - 1)
        multiplier = self._multiplier(step)
        return [base_lr * multiplier for base_lr in self.base_lrs]
