from .base import ParamScheduler

__all__ = ["ConstantParamScheduler"]


class ConstantParamScheduler(ParamScheduler):
    """
    Scheduler that returns a fixed parameter value throughout its interval.

    Examples:
        >>> scheduler = ConstantParamScheduler(0.1, begin=0, end=100, by="iteration")
        >>> scheduler.setup(steps_per_epoch=10, expected_duration=100)
        >>> scheduler(0)
        0.1
        >>> scheduler(50)
        0.1
        >>> scheduler(99)
        0.1
    """

    def __init__(self, value: float, begin: int | tuple[int, str], end: int | tuple[int, str], by: str = "iteration"):
        """
        Initialize a constant parameter schedule.

        Args:
            value: Parameter value returned at every active step.
            begin: Inclusive start boundary of the schedule.
            end: Exclusive end boundary of the schedule.
            by: Default unit for integer boundaries.
        """
        super().__init__(begin, end, by)

        self._value = value

    def __call__(self, step: int) -> float:
        """
        Return the constant parameter value.

        Args:
            step: Zero-based local step at which to evaluate the schedule.

        Returns:
            The configured constant parameter value.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
            ValueError: If ``step`` falls outside ``[0, duration)``.
        """
        # Reading self.duration also rejects an un-setup scheduler. The value does
        # not depend on step, but the bounds contract is enforced all the same.
        if step < 0 or step >= self.duration:
            raise ValueError(f"step must be in [0, {self.duration}), got {step}")

        return self._value
