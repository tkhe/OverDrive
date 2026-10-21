from .base import ParamScheduler

__all__ = ["ExponentialParamScheduler"]


class ExponentialParamScheduler(ParamScheduler):
    """
    Scheduler that exponentially scales a parameter value over its interval.

    Examples:
        >>> scheduler = ExponentialParamScheduler(1.0, 0.25, begin=0, end=100)
        >>> scheduler.setup(steps_per_epoch=10, expected_duration=100)
        >>> round(scheduler(0), 6)
        1.0
        >>> round(scheduler(50), 6)
        0.5
        >>> round(scheduler(99), 6)
        0.25349
    """

    def __init__(
        self,
        start_value: float,
        decay: float,
        begin: int | tuple[int, str],
        end: int | tuple[int, str],
        by: str = "iteration",
    ):
        """
        Initialize an exponential parameter schedule.

        Args:
            start_value: Parameter value at the start of the interval.
            decay: Multiplicative factor applied across the full interval. Must be
                positive.
            begin: Inclusive start boundary of the schedule.
            end: Exclusive end boundary of the schedule.
            by: Default unit for integer boundaries.

        Raises:
            ValueError: If ``decay`` is not positive.
        """
        super().__init__(begin=begin, end=end, by=by)

        # decay is raised to a fractional power, so a negative base yields a complex
        # number and zero collapses the schedule to 0 after the very first step.
        if decay <= 0:
            raise ValueError(f"decay must be positive, got {decay}")

        self._start_value = start_value
        self._decay = decay

    def __call__(self, step: int) -> float:
        """
        Return the exponentially scaled parameter value for a local step.

        Args:
            step: Zero-based local step at which to evaluate the schedule.

        Returns:
            The exponentially scaled parameter value.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
            ValueError: If ``step`` falls outside ``[0, duration)``.
        """
        # Reading self.duration also rejects an un-setup scheduler.
        if step < 0 or step >= self.duration:
            raise ValueError(f"step must be in [0, {self.duration}), got {step}")

        where = step / self.duration
        return self._start_value * self._decay**where
