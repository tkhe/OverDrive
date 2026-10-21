from .base import ParamScheduler

__all__ = ["PolynomialDecayParamScheduler"]


class PolynomialDecayParamScheduler(ParamScheduler):
    """
    Scheduler that decays a parameter value to zero along a polynomial curve.

    Examples:
        >>> scheduler = PolynomialDecayParamScheduler(1.0, 0.9, begin=0, end=100)
        >>> scheduler.setup(steps_per_epoch=10, expected_duration=100)
        >>> round(scheduler(0), 6)
        1.0
        >>> round(scheduler(50), 6)
        0.535887
        >>> round(scheduler(99), 6)
        0.015849
    """

    def __init__(
        self,
        base_value: float,
        power: float,
        begin: int | tuple[int, str],
        end: int | tuple[int, str],
        by: str = "iteration",
    ):
        """
        Initialize a polynomial-decay parameter schedule.

        Args:
            base_value: Parameter value at the start of the interval.
            power: Exponent controlling the shape of the decay curve.
            begin: Inclusive start boundary of the schedule.
            end: Exclusive end boundary of the schedule.
            by: Default unit for integer boundaries.
        """
        super().__init__(begin=begin, end=end, by=by)

        self._base_value = base_value
        self._power = power

    def __call__(self, step: int) -> float:
        """
        Return the polynomially decayed parameter value for a local step.

        Args:
            step: Zero-based local step at which to evaluate the schedule.

        Returns:
            The polynomially decayed parameter value.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
            ValueError: If ``step`` falls outside ``[0, duration)``.
        """
        # Reading self.duration also rejects an un-setup scheduler.
        if step < 0 or step >= self.duration:
            raise ValueError(f"step must be in [0, {self.duration}), got {step}")

        where = step / self.duration
        return self._base_value * (1 - where) ** self._power
