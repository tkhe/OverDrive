import math

from .base import ParamScheduler

__all__ = ["CosineParamScheduler"]


class CosineParamScheduler(ParamScheduler):
    """
    Scheduler that interpolates between two parameter values along a cosine curve.

    Examples:
        >>> scheduler = CosineParamScheduler(1.0, 0.0, begin=0, end=100)
        >>> scheduler.setup(steps_per_epoch=10, expected_duration=100)
        >>> round(scheduler(0), 6)
        1.0
        >>> round(scheduler(50), 6)
        0.5
        >>> round(scheduler(99), 6)
        0.000247
    """

    def __init__(
        self,
        start_value: float,
        end_value: float,
        begin: int | tuple[int, str],
        end: int | tuple[int, str],
        by: str = "iteration",
    ):
        """
        Initialize a cosine parameter schedule.

        Args:
            start_value: Parameter value at the start of the interval.
            end_value: Parameter value at the end of the cosine interpolation.
            begin: Inclusive start boundary of the schedule.
            end: Exclusive end boundary of the schedule.
            by: Default unit for integer boundaries.
        """
        super().__init__(begin=begin, end=end, by=by)

        self._start_value = start_value
        self._end_value = end_value

    def __call__(self, step: int) -> float:
        """
        Return the cosine-interpolated parameter value for a local step.

        Args:
            step: Zero-based local step at which to evaluate the schedule.

        Returns:
            The cosine-interpolated parameter value.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
            ValueError: If ``step`` falls outside ``[0, duration)``.
        """
        # Reading self.duration also rejects an un-setup scheduler.
        if step < 0 or step >= self.duration:
            raise ValueError(f"step must be in [0, {self.duration}), got {step}")

        where = step / self.duration
        return self._end_value + 0.5 * (self._start_value - self._end_value) * (1 + math.cos(math.pi * where))
