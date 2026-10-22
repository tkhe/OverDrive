import bisect

from .base import ParamScheduler
from .base import resolve_value

__all__ = ["MultiStepParamScheduler"]


class MultiStepParamScheduler(ParamScheduler):
    """
    Scheduler that switches parameter values at ordered milestones.

    Examples:
        >>> scheduler = MultiStepParamScheduler([1.0, 0.1, 0.01], [30, 60], begin=0, end=100)
        >>> scheduler.setup(steps_per_epoch=1, expected_duration=100)
        >>> scheduler(0)
        1.0
        >>> scheduler(50)
        0.1
        >>> scheduler(99)
        0.01
    """

    def __init__(
        self,
        values: list[float],
        milestones: list[int | tuple[int, str]],
        begin: int | tuple[int, str],
        end: int | tuple[int, str],
        by: str = "iteration",
    ):
        """
        Initialize a multi-step parameter schedule.

        Args:
            values: Parameter values for successive milestone intervals. Must
                contain one more item than ``milestones``.
            milestones: Ordered boundaries at which to switch to the next
                parameter value.
            begin: Inclusive start boundary of the schedule.
            end: Exclusive end boundary of the schedule.
            by: Default unit for integer boundaries.

        Raises:
            ValueError: If ``values`` does not contain exactly one more item
                than ``milestones``.
        """
        super().__init__(begin=begin, end=end, by=by)

        # Checked here rather than in setup(): it needs no unit resolution, and
        # otherwise a mismatch only surfaces as an IndexError from __call__().
        if len(values) != len(milestones) + 1:
            raise ValueError(f"values must contain exactly one more item than milestones, got {len(values)} value(s) and {len(milestones)} milestone(s)")

        # Copied into tuples so that later mutation of the caller's lists cannot
        # invalidate the length check above.
        self._values = tuple(values)
        self._milestones_raw = tuple(milestones)

    def setup(self, steps_per_epoch: int, expected_duration: int):
        """
        Resolve schedule boundaries and milestones for a training run.

        Args:
            steps_per_epoch: Number of iterations in one epoch.
            expected_duration: Number of iterations this schedule is required to
                cover.

        Raises:
            ValueError: If a milestone falls outside ``(begin, end)`` or if the
                milestones are not strictly increasing.
        """
        # Resolved locally, and again in super().setup() below, so that nothing is
        # written to self until every milestone has been validated. Committing first
        # would leave a scheduler that reports itself as set up while holding only
        # part of its milestones.
        begin = resolve_value(self._begin_raw, self._by, steps_per_epoch)
        end = resolve_value(self._end_raw, self._by, steps_per_epoch)

        resolved_milestones = []
        for i, m in enumerate(self._milestones_raw):
            abs_iter = resolve_value(m, self._by, steps_per_epoch)

            # A milestone at begin or end would leave one of self._values unreachable,
            # which almost always means the schedule was misconfigured.
            if abs_iter <= begin or abs_iter >= end:
                raise ValueError(f"milestone {i} must fall inside ({begin}, {end}), got {abs_iter}")
            if resolved_milestones and abs_iter <= resolved_milestones[-1] + begin:
                raise ValueError(f"milestones must be strictly increasing, got {abs_iter} at index {i} after {resolved_milestones[-1] + begin}")

            resolved_milestones.append(abs_iter - begin)

        super().setup(steps_per_epoch=steps_per_epoch, expected_duration=expected_duration)

        self._resolved_milestones = resolved_milestones

    def __call__(self, step: int) -> float:
        """
        Return the parameter value for a local step.

        Args:
            step: Zero-based local step at which to evaluate the schedule.

        Returns:
            The parameter value for the milestone interval containing ``step``.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
            ValueError: If ``step`` falls outside ``[0, duration)``.
        """
        # Reading self.duration also rejects an un-setup scheduler.
        if step < 0 or step >= self.duration:
            raise ValueError(f"step must be in [0, {self.duration}), got {step}")

        return self._values[bisect.bisect_right(self._resolved_milestones, step)]
