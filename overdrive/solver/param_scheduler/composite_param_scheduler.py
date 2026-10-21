import bisect
from collections.abc import Sequence

from .base import ParamScheduler
from .base import resolve_value

__all__ = ["CompositeParamScheduler"]


class CompositeParamScheduler(ParamScheduler):
    """
    Scheduler that chains sub-schedulers over consecutive training intervals.

    The sub-schedulers must tile the composite interval exactly: consecutive
    intervals have to meet end-to-end, with no overlap, no gap, and in
    increasing order. ``[0, 50)`` followed by ``[50, 100)`` is valid, while
    ``[50, 100)`` followed by ``[0, 50)`` is not, even though those two
    intervals neither overlap nor leave a gap.

    Nested composites are flattened on construction, so a composite is always a
    root scheduler covering the whole training run: it must begin at 0 and span
    exactly ``total_steps``.

    Examples:
        >>> from overdrive.solver.param_scheduler import ConstantParamScheduler
        >>> warmup = ConstantParamScheduler(0.01, begin=0, end=50)
        >>> main = ConstantParamScheduler(0.1, begin=50, end=100)
        >>> scheduler = CompositeParamScheduler([warmup, main])
        >>> scheduler.setup(steps_per_epoch=10, expected_duration=100)
        >>> scheduler.begin, scheduler.end, scheduler.duration
        (0, 100, 100)
        >>> warmup.duration, main.duration
        (50, 50)
        >>> scheduler(0), scheduler(49)
        (0.01, 0.01)
        >>> scheduler(50), scheduler(99)
        (0.1, 0.1)
    """

    def __init__(self, schedulers: Sequence[ParamScheduler]):
        """
        Initialize a composite over a sequence of sub-schedulers.

        Args:
            schedulers: Sub-schedulers tiling the composite interval, in increasing
                order. Nested composites are flattened.

        Raises:
            ValueError: If ``schedulers`` is empty.
        """
        if len(schedulers) == 0:
            raise ValueError("schedulers must not be empty")

        self._schedulers: list[ParamScheduler] = []

        for scheduler in schedulers:
            if isinstance(scheduler, CompositeParamScheduler):
                self._schedulers.extend(scheduler._schedulers)
            else:
                self._schedulers.append(scheduler)

        super().__init__(begin=self._schedulers[0]._begin_raw, end=self._schedulers[-1]._end_raw, by=self._schedulers[-1]._by)

    def setup(self, steps_per_epoch: int, expected_duration: int):
        """
        Resolve the composite and sub-scheduler boundaries for a training run.

        Each sub-scheduler's boundaries are resolved with its own ``by`` unit,
        checked for contiguity, and then used to set up that sub-scheduler over
        its own duration. The composite boundaries are derived from the
        resulting chain rather than resolved independently, so a mix of
        ``"epoch"`` and ``"iteration"`` sub-schedulers stays consistent.

        Args:
            steps_per_epoch: Number of iterations in one epoch.
            expected_duration: Number of iterations this schedule is required to
                cover. A composite is always a root scheduler, so this is the
                length of the whole training run.

        Raises:
            ValueError: If any sub-scheduler interval is empty, if consecutive
                sub-scheduler intervals overlap, leave a gap, or appear out of
                order, if the chain does not start at 0, or if the resolved
                composite duration is not positive or does not equal
                ``expected_duration``.
            TypeError: If ``steps_per_epoch`` is not an integer.
        """
        # Repeated from ParamScheduler.setup(), which this override never calls.
        if isinstance(steps_per_epoch, bool) or not isinstance(steps_per_epoch, int):
            raise TypeError(f"steps_per_epoch must be an int, got {steps_per_epoch!r}")
        if steps_per_epoch <= 0:
            raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")

        # Resolve each sub-scheduler's absolute interval up front. The boundaries
        # are needed to derive the per-scheduler total_steps below, but they are
        # only stored on a sub-scheduler once its own setup() has run.
        intervals: list[tuple[int, int]] = []
        for i, scheduler in enumerate(self._schedulers):
            begin = resolve_value(scheduler._begin_raw, scheduler.by, steps_per_epoch)
            end = resolve_value(scheduler._end_raw, scheduler.by, steps_per_epoch)
            if end <= begin:
                raise ValueError(f"scheduler {i} must span a positive duration, got [{begin}, {end})")
            intervals.append((begin, end))

        # Check that consecutive intervals meet end-to-end. This must happen
        # before the duration checks below, so that a misordered chain is
        # reported as such instead of as a duration mismatch.
        for i in range(1, len(intervals)):
            prev_begin, prev_end = intervals[i - 1]
            begin, end = intervals[i]

            if begin == prev_end:
                continue

            prev_interval = f"scheduler {i - 1} [{prev_begin}, {prev_end})"
            interval = f"scheduler {i} [{begin}, {end})"

            if begin > prev_end:
                raise ValueError(f"schedulers must be contiguous: {prev_interval} and {interval} leave a gap of {begin - prev_end} iteration(s)")
            if end <= prev_begin:
                raise ValueError(f"schedulers must be given in increasing order: {prev_interval} is scheduled after {interval}")
            raise ValueError(f"schedulers must not overlap: {prev_interval} and {interval} overlap over [{begin}, {min(prev_end, end)})")

        # Derive the composite boundaries from the sub-scheduler chain instead of
        # resolving self._begin_raw / self._end_raw, whose units come from
        # different sub-schedulers and may disagree. Kept in locals until every check
        # below passes, so a rejected setup() leaves the composite untouched.
        composite_begin = intervals[0][0]
        composite_end = intervals[-1][1]
        duration = composite_end - composite_begin

        if duration <= 0:
            raise ValueError(f"duration must be greater than 0, got {duration}")
        # Nested composites are flattened on construction, so a composite can never be a
        # sub-scheduler: it always covers the whole run and therefore starts at 0. Sub-schedulers
        # may be offset, a composite may not.
        if composite_begin != 0:
            raise ValueError(f"CompositeParamScheduler is always a root scheduler covering the whole run and must begin at 0, got {composite_begin}")
        if duration != expected_duration:
            raise ValueError(f"duration must equal expected_duration, got {duration} and {expected_duration}, respectively")

        for scheduler, (begin, end) in zip(self._schedulers, intervals, strict=True):
            scheduler.setup(steps_per_epoch=steps_per_epoch, expected_duration=end - begin)

        self._begin = composite_begin
        self._end = composite_end
        self._duration = duration
        self._intervals = intervals
        # Ascending sub-scheduler starts, for binary search in __call__(). The contiguity
        # check above guarantees they are strictly increasing.
        self._begins = [begin for begin, _ in intervals]
        self._is_setup = True

    def __call__(self, step: int) -> float:
        """
        Return the parameter value produced by the sub-scheduler covering a step.

        Args:
            step: Zero-based local step at which to evaluate the schedule.

        Returns:
            The value returned by the active sub-scheduler, evaluated at the
            corresponding step local to that sub-scheduler.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
            ValueError: If ``step`` falls outside ``[0, duration)``.
        """
        # Reading self.duration also rejects an un-setup scheduler.
        if step < 0 or step >= self.duration:
            raise ValueError(f"step must be in [0, {self.duration}), got {step}")

        # setup() enforces self._begin == 0, so a local step is also an absolute one and needs
        # no rebasing here. Rebasing onto the sub-scheduler below is still required.
        i = bisect.bisect_right(self._begins, step) - 1
        return self._schedulers[i](step - self._begins[i])
