from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence

__all__ = ["ParamScheduler"]


class ParamScheduler(ABC):
    """
    Base class for parameter schedules over a contiguous training interval.

    Boundaries may be expressed as absolute iteration counts or as
    ``(value, unit)`` pairs, where ``unit`` is ``"epoch"`` or ``"iteration"``.
    Call :meth:`setup` before reading resolved boundaries or evaluating a
    schedule. Subclasses receive zero-based, interval-relative steps through
    :meth:`__call__`.
    """

    def __init__(
        self,
        begin: int | tuple[int, str],
        end: int | tuple[int, str],
        by: str = "iteration",
    ):
        """
        Initialize the schedule definition.

        Args:
            begin: Inclusive start boundary of the schedule.
            end: Exclusive end boundary of the schedule.
            by: Default unit for integer boundaries. Must be ``"epoch"`` or
                ``"iteration"``.
        """
        if not isinstance(by, str) or by not in {"epoch", "iteration"}:
            raise ValueError(f"by must be either 'iteration' or 'epoch', got {by!r}")

        self._begin_raw = begin
        self._end_raw = end
        self._by = by
        self._is_setup = False

    def setup(self, steps_per_epoch: int, expected_duration: int):
        """
        Resolve the schedule boundaries for a training run.

        Args:
            steps_per_epoch: Number of iterations in one epoch.
            expected_duration: Number of iterations this schedule is required to
                cover. For a root scheduler this is the length of the whole
                training run; for a sub-scheduler of a
                :class:`CompositeParamScheduler` it is the length of that
                sub-scheduler's own interval.

        Raises:
            TypeError: If ``steps_per_epoch`` is not an integer.
            ValueError: If ``steps_per_epoch`` is not positive, or if the resolved
                schedule duration is not positive or does not equal
                ``expected_duration``.
        """
        # Checked directly because an "iteration" schedule never multiplies by it, so a
        # nonsensical value would otherwise pass unnoticed on that path and surface as a
        # confusing duration error on the "epoch" one.
        if isinstance(steps_per_epoch, bool) or not isinstance(steps_per_epoch, int):
            raise TypeError(f"steps_per_epoch must be an int, got {steps_per_epoch!r}")

        if steps_per_epoch <= 0:
            raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")

        # Validated as locals and committed only once everything holds, so a rejected
        # setup() leaves the scheduler exactly as it was rather than half-resolved.
        begin = resolve_value(self._begin_raw, self.by, steps_per_epoch)
        end = resolve_value(self._end_raw, self.by, steps_per_epoch)
        duration = end - begin

        if duration <= 0:
            raise ValueError(f"duration must be greater than 0, got {duration}")

        if duration != expected_duration:
            raise ValueError(f"duration must equal expected_duration, got {duration} and {expected_duration}, respectively")

        self._begin = begin
        self._end = end
        self._duration = duration
        self._is_setup = True

    @abstractmethod
    def __call__(self, step: int) -> float:
        """
        Return the scheduled parameter value for a local step.

        Subclasses must implement this method. The input ``step`` is relative to
        :attr:`begin`, and implementations are required to reject any ``step``
        outside ``[0, duration)`` with a :exc:`ValueError` rather than
        extrapolating. Reading :attr:`duration` rather than ``self._duration``
        additionally makes an un-setup scheduler raise :exc:`RuntimeError`.

        Args:
            step: Zero-based local step at which to evaluate the schedule.

        Returns:
            The scheduled parameter value.
        """
        raise NotImplementedError("ParamScheduler subclasses must implement __call__().")

    @property
    def begin(self):
        """
        Inclusive start boundary, resolved to an absolute iteration.
        """
        if not self._is_setup:
            raise RuntimeError("setup() must be called before accessing 'begin'.")
        return self._begin

    @property
    def end(self):
        """
        Exclusive end boundary, resolved to an absolute iteration.
        """
        if not self._is_setup:
            raise RuntimeError("setup() must be called before accessing 'end'.")
        return self._end

    @property
    def duration(self):
        """
        Number of iterations in the resolved schedule interval.
        """
        if not self._is_setup:
            raise RuntimeError("setup() must be called before accessing 'duration'.")
        return self._duration

    @property
    def by(self):
        """
        Default unit used for integer schedule boundaries.
        """
        return self._by


def resolve_value(value: int | tuple[int, str], default_by: str, steps_per_epoch: int) -> int:
    """
    Resolve a boundary specification to an absolute iteration count.

    Args:
        value: A non-negative integer boundary or a ``(value, unit)`` pair.
        default_by: Unit to use when ``value`` is an integer. Must be
            ``"epoch"`` or ``"iteration"``.
        steps_per_epoch: Number of iterations in one epoch.

    Returns:
        The boundary as an absolute iteration count.

    Raises:
        TypeError: If ``value`` is neither a non-negative integer nor a
            two-element sequence.
        ValueError: If ``default_by`` or the resolved unit is unsupported, or if
            the boundary is negative.
    """
    if not isinstance(default_by, str) or default_by not in {"epoch", "iteration"}:
        raise ValueError(f"default_by must be either 'iteration' or 'epoch', got {default_by!r}")

    # bool is a subclass of int, so it has to be rejected before the int branch.
    if isinstance(value, bool):
        raise TypeError(f"value must be an int or a (value, unit) pair, got {value!r}")

    if isinstance(value, int):
        v, by = value, default_by
    elif isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2:
        v, by = value
    else:
        raise TypeError(f"value must be an int or a (value, unit) pair, got {value!r}")

    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f"value must be an int, got {v!r}")

    if v < 0:
        raise ValueError(f"value must be non-negative, got {v}")

    if not isinstance(by, str) or by not in {"epoch", "iteration"}:
        raise ValueError(f"by must be either 'iteration' or 'epoch', got {by!r}")

    return v * steps_per_epoch if by == "epoch" else v
