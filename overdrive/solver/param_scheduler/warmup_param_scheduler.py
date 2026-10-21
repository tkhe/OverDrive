from .base import ParamScheduler
from .base import resolve_value
from .composite_param_scheduler import CompositeParamScheduler
from .constant_param_scheduler import ConstantParamScheduler
from .linear_param_scheduler import LinearParamScheduler

__all__ = ["WarmupParamScheduler"]


class WarmupParamScheduler(CompositeParamScheduler):
    """
    Prepend a warmup phase to a parameter scheduler.

    During warmup, the scheduled value starts at ``warmup_factor`` and either
    increases linearly to ``1.0`` or remains constant. After warmup, values are
    produced by ``scheduler``.

    Warmup occupies ``[0, warmup_length)``, and ``scheduler`` must declare a
    ``begin`` at the iteration where warmup ends. Boundaries stay absolute, so a
    schedule reads as a timeline: warmup over the first 1000 iterations followed
    by cosine decay from iteration 1000 to epoch 300 is written as
    ``warmup_length=1000`` plus ``begin=1000, end=(300, "epoch")``. The two
    boundaries need not share a unit, so a warmup of 500 iterations also lines up
    with a main scheduler starting at epoch 1 when an epoch is 500 iterations.

    Examples:
        >>> from overdrive.solver.param_scheduler import CosineParamScheduler
        >>> from overdrive.solver.param_scheduler.warmup_param_scheduler import WarmupParamScheduler
        >>> main = CosineParamScheduler(1.0, 0.0, begin=2, end=6)
        >>> scheduler = WarmupParamScheduler(
        ...     scheduler=main,
        ...     warmup_length=2,
        ...     warmup_factor=0.001,
        ...     warmup_method="linear",
        ... )
        >>> scheduler.setup(steps_per_epoch=2, expected_duration=6)
        >>> [round(scheduler(step), 3) for step in range(6)]
        [0.001, 0.5, 1.0, 0.854, 0.5, 0.146]
    """

    def __init__(
        self,
        scheduler: ParamScheduler,
        warmup_length,
        warmup_by="iteration",
        warmup_factor=0.001,
        warmup_method="linear",
    ):
        """
        Initialize a scheduler with a warmup phase.

        Args:
            scheduler: Main scheduler used after warmup. Its start boundary must
                match the end of the warmup interval.
            warmup_length: Duration of the warmup phase, expressed as an integer
                or a ``(value, unit)`` pair. Must be positive.
            warmup_by: Unit used when ``warmup_length`` is an integer. Must be
                ``"iteration"`` or ``"epoch"``.
            warmup_factor: Parameter value at the start of warmup for the linear
                method, or throughout warmup for the constant method.
            warmup_method: Warmup shape. Must be ``"linear"`` or ``"constant"``.

        Raises:
            TypeError: If ``warmup_length`` is not an integer or a
                ``(value, unit)`` pair.
            ValueError: If ``warmup_length`` is not positive or ``warmup_method``
                is unsupported.
        """
        # warmup_length may be a plain int, a (value, unit) pair or a ListConfig, so
        # resolve_value does the unpacking and type validation. steps_per_epoch=1 makes
        # the unit conversion a no-op, which is enough to test for zero: a positive
        # number of epochs is a positive number of iterations whatever an epoch is.
        if resolve_value(warmup_length, warmup_by, steps_per_epoch=1) == 0:
            raise ValueError(f"WarmupParamScheduler: warmup_length must be positive, got {warmup_length!r}")

        if warmup_method == "linear":
            warmup_sched = LinearParamScheduler(
                start_value=warmup_factor,
                end_value=1.0,
                begin=0,
                end=warmup_length,
                by=warmup_by,
            )
        elif warmup_method == "constant":
            warmup_sched = ConstantParamScheduler(
                value=warmup_factor,
                begin=0,
                end=warmup_length,
                by=warmup_by,
            )
        else:
            raise ValueError(f"Unknown warmup method: {warmup_method}")

        super().__init__(schedulers=[warmup_sched, scheduler])

        # Boundaries are validated in setup(): warmup and the main scheduler may be
        # expressed in different units, and comparing them needs steps_per_epoch.
        #
        # Read after super().__init__() so that the fields come from the flattened
        # chain. A CompositeParamScheduler builds its own (begin, by) from different
        # sub-schedulers, so those two can disagree -- reading `scheduler` directly
        # would take begin from its first sub-scheduler and the unit from its last.
        # Flattening leaves self._schedulers[1] as the first leaf of the main
        # scheduler, whose boundary and unit always belong together.
        self._warmup_length_raw = warmup_length
        self._warmup_by = warmup_by
        self._main_begin_raw = self._schedulers[1]._begin_raw
        self._main_by = self._schedulers[1].by

    def setup(self, steps_per_epoch: int, expected_duration: int):
        """
        Resolve the warmup and main scheduler boundaries for a training run.

        Args:
            steps_per_epoch: Number of iterations in one epoch.
            expected_duration: Total number of iterations in the training run.

        Raises:
            ValueError: If the main scheduler does not begin where warmup ends.
        """
        # Checked here rather than left to the composite, whose messages refer to
        # sub-scheduler indices the caller never wrote. Unlike warmup_length, the two
        # boundaries may use different units, so comparing them needs steps_per_epoch.
        warmup_end = resolve_value(self._warmup_length_raw, self._warmup_by, steps_per_epoch)
        main_begin = resolve_value(self._main_begin_raw, self._main_by, steps_per_epoch)
        if main_begin != warmup_end:
            raise ValueError(f"WarmupParamScheduler: main scheduler must begin where warmup ends (iteration {warmup_end}), got begin={main_begin}")

        super().setup(steps_per_epoch=steps_per_epoch, expected_duration=expected_duration)
