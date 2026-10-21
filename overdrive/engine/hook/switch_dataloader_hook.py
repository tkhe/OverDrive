import logging
from collections.abc import Callable
from collections.abc import Iterable

from overdrive.data.utils import compute_max_step_in_epoch

from .base import Hook

__all__ = ["SwitchDataloaderHook"]


class SwitchDataloaderHook(Hook):
    """
    Switch the training dataloader at an epoch or iteration boundary.
    """

    def __init__(
        self,
        switch_step: int,
        dataloader_builder: Callable[[], Iterable],
        by: str = "epoch",
    ):
        """
        Args:
            switch_step: Zero-based epoch or iteration at which to switch dataloaders.
            dataloader_builder: Callable that builds the new dataloader.
            by: Whether switch_step counts epochs or iterations.
        """
        if switch_step < 0:
            raise ValueError("switch_step must be nonnegative")
        if by not in ("epoch", "iteration"):
            raise ValueError("by must be 'epoch' or 'iteration'")
        if not callable(dataloader_builder):
            raise TypeError("dataloader_builder must be callable")

        self.logger = logging.getLogger(__name__)
        self.switch_step = switch_step
        self.by = by
        self.dataloader_builder = dataloader_builder
        self._has_switched = False

    def before_epoch(self):
        if self.by == "epoch" and self.loop.epoch >= self.switch_step and not self._has_switched:
            self.logger.info("Switch dataloader now!")
            self.loop.backend.reset_dataloader(self.dataloader_builder)
            self.loop.max_step_in_epoch = compute_max_step_in_epoch(self.loop.backend.dataloader, self.loop.backend.gradient_accumulation_steps)
            self._has_switched = True

    def before_step(self):
        if self.by == "iteration" and self.loop.mode == "iteration" and self.loop.iter >= self.switch_step and not self._has_switched:
            self.logger.info("Switch dataloader now!")
            self.loop.backend.reset_dataloader(self.dataloader_builder)
            self._has_switched = True
