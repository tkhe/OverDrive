import logging

from overdrive.data.utils import compute_max_step_in_epoch
from overdrive.utils.events import EventStorage

from .base import Loop

__all__ = ["EpochLoop"]


class EpochLoop(Loop):
    mode: str = "epoch"

    def __init__(self, backend):
        super().__init__(backend)

        self.backend = backend

        self.start_epoch: int = 0
        self.epoch: int = 0
        self.max_epoch: int = 0
        self.step_in_epoch: int = 0
        self.max_step_in_epoch: int = 0

    def run(self, start_epoch: int, max_epoch: int):
        logger = logging.getLogger(__name__)
        logger.info(f"Starting training from epoch {start_epoch + 1}")

        self.start_epoch = start_epoch
        self.epoch = start_epoch
        self.max_epoch = max_epoch
        self.max_step_in_epoch = compute_max_step_in_epoch(self.backend.dataloader, self.backend.gradient_accumulation_steps)

        start_step = self.start_epoch * self.max_step_in_epoch

        with EventStorage(start_step) as self.storage:
            try:
                self.before_train()
                for epoch in range(self.start_epoch, self.max_epoch):
                    self.epoch = epoch
                    self.before_epoch()
                    for step_in_epoch in range(self.max_step_in_epoch):
                        self.step_in_epoch = step_in_epoch
                        self.before_step()
                        self.run_step()
                        self.after_step()
                        self.storage.step()
                    self.after_epoch()
                self.epoch += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def run_step(self):
        self.backend.run_step()

    def state_dict(self):
        ret = {"epoch": self.epoch}
        ret.update(super().state_dict())
        return ret

    def load_state_dict(self, state_dict):
        self.epoch = state_dict.pop("epoch", 0)
        self.start_epoch = self.epoch + 1
        super().load_state_dict(state_dict)
