import logging

from overdrive.utils.events import EventStorage

from .base import Loop

__all__ = ["IterationLoop"]


class IterationLoop(Loop):
    mode: str = "iteration"

    def __init__(self, backend):
        super().__init__(backend)

        self.start_iter: int = 0
        self.iter: int = 0
        self.max_iter: int = 0

    def run(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info(f"Starting training from iteration {start_iter + 1}")

        self.start_iter = start_iter
        self.iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                    self.storage.step()
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def run_step(self):
        self.backend.run_step()

    def state_dict(self):
        ret = {"iteration": self.iter}
        ret.update(super().state_dict())
        return ret

    def load_state_dict(self, state_dict):
        self.iter = state_dict.pop("iteration", 0)
        self.start_iter = self.iter + 1
        super().load_state_dict(state_dict)
