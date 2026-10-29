import weakref

from .checkpoint import Checkpointer

__all__ = ["DMLBestCheckpointer", "DMLLoopState"]


class DMLLoopState:
    """Persist the last completed index, including after Loop's final increment."""

    def __init__(self, loop):
        self.loop = weakref.proxy(loop)

    def state_dict(self):
        state = self.loop.state_dict()
        key, maximum = ("epoch", self.loop.max_epoch) if self.loop.mode == "epoch" else ("iteration", self.loop.max_iter)
        if maximum > 0:
            state[key] = min(state[key], maximum - 1)
        return state

    def load_state_dict(self, state_dict):
        self.loop.load_state_dict(state_dict)


class DMLBestCheckpointer(Checkpointer):
    """Save complete best-model snapshots without moving the resume marker."""

    def tag_last_checkpoint(self, last_filename_basename):
        pass
