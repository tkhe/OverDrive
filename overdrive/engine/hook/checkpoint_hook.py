from overdrive.checkpoint import Checkpointer

from .base import Hook
from .priority import HookPriority

__all__ = ["CheckpointHook"]


class CheckpointHook(Hook):
    def __init__(
        self,
        checkpointer: Checkpointer,
        period: int,
        max_to_keep: int | None = None,
        file_prefix: str = "model",
    ):
        if max_to_keep is not None and max_to_keep <= 0:
            raise ValueError(f"Expected max_to_keep > 0, but got max_to_keep={max_to_keep}")

        self.checkpointer = checkpointer
        self.period = period
        self.max_to_keep = max_to_keep
        self.recent_checkpoints: list[str] = []
        self.file_prefix = file_prefix
        self.path_manager = self.checkpointer.path_manager

    def after_step(self):
        if self.loop.mode == "iteration":
            iteration = self.loop.iter

            if (iteration + 1) % self.period == 0:
                self.checkpointer.save(f"{self.file_prefix}_{iteration + 1:07d}")

                if self.max_to_keep is not None:
                    self.recent_checkpoints.append(self.checkpointer.get_checkpoint_file())
                    if len(self.recent_checkpoints) > self.max_to_keep:
                        file_to_delete = self.recent_checkpoints.pop(0)
                        if self.path_manager.exists(file_to_delete) and not file_to_delete.endswith(f"{self.file_prefix}_final.pth"):
                            self.path_manager.rm(file_to_delete)

    def after_epoch(self):
        epoch = self.loop.epoch

        if (epoch + 1) % self.period == 0:
            self.checkpointer.save(f"{self.file_prefix}_{epoch + 1:07d}")

            if self.max_to_keep is not None:
                self.recent_checkpoints.append(self.checkpointer.get_checkpoint_file())
                if len(self.recent_checkpoints) > self.max_to_keep:
                    file_to_delete = self.recent_checkpoints.pop(0)
                    if self.path_manager.exists(file_to_delete) and not file_to_delete.endswith(f"{self.file_prefix}_final.pth"):
                        self.path_manager.rm(file_to_delete)

    def after_train(self):
        if self.loop.mode == "epoch":
            if self.loop.epoch == self.loop.max_epoch:
                self.checkpointer.save(f"{self.file_prefix}_final")
        elif self.loop.mode == "iteration":
            if self.loop.iter == self.loop.max_iter:
                self.checkpointer.save(f"{self.file_prefix}_final")
        else:
            raise NotImplementedError(f"Unknown loop mode: {self.loop.mode}")

    @property
    def priority(self):
        return HookPriority.LOW
