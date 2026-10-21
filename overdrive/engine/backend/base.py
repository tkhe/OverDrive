import logging
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from overdrive.utils import comm
from overdrive.utils.events import get_event_storage

__all__ = ["Backend"]


class Backend(ABC):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        dataloader: Iterable,
        ddp: Mapping[str, Any] | None = None,
        amp: Mapping[str, Any] | None = None,
        gradient_clipping: Mapping[str, Any] | None = None,
        gradient_accumulation_steps: int = 1,
    ):
        ddp = ddp or {}
        amp = amp or {}
        gradient_clipping = gradient_clipping or {}

        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self._dataloader_iter_obj = None

        self.setup(
            ddp=ddp,
            amp=amp,
            gradient_clipping=gradient_clipping,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    @property
    def _dataloader_iter(self):
        if self._dataloader_iter_obj is None:
            self._dataloader_iter_obj = iter(self.dataloader)
        return self._dataloader_iter_obj

    def reset_dataloader(self, dataloader_builder):
        del self.dataloader
        dataloader = dataloader_builder()
        self.dataloader = dataloader
        self._dataloader_iter_obj = None

    @abstractmethod
    def setup(
        self,
        ddp: Mapping[str, Any] | None = None,
        amp: Mapping[str, Any] | None = None,
        gradient_clipping: Mapping[str, Any] | None = None,
        gradient_accumulation_steps: int = 1,
    ):
        raise NotImplementedError("setup() must be implemented in the subclass.")

    @abstractmethod
    def run_step(self):
        raise NotImplementedError("run_step() must be implemented in the subclass.")

    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ):
        logger = logging.getLogger(__name__)
        cur_step = get_event_storage().iter

        try:
            Backend.write_metrics(loss_dict, data_time, cur_step, prefix)
        except Exception:
            logger.exception("Exception during writing metrics:")
            raise

    @staticmethod
    def write_metrics(
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        cur_iter: int,
        prefix: str = "",
    ):
        metrics_dict = {k: (v.detach().cpu().item() if isinstance(v, torch.Tensor) else v) for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        storage = get_event_storage()

        all_metrics_dict = comm.all_gather(metrics_dict)

        if comm.is_main_process():
            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time, cur_iter=cur_iter)

            # average the rest metrics
            metrics_dict = {k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()}
            total_losses_reduced = sum([v for k, v in metrics_dict.items() if "loss" in k and "[metric]" not in k])
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(f"Loss became infinite or Nan at iteration={cur_iter}!\nloss_dict = {metrics_dict}")

            storage.put_scalar(f"{prefix}total_loss", total_losses_reduced, cur_iter=cur_iter)
            if len(metrics_dict) > 1:
                storage.put_scalars(cur_iter=cur_iter, **metrics_dict)

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer"])
