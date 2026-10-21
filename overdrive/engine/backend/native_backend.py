import time
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel

from overdrive.config import try_get_key
from overdrive.utils import comm

from .base import Backend

__all__ = ["NativeBackend"]


class NativeBackend(Backend):
    def setup(
        self,
        ddp: Mapping[str, Any] = None,
        amp: Mapping[str, Any] = None,
        gradient_clipping: Mapping[str, Any] = None,
        gradient_accumulation_steps: int = 1,
    ):
        # DDP
        self.model = NativeBackend.create_ddp_model(self.model, **ddp)

        # AMP
        self.enable_amp = try_get_key(amp, "enable", default=False)
        self.amp_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[try_get_key(amp, "mixed_precision", default="fp16")]
        enable_grad_scaler = self.enable_amp and self.amp_dtype == torch.float16
        self.grad_scaler = torch.amp.GradScaler(
            device="cuda",
            enabled=enable_grad_scaler,
            init_scale=try_get_key(amp, "init_scale", default=65536),
            growth_factor=try_get_key(amp, "growth_factor", default=2.0),
            backoff_factor=try_get_key(amp, "backoff_factor", default=0.5),
            growth_interval=try_get_key(amp, "growth_interval", default=2000),
        )

        # gradient clipping
        self.enable_gradient_clipping = try_get_key(gradient_clipping, "enable", default=False)
        self.gradient_clipping_type = try_get_key(gradient_clipping, "type", default="norm")
        self.gradient_clipping_value = try_get_key(gradient_clipping, "value", default=1.0)
        self.gradient_clipping_norm = try_get_key(gradient_clipping, "norm", default=2.0)

        # gradient accumulation
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def run_step(self):
        if not self.model.training:
            raise RuntimeError("[NativeBackend] model was changed to eval mode!")

        data_time = 0
        accumulated_loss_dict = {}

        for i in range(self.gradient_accumulation_steps):
            start = time.perf_counter()
            data = next(self._dataloader_iter)
            data_time += time.perf_counter() - start

            context = nullcontext()
            if i + 1 != self.gradient_accumulation_steps and isinstance(self.model, DistributedDataParallel):
                context = self.model.no_sync()

            with context:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
                    loss_dict = self.model(data)

                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                else:
                    losses = sum([v for k, v in loss_dict.items() if "loss" in k and "[metric]" not in k])

                for key, value in loss_dict.items():
                    value = value.detach() if isinstance(value, torch.Tensor) else value
                    accumulated_loss_dict[key] = accumulated_loss_dict.get(key, 0) + value / self.gradient_accumulation_steps

                scaled_loss = losses / self.gradient_accumulation_steps
                self.grad_scaler.scale(scaled_loss).backward()

        loss_dict = accumulated_loss_dict

        if self.enable_gradient_clipping:
            self.grad_scaler.unscale_(self.optimizer)
            parameters = (p for p in self.model.parameters() if p.requires_grad and p.grad is not None)
            if self.gradient_clipping_type == "norm":
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clipping_value, norm_type=self.gradient_clipping_norm)
                loss_dict["[metric]grad_norm"] = grad_norm
            elif self.gradient_clipping_type == "value":
                torch.nn.utils.clip_grad_value_(parameters, self.gradient_clipping_value)
            else:
                raise ValueError(f"Unknown gradient clipping type: {self.gradient_clipping_type}")

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        if self.grad_scaler.is_enabled():
            loss_dict["[metric]grad_scaler"] = self.grad_scaler.get_scale()
        self.optimizer.zero_grad(set_to_none=True)

        self._write_metrics(loss_dict, data_time)

    def state_dict(self):
        state_dict = super().state_dict()
        if self.grad_scaler.is_enabled():
            state_dict["grad_scaler"] = self.grad_scaler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.grad_scaler.is_enabled() and "grad_scaler" in state_dict:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])

    @staticmethod
    def create_ddp_model(model, *, fp16_compression=False, **kwargs):
        if comm.get_world_size() == 1:
            return model

        if "device_ids" not in kwargs:
            kwargs["device_ids"] = [comm.get_local_rank()]

        ddp = DistributedDataParallel(model, **kwargs)
        if fp16_compression:
            from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

            ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

        return ddp
