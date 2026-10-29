import time
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel

from .base import Backend
from .native_backend import NativeBackend

__all__ = ["MultiOptimizerNativeBackend"]


class MultiOptimizerNativeBackend(Backend):
    """Native training with disjoint optimizers and one shared AMP/accumulation policy."""

    def __init__(self, model, optimizers, dataloader, ddp=None, amp=None, gradient_clipping=None, gradient_accumulation_steps=1):
        # Backend.__init__ assumes a single optimizer; keep the plural API explicit.
        self.model = model
        self.optimizers = dict(optimizers)
        self.dataloader = dataloader
        self._dataloader_iter_obj = None
        seen = set()
        for name, optimizer in self.optimizers.items():
            parameters = {id(p) for group in optimizer.param_groups for p in group["params"]}
            if seen & parameters:
                raise ValueError(f"Optimizer {name} shares parameters with another optimizer")
            seen.update(parameters)
            optimizer.zero_grad(set_to_none=True)
        self.setup(ddp, amp, gradient_clipping, gradient_accumulation_steps)

    def setup(self, ddp=None, amp=None, gradient_clipping=None, gradient_accumulation_steps=1):
        self.model = NativeBackend.create_ddp_model(self.model, **(ddp or {}))
        amp = amp or {}
        self.device_type = next(self.model.parameters()).device.type
        self.enable_amp = amp.get("enable", False)
        self.amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[amp.get("mixed_precision", "fp16")]
        self.grad_scaler = torch.amp.GradScaler(
            device=self.device_type,
            enabled=self.enable_amp and self.amp_dtype == torch.float16,
            init_scale=amp.get("init_scale", 65536),
            growth_factor=amp.get("growth_factor", 2.0),
            backoff_factor=amp.get("backoff_factor", 0.5),
            growth_interval=amp.get("growth_interval", 2000),
        )
        self.gradient_clipping = gradient_clipping or {}
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def run_step(self):
        if not self.model.training:
            raise RuntimeError("[MultiOptimizerNativeBackend] model was changed to eval mode!")
        data_time = 0
        metrics = {}
        for i in range(self.gradient_accumulation_steps):
            start = time.perf_counter()
            data = next(self._dataloader_iter)
            data_time += time.perf_counter() - start
            context = nullcontext()
            if i + 1 != self.gradient_accumulation_steps and isinstance(self.model, DistributedDataParallel):
                context = self.model.no_sync()
            with context:
                with torch.autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.enable_amp):
                    loss_dict = self.model(data)
                    loss = sum(value for key, value in loss_dict.items() if "loss" in key and "[metric]" not in key)
                for key, value in loss_dict.items():
                    value = value.detach() if isinstance(value, torch.Tensor) else value
                    metrics[key] = metrics.get(key, 0) + value / self.gradient_accumulation_steps
                self.grad_scaler.scale(loss / self.gradient_accumulation_steps).backward()

        for name, optimizer in self.optimizers.items():
            clipping = self.gradient_clipping.get(name, {})
            if clipping.get("enable", False):
                self.grad_scaler.unscale_(optimizer)
                parameters = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
                clip_type = clipping.get("type", "norm")
                if clip_type == "norm":
                    metrics[f"{name}/[metric]grad_norm"] = torch.nn.utils.clip_grad_norm_(
                        parameters, clipping.get("value", 1.0), norm_type=clipping.get("norm", 2.0)
                    )
                elif clip_type == "value":
                    torch.nn.utils.clip_grad_value_(parameters, clipping.get("value", 1.0))
                else:
                    raise ValueError(f"Unknown gradient clipping type: {clip_type}")
            self.grad_scaler.step(optimizer)
        self.grad_scaler.update()
        if self.grad_scaler.is_enabled():
            metrics["[metric]grad_scaler"] = self.grad_scaler.get_scale()
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        self._write_metrics(metrics, data_time)

    def state_dict(self):
        state = {"optimizers": {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}}
        if self.grad_scaler.is_enabled():
            state["grad_scaler"] = self.grad_scaler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state_dict["optimizers"][name])
        if self.grad_scaler.is_enabled() and "grad_scaler" in state_dict:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
