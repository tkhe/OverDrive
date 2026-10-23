import torch
from overdrive.config import LazyCall as L
from overdrive.solver import get_default_optimizer_params

AdamW = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        base_lr="${..lr}",
        lr_factor_func=lambda name: 0.1 if "backbone" in name else 1,
    ),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
)
