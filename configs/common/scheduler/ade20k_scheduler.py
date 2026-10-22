from overdrive.config import LazyCall as L
from overdrive.solver.param_scheduler import PolynomialDecayParamScheduler
from overdrive.solver.param_scheduler import WarmupParamScheduler


def default_X_scheduler(total_steps):
    return L(WarmupParamScheduler)(
        scheduler=L(PolynomialDecayParamScheduler)(
            base_value=1.0,
            power=0.9,
            begin=(500, "iteration"),
            end=(total_steps, "iteration"),
        ),
        warmup_length=500,
        warmup_by="iteration",
        warmup_factor=0.001,
        warmup_method="linear",
    )


lr_multiplier_20k = default_X_scheduler(20000)
lr_multiplier_40k = default_X_scheduler(40000)
lr_multiplier_80k = default_X_scheduler(80000)
lr_multiplier_160k = default_X_scheduler(160000)
lr_multiplier_240k = default_X_scheduler(240000)
lr_multiplier_320k = default_X_scheduler(320000)
