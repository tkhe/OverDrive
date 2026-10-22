from overdrive.config import LazyCall as L
from overdrive.solver.param_scheduler import MultiStepParamScheduler
from overdrive.solver.param_scheduler import WarmupParamScheduler


def default_X_scheduler(num_X):
    num_epochs = num_X * 12

    return L(WarmupParamScheduler)(
        scheduler=L(MultiStepParamScheduler)(
            values=[1.0, 0.1, 0.01],
            milestones=[num_X * 8, num_X * 11],
            begin=(500, "iteration"),
            end=(num_epochs, "epoch"),
            by="epoch",
        ),
        warmup_length=500,
        warmup_by="iteration",
        warmup_factor=0.001,
        warmup_method="linear",
    )


lr_multiplier_1x = default_X_scheduler(1)
lr_multiplier_2x = default_X_scheduler(2)
lr_multiplier_3x = default_X_scheduler(3)
lr_multiplier_6x = default_X_scheduler(6)
lr_multiplier_9x = default_X_scheduler(9)
