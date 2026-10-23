from overdrive.config import LazyCall as L
from overdrive.solver import MultiStepParamScheduler

lr_multiplier_12e = L(MultiStepParamScheduler)(
    values=[1.0, 0.1],
    milestones=[11],
    begin=0,
    end=(12, "epoch"),
    by="epoch",
)

lr_multiplier_36e = L(MultiStepParamScheduler)(
    values=[1.0, 0.1],
    milestones=[30],
    begin=0,
    end=(36, "epoch"),
    by="epoch",
)
