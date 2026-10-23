from overdrive.config import LazyCall as L
from overdrive.solver import MultiStepParamScheduler

lr_multiplier_50e = L(MultiStepParamScheduler)(
    values=[1.0, 0.1],
    milestones=[40],
    begin=0,
    end=(50, "epoch"),
    by="epoch",
)
