from overdrive.config import LazyCall as L
from overdrive.solver import CompositeParamScheduler
from overdrive.solver import ConstantParamScheduler
from overdrive.solver import CosineParamScheduler
from overdrive.solver import LinearParamScheduler

lr_multiplier_140e = L(CompositeParamScheduler)(
    schedulers=[
        L(LinearParamScheduler)(
            start_value=0.001,
            end_value=1.0,
            begin=0,
            end=1,
            by="epoch",
        ),
        L(ConstantParamScheduler)(
            value=1.0,
            begin=1,
            end=70,
            by="epoch",
        ),
        L(CosineParamScheduler)(
            start_value=1.0,
            end_value=1e-4,
            begin=70,
            end=140,
            by="epoch",
        ),
    ],
)
