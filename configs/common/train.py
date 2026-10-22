from overdrive.config import LazyCall as L
from overdrive.engine.hook import BestCheckpointHook


train = dict(
    output_dir="./output",
    init_checkpoint="",
    backend="native",
    mode="epoch",
    max_epoch=12,
    gradient_accumulation_steps=1,
    amp=dict(enable=False),
    ddp=dict(
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    checkpointer=dict(
        period=1,
        max_to_keep=1,
    ),
    eval_period=1,
    log_period=10,
    device="cuda",
    custom_hooks=[
        L(BestCheckpointHook)(
            eval_period="${...eval_period}",
            val_metric="bbox/AP",
        ),
    ],
)
