from overdrive.config import LazyCall as L
from overdrive.engine.hook import BestCheckpointHook

train = dict(
    backend="native",
    mode="epoch",
    max_epoch=12,
    gradient_accumulation_steps=1,
    gradient_clipping=dict(
        enable=True,
        type="norm",
        norm=2,
        value=0.1,
    ),
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
            mode="max",
        ),
    ],
    output_dir="./output/dino_4scale_resnet50_12e_coco_detection",
    init_checkpoint="torchvision://ImageNet/ResNet-50",
)
