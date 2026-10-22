from overdrive.config import LazyCall as L
from overdrive.layers import ShapeSpec
from overdrive.model_zoo import get_config
from overdrive.modeling.backbone import ResNet
from overdrive.modeling.backbone.resnet import Bottleneck
from overdrive.modeling.backbone.resnet import DeepStem
from overdrive.modeling.loss import CrossEntropyLoss
from overdrive.modeling.loss import DiceLoss
from overdrive.modeling.meta_arch import UPerNet
from overdrive.modeling.sem_seg_head import UPerNetHead

constants = get_config("common/data/constants.py").constants

model = L(UPerNet)(
    backbone=L(ResNet)(
        stem_class=DeepStem,
        block_class=Bottleneck,
        num_blocks=[3, 4, 6, 3],
        out_channels_per_stage=[256, 512, 1024, 2048],
        stride_in_1x1=False,
        out_features=["res2", "res3", "res4", "res5"],
    ),
    sem_seg_head=L(UPerNetHead)(
        input_shape={
            "res2": ShapeSpec(channels=256, stride=4),
            "res3": ShapeSpec(channels=512, stride=8),
            "res4": ShapeSpec(channels=1024, stride=16),
            "res5": ShapeSpec(channels=2048, stride=32),
        },
        in_features=["res2", "res3", "res4", "res5"],
        num_classes=150,
        channels=512,
        pool_scales=[1, 2, 3, 6],
        dropout=0.1,
        ignore_index=255,
        norm="SyncBN",
        loss_sem_seg={
            "loss_ce": CrossEntropyLoss(
                loss_weight=1.0,
                reduction="mean",
                ignore_index=255,
            ),
            "loss_dice": DiceLoss(
                loss_weight=3.0,
                reduction="mean",
                ignore_index=255,
            ),
        },
        align_corners=False,
    ),
    pixel_mean=constants.imagenet_rgb256_mean,
    pixel_std=constants.imagenet_rgb256_std,
)
