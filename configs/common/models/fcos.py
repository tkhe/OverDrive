from overdrive.config import LazyCall as L
from overdrive.layers import ShapeSpec
from overdrive.model_zoo import get_config
from overdrive.modeling.backbone import ResNet
from overdrive.modeling.box_coder import DistancePointBoxCoder
from overdrive.modeling.dense_head.fcos_head import FCOSHead
from overdrive.modeling.loss import BCEWithLogitsLoss
from overdrive.modeling.loss import GIoULoss
from overdrive.modeling.loss import SigmoidFocalLoss
from overdrive.modeling.meta_arch import FCOS
from overdrive.modeling.neck import FPN
from overdrive.modeling.neck.fpn import LastLevelP6P7

constants = get_config("common/data/constants.py").constants

model = L(FCOS)(
    backbone=L(ResNet.build)(
        name="ResNet-50",
        stride_in_1x1=False,
        norm="FrozenBN",
        out_features=["res3", "res4", "res5"],
        freeze_at=2,
    ),
    neck=L(FPN)(
        input_shape={
            "res3": ShapeSpec(channels=512, stride=8),
            "res4": ShapeSpec(channels=1024, stride=16),
            "res5": ShapeSpec(channels=2048, stride=32),
        },
        in_features="${..backbone.out_features}",
        out_channels=256,
        top_block=L(LastLevelP6P7)(
            in_channels=256,
            out_channels="${..out_channels}",
            in_feature="p5",
        ),
    ),
    box_head=L(FCOSHead)(
        input_shape={
            "p3": ShapeSpec(channels=256, stride=8),
            "p4": ShapeSpec(channels=256, stride=16),
            "p5": ShapeSpec(channels=256, stride=32),
            "p6": ShapeSpec(channels=256, stride=64),
            "p7": ShapeSpec(channels=256, stride=128),
        },
        in_features=["p3", "p4", "p5", "p6", "p7"],
        num_classes=80,
        conv_dims=[256, 256, 256, 256],
        norm="GN",
        box_coder=L(DistancePointBoxCoder)(),
        regress_ranges=[(-1, 64), (64, 128), (128, 256), (256, 512), (512, 1e8)],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        loss_cls=L(SigmoidFocalLoss)(
            alpha=0.25,
            gamma=2.0,
            reduction="sum",
        ),
        loss_reg=L(GIoULoss)(
            reduction="none",
        ),
        loss_centerness=L(BCEWithLogitsLoss)(
            reduction="sum",
        ),
        test_score_threshold=0.05,
        test_topk_candidates=1000,
        test_nms_threshold=0.6,
        max_detections_per_image=100,
    ),
    pixel_mean=constants.imagenet_rgb256_mean,
    pixel_std=constants.imagenet_rgb256_std,
)
