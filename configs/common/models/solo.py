from overdrive.config import LazyCall as L
from overdrive.layers import ShapeSpec
from overdrive.model_zoo import get_config
from overdrive.modeling.backbone import ResNet
from overdrive.modeling.dense_head import SOLOHead
from overdrive.modeling.loss import SigmoidDiceLoss, SigmoidFocalLoss
from overdrive.modeling.meta_arch import SOLO
from overdrive.modeling.neck import FPN, LastLevelMaxPool

constants = get_config("common/data/constants.py").constants

model = L(SOLO)(
    backbone=L(ResNet.build)(
        name="ResNet-50",
        stride_in_1x1=False,
        norm="FrozenBN",
        out_features=["res2", "res3", "res4", "res5"],
        freeze_at=2,
    ),
    neck=L(FPN)(
        input_shape={
            "res2": ShapeSpec(channels=256, stride=4),
            "res3": ShapeSpec(channels=512, stride=8),
            "res4": ShapeSpec(channels=1024, stride=16),
            "res5": ShapeSpec(channels=2048, stride=32),
        },
        in_features="${..backbone.out_features}",
        out_channels=256,
        top_block=L(LastLevelMaxPool)(in_feature="p5"),
    ),
    mask_head=L(SOLOHead)(
        input_shape={
            "p2": ShapeSpec(channels=256, stride=4),
            "p3": ShapeSpec(channels=256, stride=8),
            "p4": ShapeSpec(channels=256, stride=16),
            "p5": ShapeSpec(channels=256, stride=32),
            "p6": ShapeSpec(channels=256, stride=64),
        },
        in_features=["p2", "p3", "p4", "p5", "p6"],
        num_classes=80,
        feat_channels=256,
        stacked_convs=7,
        norm="GN",
        cls_down_index=0,
        num_grids=[40, 36, 24, 16, 12],
        scale_ranges=[(1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)],
        pos_scale=0.2,
        loss_cls=L(SigmoidFocalLoss)(alpha=0.25, gamma=2.0, reduction="sum"),
        loss_mask=L(SigmoidDiceLoss)(loss_weight=3.0, reduction="sum"),
        test_score_threshold=0.1,
        test_mask_threshold=0.5,
        test_topk_candidates=500,
        test_filter_threshold=0.05,
        test_nms_kernel="gaussian",
        test_nms_sigma=2.0,
        max_detections_per_image=100,
    ),
    pixel_mean=constants.imagenet_rgb256_mean,
    pixel_std=constants.imagenet_rgb256_std,
)
