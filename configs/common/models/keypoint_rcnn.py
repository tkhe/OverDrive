from overdrive.config import LazyCall as L
from overdrive.layers import ShapeSpec
from overdrive.model_zoo import get_config
from overdrive.modeling.anchor_generator import AnchorGenerator
from overdrive.modeling.backbone import ResNet
from overdrive.modeling.box_coder import DeltaXYWHBoxCoder
from overdrive.modeling.dense_head import RPNHead
from overdrive.modeling.loss import BCEWithLogitsLoss
from overdrive.modeling.loss import CrossEntropyLoss
from overdrive.modeling.loss import SmoothL1Loss
from overdrive.modeling.matcher import MaxIoUMatcher
from overdrive.modeling.meta_arch import KeypointRCNN
from overdrive.modeling.neck import FPN
from overdrive.modeling.neck.fpn import LastLevelMaxPool
from overdrive.modeling.pooler import ROIPooler
from overdrive.modeling.roi_head import StandardROIHead
from overdrive.modeling.roi_head.box_head import ConvFCBoxHead
from overdrive.modeling.roi_head.keypoint_head import ConvDeconvUpsampleKeypointHead
from overdrive.modeling.sampler import RandomSampler

constants = get_config("common/data/constants.py").constants

model = L(KeypointRCNN)(
    backbone=L(ResNet.build)(
        name="ResNet-50",
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
    rpn_head=L(RPNHead)(
        input_shape={
            "p2": ShapeSpec(channels=256, stride=4),
            "p3": ShapeSpec(channels=256, stride=8),
            "p4": ShapeSpec(channels=256, stride=16),
            "p5": ShapeSpec(channels=256, stride=32),
            "p6": ShapeSpec(channels=256, stride=64),
        },
        in_features=["p2", "p3", "p4", "p5", "p6"],
        conv_dims=[-1],
        anchor_generator=L(AnchorGenerator)(
            sizes=[[32], [64], [128], [256], [512]],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            offset=0.0,
        ),
        box_coder=L(DeltaXYWHBoxCoder)(weights=[1.0, 1.0, 1.0, 1.0]),
        matcher=L(MaxIoUMatcher)(
            thresholds=[0.3, 0.7],
            labels=[0, -1, 1],
            allow_low_quality_matches=True,
        ),
        sampler=L(RandomSampler)(
            num_samples=256,
            positive_fraction=0.5,
        ),
        loss_cls=L(BCEWithLogitsLoss)(
            reduction="sum",
            loss_weight=1.0,
        ),
        loss_reg=L(SmoothL1Loss)(
            beta=0,
            reduction="sum",
            loss_weight=1.0,
        ),
        reg_decoded_box=False,
        pre_nms_topk=(2000, 1000),
        post_nms_topk=(1000, 1000),
        nms_threshold=0.7,
    ),
    roi_head=L(StandardROIHead)(
        box_in_features=["p2", "p3", "p4", "p5"],
        box_pooler=L(ROIPooler)(
            output_size=7,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        box_head=L(ConvFCBoxHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[],
            fc_dims=[1024, 1024],
            num_classes="${..num_classes}",
            box_coder=L(DeltaXYWHBoxCoder)(weights=[10, 10, 5, 5]),
            loss_cls=L(CrossEntropyLoss)(
                loss_weight=1.0,
                reduction="mean",
            ),
            loss_reg=L(SmoothL1Loss)(
                beta=0,
                reduction="sum",
                loss_weight=1.0,
            ),
            reg_decoded_box=False,
            cls_agnostic_box_reg=False,
            test_score_threshold=0.05,
            test_nms_threshold=0.5,
            max_detections_per_image=100,
        ),
        keypoint_in_features=["p2", "p3", "p4", "p5"],
        keypoint_pooler=L(ROIPooler)(
            output_size=14,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        keypoint_head=L(ConvDeconvUpsampleKeypointHead)(
            input_shape=ShapeSpec(channels=256, height=14, width=14),
            conv_dims=[512 for _ in range(8)],
            num_keypoints=17,
            loss_keypoint=L(CrossEntropyLoss)(
                reduction="sum",
                loss_weight=1.0,
            ),
            loss_normalizer="visible",
        ),
        num_classes=1,
        train_on_pred_boxes=False,
        matcher=L(MaxIoUMatcher)(
            thresholds=[0.5],
            labels=[0, 1],
            allow_low_quality_matches=False,
        ),
        sampler=L(RandomSampler)(
            num_samples=512,
            positive_fraction=0.25,
        ),
        proposal_append_gt=True,
    ),
    pixel_mean=constants.imagenet_rgb256_mean,
    pixel_std=constants.imagenet_rgb256_std,
)
