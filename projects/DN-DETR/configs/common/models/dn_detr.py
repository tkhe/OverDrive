from overdrive.config import LazyCall as L
from overdrive.layers import ShapeSpec
from overdrive.modeling.backbone import ResNet
from overdrive.modeling.loss import GIoULoss
from overdrive.modeling.loss import L1Loss
from overdrive.modeling.loss import SigmoidFocalLoss
from overdrive.modeling.matcher import FocalLossCost
from overdrive.modeling.matcher import GIoUCost
from overdrive.modeling.matcher import HungarianMatcher
from overdrive.modeling.matcher import L1Cost
from overdrive.modeling.neck import ChannelMapper
from overdrive.modeling.transformer import SinePositionEmbedding

from dn_detr.model_zoo import get_config
from dn_detr.modeling.denoising import DNQueryGenerator
from dn_detr.modeling.dense_head import DNDETRHead
from dn_detr.modeling.meta_arch import DNDETR
from dn_detr.modeling.transformer import DNDETRTransformer
from dn_detr.modeling.transformer import DNDETRTransformerDecoder
from dn_detr.modeling.transformer import DNDETRTransformerEncoder

constants = get_config("common/data/constants.py").constants

model = L(DNDETR)(
    backbone=L(ResNet.build)(
        name="ResNet-50",
        out_features=["res5"],
        freeze_at=1,
        norm="FrozenBN",
    ),
    neck=L(ChannelMapper)(
        input_shape={
            "res5": ShapeSpec(channels=2048, stride=32),
        },
        in_features="${..backbone.out_features}",
        out_channels=256,
        kernel_size=1,
        norm="",
        activation="",
    ),
    position_embedding=L(SinePositionEmbedding)(
        embed_dim=256,
        temperature=20,
        normalize=True,
    ),
    transformer=L(DNDETRTransformer)(
        in_features=["p5"],
        encoder=L(DNDETRTransformerEncoder)(
            num_layers=6,
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_channels=2048,
            ffn_dropout=0.0,
        ),
        decoder=L(DNDETRTransformerDecoder)(
            num_layers=6,
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_channels=2048,
            ffn_dropout=0.0,
        ),
    ),
    box_head=L(DNDETRHead)(
        num_classes=80,
        embed_dim=256,
        matcher=L(HungarianMatcher)(
            match_costs=[
                L(FocalLossCost)(weight=2.0),
                L(GIoUCost)(weight=2.0),
                L(L1Cost)(weight=5.0),
            ],
        ),
        loss_cls=L(SigmoidFocalLoss)(
            reduction="sum",
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
        ),
        loss_iou=L(GIoULoss)(
            reduction="sum",
            loss_weight=2.0,
        ),
        loss_reg=L(L1Loss)(
            reduction="sum",
            loss_weight=5.0,
        ),
        use_auxiliary_loss=True,
        test_score_threshold=0.0,
        max_detections_per_image="${..num_queries}",
    ),
    denoising_generator=L(DNQueryGenerator)(
        num_queries="${..num_queries}",
        num_classes="${..box_head.num_classes}",
        embed_dim="${..box_head.embed_dim}",
        denoising_groups=5,
        label_noise_prob=0.2,
        box_noise_scale=0.4,
        with_indicator=True,
    ),
    num_queries=300,
    freeze_anchor_box_center=False,
    pixel_mean=constants.imagenet_rgb256_mean,
    pixel_std=constants.imagenet_rgb256_std,
)
