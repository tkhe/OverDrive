from overdrive.config import LazyCall as L
from overdrive.layers import ShapeSpec
from overdrive.model_zoo import get_config
from overdrive.modeling.backbone import ResNet
from overdrive.modeling.dense_head import DeformableDETRHead
from overdrive.modeling.loss import GIoULoss
from overdrive.modeling.loss import L1Loss
from overdrive.modeling.loss import SigmoidFocalLoss
from overdrive.modeling.matcher import FocalLossCost
from overdrive.modeling.matcher import GIoUCost
from overdrive.modeling.matcher import HungarianMatcher
from overdrive.modeling.matcher import L1Cost
from overdrive.modeling.meta_arch import DeformableDETR
from overdrive.modeling.neck import ChannelMapper
from overdrive.modeling.neck.channel_mapper import LastLevelConvs
from overdrive.modeling.transformer import DeformableDETRTransformer
from overdrive.modeling.transformer import DeformableDETRTransformerDecoder
from overdrive.modeling.transformer import DeformableDETRTransformerEncoder
from overdrive.modeling.transformer import SinePositionEmbedding

constants = get_config("common/data/constants.py").constants

model = L(DeformableDETR)(
    backbone=L(ResNet.build)(
        name="ResNet-50",
        out_features=["res3", "res4", "res5"],
        norm="FrozenBN",
        freeze_at=1,
    ),
    neck=L(ChannelMapper)(
        input_shape={
            "res3": ShapeSpec(channels=512, stride=8),
            "res4": ShapeSpec(channels=1024, stride=16),
            "res5": ShapeSpec(channels=2048, stride=32),
        },
        in_features="${..backbone.out_features}",
        out_channels=256,
        kernel_size=1,
        norm="GN",
        activation=None,
        top_block=L(LastLevelConvs)(
            in_channels=2048,
            out_channels="${..out_channels}",
            num_levels=1,
            norm="${..norm}",
            activation="${..activation}",
            in_feature="res5",
        ),
    ),
    position_embedding=L(SinePositionEmbedding)(
        embed_dim=256,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),
    transformer=L(DeformableDETRTransformer)(
        in_features=["p3", "p4", "p5", "p6"],
        encoder=L(DeformableDETRTransformerEncoder)(
            num_layers=6,
            embed_dim=256,
            num_heads=8,
            num_levels=4,
            num_points=4,
            attn_dropout=0.1,
            feedforward_channels=1024,
            ffn_dropout=0.1,
        ),
        decoder=L(DeformableDETRTransformerDecoder)(
            num_layers=6,
            embed_dim=256,
            num_heads=8,
            num_levels=4,
            num_points=4,
            attn_dropout=0.1,
            feedforward_channels=1024,
            ffn_dropout=0.1,
        ),
    ),
    box_head=L(DeformableDETRHead)(
        num_classes=80,
        embed_dim=256,
        num_decoder_layers=6,
        share_pred_layer=True,
        as_two_stage=False,
        matcher=L(HungarianMatcher)(
            match_costs=[
                L(FocalLossCost)(weight=2.0),
                L(GIoUCost)(weight=2.0),
                L(L1Cost)(weight=5.0, box_format="xywh"),
            ],
        ),
        loss_cls=L(SigmoidFocalLoss)(
            reduction="sum",
            alpha=0.25,
            gamma=2.0,
            loss_weight=2.0,
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
        max_detections_per_image=300,
    ),
    num_queries=300,
    num_feature_levels=4,
    with_box_refine=False,
    as_two_stage=False,
    pixel_mean=constants.imagenet_rgb256_mean,
    pixel_std=constants.imagenet_rgb256_std,
)
