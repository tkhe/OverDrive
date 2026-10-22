from overdrive.config import LazyCall as L
from overdrive.layers import ShapeSpec
from overdrive.model_zoo import get_config
from overdrive.modeling.backbone import ResNet
from overdrive.modeling.dense_head import DABDETRHead
from overdrive.modeling.loss import GIoULoss
from overdrive.modeling.loss import L1Loss
from overdrive.modeling.loss import SigmoidFocalLoss
from overdrive.modeling.matcher import FocalLossCost
from overdrive.modeling.matcher import GIoUCost
from overdrive.modeling.matcher import HungarianMatcher
from overdrive.modeling.matcher import L1Cost
from overdrive.modeling.meta_arch import DABDETR
from overdrive.modeling.neck import ChannelMapper
from overdrive.modeling.transformer import DABDETRTransformer
from overdrive.modeling.transformer import DABDETRTransformerDecoder
from overdrive.modeling.transformer import DABDETRTransformerEncoder
from overdrive.modeling.transformer import SinePositionEmbedding

constants = get_config("common/data/constants.py").constants

model = L(DABDETR)(
    backbone=L(ResNet.build)(
        name="ResNet-50",
        out_features=["res5"],
        freeze_at=1,
        norm="FrozenBN",
    ),
    neck=L(ChannelMapper)(
        input_shape={"res5": ShapeSpec(channels=2048, stride=32)},
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
    transformer=L(DABDETRTransformer)(
        in_features=["p5"],
        encoder=L(DABDETRTransformerEncoder)(
            num_layers=6,
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_channels=2048,
            ffn_dropout=0.0,
            activation="PReLU",
        ),
        decoder=L(DABDETRTransformerDecoder)(
            num_layers=6,
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_channels=2048,
            ffn_dropout=0.0,
            activation="PReLU",
        ),
    ),
    box_head=L(DABDETRHead)(
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
        max_detections_per_image="${..num_queries}",
    ),
    num_queries=300,
    freeze_anchor_box_center=True,
    pixel_mean=constants.imagenet_rgb256_mean,
    pixel_std=constants.imagenet_rgb256_std,
)
