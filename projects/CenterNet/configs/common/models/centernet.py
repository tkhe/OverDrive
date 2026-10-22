from overdrive.config import LazyCall as L
from overdrive.layers import ShapeSpec
from overdrive.modeling.backbone import ResNet
from overdrive.modeling.loss import GaussianFocalLoss
from overdrive.modeling.loss import L1Loss

from centernet.model_zoo import get_config
from centernet.modeling.dense_head import CenterNetHead
from centernet.modeling.meta_arch import CenterNet
from centernet.modeling.neck import CenterNetResNetNeck

constants = get_config("common/data/constants.py").constants

model = L(CenterNet)(
    backbone=L(ResNet.build)(
        name="ResNet-18",
        out_features=["res5"],
    ),
    neck=L(CenterNetResNetNeck)(
        input_shape={"res5": ShapeSpec(channels=512, stride=32)},
        in_features="${..backbone.out_features}",
        num_deconv_kernel_sizes=[4, 4, 4],
        num_deconv_channels=[256, 128, 64],
        conv_op="Conv2d",
    ),
    box_head=L(CenterNetHead)(
        input_shape={"p2": ShapeSpec(channels=64, stride=4)},
        in_features=["p2"],
        num_classes=80,
        channels=64,
        loss_heatmap=L(GaussianFocalLoss)(
            reduction="sum",
            loss_weight=1.0,
        ),
        loss_wh=L(L1Loss)(
            reduction="sum",
            loss_weight=0.1,
        ),
        loss_offset=L(L1Loss)(
            reduction="sum",
            loss_weight=1.0,
        ),
        test_local_maximum_kernel=3,
        test_topk=100,
        test_score_threshold=0.05,
    ),
    pixel_mean=constants.imagenet_rgb256_mean,
    pixel_std=constants.imagenet_rgb256_std,
)
