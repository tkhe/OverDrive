from overdrive.config import LazyCall as L
from overdrive.modeling.dense_head import SOLOV2Head, SOLOV2MaskFeatHead
from overdrive.modeling.meta_arch import SOLOv2

from .solo import model

model._target_ = SOLOv2
model.mask_head._target_ = SOLOV2Head
model.mask_head.pop("cls_down_index")
model.mask_head.feat_channels = 512
model.mask_head.stacked_convs = 4
model.mask_head.dynamic_conv_size = 1
model.mask_head.mask_feature_head = L(SOLOV2MaskFeatHead)(
    input_shape="${..input_shape}",
    in_features=["p2", "p3", "p4", "p5"],
    feat_channels=128,
    out_channels=256,
    norm="GN",
)
