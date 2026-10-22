from .centernet_resnet18_140e import dataloader
from .centernet_resnet18_140e import lr_multiplier
from .centernet_resnet18_140e import model
from .centernet_resnet18_140e import optimizer
from .centernet_resnet18_140e import train

model.neck.conv_op = "DCNv2"

train.output_dir = "./output/centernet_resnet18_dcnv2_140e_coco_detection"
