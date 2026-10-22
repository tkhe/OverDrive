from .fcos_resnet50_fpn_3x import dataloader
from .fcos_resnet50_fpn_3x import lr_multiplier
from .fcos_resnet50_fpn_3x import model
from .fcos_resnet50_fpn_3x import optimizer
from .fcos_resnet50_fpn_3x import train

model.backbone.name = "ResNet-101"

train.init_checkpoint = "torchvision://ImageNet/ResNet-101"
train.output_dir = "./output/fcos_resnet101_fpn_3x_coco_detection"
