from .panoptic_fpn_resnet50_fpn_3x import dataloader
from .panoptic_fpn_resnet50_fpn_3x import lr_multiplier
from .panoptic_fpn_resnet50_fpn_3x import model
from .panoptic_fpn_resnet50_fpn_3x import optimizer
from .panoptic_fpn_resnet50_fpn_3x import train

model.backbone.name = "ResNet-101"

train.init_checkpoint = "torchvision://ImageNet/ResNet-101"
train.output_dir = "./output/panoptic_fpn_resnet101_fpn_3x_coco_panoptic_segmentation"
