from overdrive.layers import ShapeSpec

from .panoptic_fpn_resnet50_fpn_3x import dataloader
from .panoptic_fpn_resnet50_fpn_3x import lr_multiplier
from .panoptic_fpn_resnet50_fpn_3x import model
from .panoptic_fpn_resnet50_fpn_3x import optimizer
from .panoptic_fpn_resnet50_fpn_3x import train

model.backbone.name = "ResNet-18"
model.neck.input_shape = {
    "res2": ShapeSpec(channels=64, stride=4),
    "res3": ShapeSpec(channels=128, stride=8),
    "res4": ShapeSpec(channels=256, stride=16),
    "res5": ShapeSpec(channels=512, stride=32),
}

train.init_checkpoint = "torchvision://ImageNet/ResNet-18"
train.output_dir = "./output/panoptic_fpn_resnet18_fpn_3x_coco_panoptic_segmentation"
