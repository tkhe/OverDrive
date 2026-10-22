from overdrive.layers import ShapeSpec

from .autoassign_resnet50_fpn_3x import dataloader
from .autoassign_resnet50_fpn_3x import lr_multiplier
from .autoassign_resnet50_fpn_3x import model
from .autoassign_resnet50_fpn_3x import optimizer
from .autoassign_resnet50_fpn_3x import train

model.backbone.name = "ResNet-18"
model.neck.input_shape = {
    "res3": ShapeSpec(channels=128, stride=8),
    "res4": ShapeSpec(channels=256, stride=16),
    "res5": ShapeSpec(channels=512, stride=32),
}
model.neck.top_block.in_channels = 512

train.init_checkpoint = "torchvision://ImageNet/ResNet-18"
train.output_dir = "./output/autoassign_resnet18_fpn_3x_coco_detection"
