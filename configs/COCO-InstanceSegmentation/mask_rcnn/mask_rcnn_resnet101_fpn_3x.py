from .mask_rcnn_resnet50_fpn_3x import dataloader
from .mask_rcnn_resnet50_fpn_3x import lr_multiplier
from .mask_rcnn_resnet50_fpn_3x import model
from .mask_rcnn_resnet50_fpn_3x import optimizer
from .mask_rcnn_resnet50_fpn_3x import train

model.backbone.name = "ResNet-101"

train.init_checkpoint = "torchvision://ImageNet/ResNet-101"
train.output_dir = "./output/mask_rcnn_resnet101_fpn_3x_coco_instance_segmentation"
