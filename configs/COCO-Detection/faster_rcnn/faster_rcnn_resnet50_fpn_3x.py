from overdrive.model_zoo import get_config

from .faster_rcnn_resnet50_fpn_1x import dataloader
from .faster_rcnn_resnet50_fpn_1x import model
from .faster_rcnn_resnet50_fpn_1x import optimizer
from .faster_rcnn_resnet50_fpn_1x import train

lr_multiplier = get_config("common/scheduler/coco_scheduler.py").lr_multiplier_3x

train.max_epoch = 36
train.output_dir = "./output/faster_rcnn_resnet50_fpn_3x_coco_detection"
