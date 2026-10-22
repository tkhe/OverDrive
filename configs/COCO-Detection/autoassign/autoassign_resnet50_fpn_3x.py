from overdrive.model_zoo import get_config

from .autoassign_resnet50_fpn_1x import dataloader
from .autoassign_resnet50_fpn_1x import model
from .autoassign_resnet50_fpn_1x import optimizer
from .autoassign_resnet50_fpn_1x import train

lr_multiplier = get_config("common/scheduler/coco_scheduler.py").lr_multiplier_3x
lr_multiplier.warmup_length = 1000
lr_multiplier.scheduler.begin = (1000, "iteration")

train.max_epoch = 36
train.output_dir = "./output/autoassign_resnet50_fpn_3x_coco_detection"
