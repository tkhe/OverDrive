from overdrive.model_zoo import get_config

from .solo_resnet50_fpn_1x import dataloader, model, optimizer, train

dataloader.train.mapper.augmentations[0].short_edge_length = (640, 672, 704, 736, 768, 800)
lr_multiplier = get_config("common/scheduler/coco_scheduler.py").lr_multiplier_3x
train.max_epoch = 36
train.output_dir = "./output/solo_resnet50_fpn_3x_coco_instance_segmentation"
