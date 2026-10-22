from overdrive.model_zoo import get_config

model = get_config("common/models/panoptic_fpn.py").model

dataloader = get_config("common/data/coco_panoptic_segmentation.py").dataloader

optimizer = get_config("common/optim.py").SGD

lr_multiplier = get_config("common/scheduler/coco_scheduler.py").lr_multiplier_1x

train = get_config("common/train.py").train
train.max_epoch = 12
train.custom_hooks[0].val_metric = "panoptic_seg/PQ"
train.init_checkpoint = "torchvision://ImageNet/ResNet-50"
train.output_dir = "./output/panoptic_fpn_resnet50_fpn_1x_coco_panoptic_segmentation"
