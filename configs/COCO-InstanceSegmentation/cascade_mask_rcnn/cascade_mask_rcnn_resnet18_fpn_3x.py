from overdrive.layers import ShapeSpec
from overdrive.model_zoo import get_config

model = get_config("common/models/cascade_mask_rcnn.py").model
model.backbone.name = "ResNet-18"
model.neck.input_shape = {
    "res2": ShapeSpec(channels=64, stride=4),
    "res3": ShapeSpec(channels=128, stride=8),
    "res4": ShapeSpec(channels=256, stride=16),
    "res5": ShapeSpec(channels=512, stride=32),
}

dataloader = get_config("common/data/coco_instance_segmentation.py").dataloader

optimizer = get_config("common/optim.py").SGD

lr_multiplier = get_config("common/scheduler/coco_scheduler.py").lr_multiplier_3x

train = get_config("common/train.py").train
train.custom_hooks[0].val_metric = "segm/AP"
train.max_epoch = 36
train.init_checkpoint = "torchvision://ImageNet/ResNet-18"
train.output_dir = "./output/cascade_mask_rcnn_resnet18_fpn_3x_coco_instance_segmentation"
