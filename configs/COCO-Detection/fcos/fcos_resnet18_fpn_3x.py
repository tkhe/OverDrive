from overdrive.layers import ShapeSpec
from overdrive.model_zoo import get_config

model = get_config("common/models/fcos.py").model
model.backbone.name = "ResNet-18"
model.neck.input_shape = {
    "res3": ShapeSpec(channels=128, stride=8),
    "res4": ShapeSpec(channels=256, stride=16),
    "res5": ShapeSpec(channels=512, stride=32),
}

dataloader = get_config("common/data/coco_detection.py").dataloader

optimizer = get_config("common/optim.py").SGD
optimizer.lr = 0.01

lr_multiplier = get_config("common/scheduler/coco_scheduler.py").lr_multiplier_3x

train = get_config("common/train.py").train
train.max_epoch = 36
train.init_checkpoint = "torchvision://ImageNet/ResNet-18"
train.output_dir = "./output/fcos_resnet18_fpn_3x_coco_detection"
