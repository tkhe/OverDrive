from overdrive.model_zoo import get_config

model = get_config("common/models/retinanet.py").model

dataloader = get_config("common/data/coco_detection.py").dataloader

optimizer = get_config("common/optim.py").SGD
optimizer.lr = 0.01

lr_multiplier = get_config("common/scheduler/coco_scheduler.py").lr_multiplier_1x

train = get_config("common/train.py").train
train.init_checkpoint = "torchvision://ImageNet/ResNet-50"
train.output_dir = "./output/retinanet_resnet50_fpn_1x_coco_detection"
