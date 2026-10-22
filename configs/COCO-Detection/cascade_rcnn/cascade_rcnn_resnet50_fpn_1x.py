from overdrive.model_zoo import get_config

model = get_config("common/models/cascade_rcnn.py").model

dataloader = get_config("common/data/coco_detection.py").dataloader

optimizer = get_config("common/optim.py").SGD

lr_multiplier = get_config("common/scheduler/coco_scheduler.py").lr_multiplier_1x

train = get_config("common/train.py").train
train.init_checkpoint = "torchvision://ImageNet/ResNet-50"
train.output_dir = "./output/cascade_rcnn_resnet50_fpn_1x_coco_detection"
