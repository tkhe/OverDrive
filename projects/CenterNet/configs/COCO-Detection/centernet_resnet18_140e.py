from centernet.model_zoo import get_config

model = get_config("common/models/centernet.py").model

dataloader = get_config("common/data/coco_detection.py").dataloader

optimizer = get_config("common/optim.py").SGD

lr_multiplier = get_config("common/scheduler/coco_scheduler.py").lr_multiplier_140e

train = get_config("common/train.py").train
train.output_dir = "./output/centernet_resnet18_140e_coco_detection"
