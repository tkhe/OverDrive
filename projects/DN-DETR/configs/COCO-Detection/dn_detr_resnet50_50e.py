from dn_detr.model_zoo import get_config

model = get_config("common/models/dn_detr.py").model

dataloader = get_config("common/data/coco_detection.py").dataloader

optimizer = get_config("common/optim.py").AdamW

lr_multiplier = get_config("common/scheduler/coco_scheduler.py").lr_multiplier_50e

train = get_config("common/train.py").train
