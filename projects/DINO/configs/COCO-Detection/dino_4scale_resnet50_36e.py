from dino.model_zoo import get_config

model = get_config("common/models/dino.py").model
dataloader = get_config("common/data/coco_detection.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/scheduler/coco_scheduler.py").lr_multiplier_36e
train = get_config("common/train.py").train
train.max_epoch = 36
train.output_dir = "./output/dino_4scale_resnet50_36e_coco_detection"
