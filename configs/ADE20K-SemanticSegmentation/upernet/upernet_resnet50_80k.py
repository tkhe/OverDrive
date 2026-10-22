from overdrive.model_zoo import get_config

model = get_config("common/models/upernet.py").model

dataloader = get_config("common/data/ade20k_semantic_segmentation.py").dataloader

optimizer = get_config("common/optim.py").SGD
optimizer.lr = 0.01
optimizer.weight_decay = 0.0005

lr_multiplier = get_config("common/scheduler/ade20k_scheduler.py").lr_multiplier_80k

train = get_config("common/train.py").train
train.custom_hooks[0].val_metric = "sem_seg/mIoU"
train.mode = "iteration"
train.max_iter = 80000
train.checkpointer.period = 8000
train.eval_period = 8000
train.init_checkpoint = "torchvision://ImageNet/ResNet-50"
train.output_dir = "./output/upernet_resnet50_80k_ade20k_sem_seg"
