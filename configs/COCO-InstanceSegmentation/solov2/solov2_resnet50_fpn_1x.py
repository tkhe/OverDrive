from overdrive.model_zoo import get_config

model = get_config("common/models/solov2.py").model
dataloader = get_config("common/data/coco_instance_segmentation.py").dataloader
dataloader.train.mapper.mask_format = "bitmask"
dataloader.train.mapper.augmentations[0].short_edge_length = 800

optimizer = get_config("common/optim.py").SGD
optimizer.lr = 0.01
lr_multiplier = get_config("common/scheduler/coco_scheduler.py").lr_multiplier_1x

train = get_config("common/train.py").train
train.custom_hooks[0].val_metric = "segm/AP"
train.gradient_clipping = dict(enable=True, type="norm", value=35.0, norm=2.0)
train.init_checkpoint = "torchvision://ImageNet/ResNet-50"
train.output_dir = "./output/solov2_resnet50_fpn_1x_coco_instance_segmentation"
