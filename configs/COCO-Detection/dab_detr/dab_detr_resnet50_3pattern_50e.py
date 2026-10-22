from .dab_detr_resnet50_50e import dataloader
from .dab_detr_resnet50_50e import lr_multiplier
from .dab_detr_resnet50_50e import model
from .dab_detr_resnet50_50e import optimizer
from .dab_detr_resnet50_50e import train

model.transformer.num_patterns = 3

train.output_dir = "./output/dab_detr_resnet50_3patterns_50e_coco_detection"
