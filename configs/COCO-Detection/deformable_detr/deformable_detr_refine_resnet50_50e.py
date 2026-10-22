from .deformable_detr_resnet50_50e import dataloader
from .deformable_detr_resnet50_50e import lr_multiplier
from .deformable_detr_resnet50_50e import model
from .deformable_detr_resnet50_50e import optimizer
from .deformable_detr_resnet50_50e import train

model.with_box_refine = True
model.box_head.share_pred_layer = False

train.output_dir = "./output/deformable_detr_refine_resnet50_50e_coco_detection"
