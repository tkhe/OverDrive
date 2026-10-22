from .coco_detection import dataloader

dataloader.train.mapper.use_mask = True
dataloader.train.mapper.recompute_boxes = True
