import torch

from overdrive.config import LazyCall as L
from overdrive.data import transforms as T
from overdrive.model_zoo import get_config
from overdrive.solver import MultiStepParamScheduler
from overdrive.solver import WarmupParamScheduler
from overdrive.solver import get_default_optimizer_params

model = get_config("common/models/detr.py").model

dataloader = get_config("common/data/coco_detection.py").dataloader
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(prob=0.5),
    L(T.OneOf)(
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
            ),
            L(T.AugmentationList)(
                augs=[
                    L(T.ResizeShortestEdge)(short_edge_length=(400, 500, 600)),
                    L(T.RandomCrop)(
                        crop_type="absolute_range",
                        crop_size=(384, 600),
                    ),
                    L(T.ResizeShortestEdge)(
                        short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                        max_size=1333,
                        sample_style="choice",
                    ),
                ],
            ),
        ],
        prob=1.0,
    ),
]
dataloader.train.total_batch_size = 16
dataloader.train.num_workers = 8

optimizer = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        base_lr="${..lr}",
        weight_decay_norm=1e-4,
        lr_factor_func=lambda x: 0.1 if "backbone" in x else 1,
    ),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
)

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[334],
        begin=(500, "iteration"),
        end=(500, "epoch"),
        by="epoch",
    ),
    warmup_length=500,
    warmup_by="iteration",
    warmup_factor=0.001,
    warmup_method="linear",
)

train = get_config("common/train.py").train
train.gradient_clipping = {
    "enable": True,
    "type": "norm",
    "norm": 2,
    "value": 0.1,
}
train.max_epoch = 500
train.init_checkpoint = "torchvision://ImageNet/ResNet-50"
train.output_dir = "./output/detr_resnet50_500e_coco_detection"
