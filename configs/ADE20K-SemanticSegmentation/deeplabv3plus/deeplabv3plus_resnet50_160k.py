from overdrive.model_zoo import get_config

from .deeplabv3plus_resnet50_80k import dataloader
from .deeplabv3plus_resnet50_80k import model
from .deeplabv3plus_resnet50_80k import optimizer
from .deeplabv3plus_resnet50_80k import train

lr_multiplier = get_config("common/scheduler/ade20k_scheduler.py").lr_multiplier_160k

train.max_iter = 160000
train.output_dir = "./output/deeplabv3plus_resnet50_160k_ade20k_sem_seg"
