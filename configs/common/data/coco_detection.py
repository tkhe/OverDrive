from omegaconf import OmegaConf

from overdrive.config import LazyCall as L
from overdrive.data import DatasetMapper
from overdrive.data import build_detection_test_loader
from overdrive.data import build_detection_train_loader
from overdrive.data import get_detection_dataset_dicts
from overdrive.data import transforms as T

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names=["coco_2017_train"]),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=(640, 672, 704, 736, 768, 800),
                sample_style="choice",
                max_size=1333,
            ),
            L(T.RandomFlip)(direction="horizontal"),
        ],
        image_format="RGB",
    ),
    persistent_workers=True,
    pin_memory=True,
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names=["coco_2017_val"],
        filter_empty=False,
    ),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
)
