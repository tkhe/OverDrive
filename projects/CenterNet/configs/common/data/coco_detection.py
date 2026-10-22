from omegaconf import OmegaConf
from overdrive.config import LazyCall as L
from overdrive.data import DatasetMapper
from overdrive.data import build_detection_test_loader
from overdrive.data import build_detection_train_loader
from overdrive.data import get_detection_dataset_dicts
from overdrive.data import transforms as T

from centernet.data.transforms import RandomCropWithInstance
from centernet.data.transforms import RandomLighting

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names=["coco_2017_train"]),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(RandomCropWithInstance)(
                crop_type="relative_range",
                crop_size=(0.6, 0.6),
                prob=1.0,
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(384, 448, 512),
                sample_style="choice",
                max_size=512,
            ),
            L(T.RandomBrightness)(),
            L(T.RandomContrast)(),
            L(T.RandomSaturation)(),
            L(RandomLighting)(scale=0.1),
            L(T.RandomFlip)(direction="horizontal"),
        ],
        image_format="RGB",
    ),
    persistent_workers=True,
    pin_memory=True,
    total_batch_size=64,
    num_workers=8,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names=["coco_2017_val"],
        filter_empty=False,
    ),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[L(T.NoOpTransform)()],
        image_format="${...train.mapper.image_format}",
    ),
    batch_size=1,
    num_workers=2,
)
