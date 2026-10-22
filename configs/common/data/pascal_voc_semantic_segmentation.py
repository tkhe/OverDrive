from omegaconf import OmegaConf

from overdrive.config import LazyCall as L
from overdrive.data import DatasetMapper
from overdrive.data import build_detection_test_loader
from overdrive.data import build_detection_train_loader
from overdrive.data import get_detection_dataset_dicts
from overdrive.data import transforms as T

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names=["voc_2012_sem_seg_train"]),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeScale)(
                min_scale=0.5,
                max_scale=2.0,
                target_short_edge=512,
                target_long_edge=2048,
            ),
            L(T.RandomCrop_CategoryAreaConstraint)(
                crop_type="absolute",
                crop_size=(512, 512),
                single_category_max_area=0.75,
                ignored_category=255,
            ),
            L(T.RandomFlip)(direction="horizontal"),
            L(T.RandomBrightness)(),
            L(T.RandomContrast)(),
            L(T.RandomSaturation)(),
            L(T.RandomHue)(),
        ],
        image_format="RGB",
        use_box=False,
        use_sem_seg=True,
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names=["voc_2007_test"],
        filter_empty=False,
    ),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=512,
                max_size=2048,
            ),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
)
