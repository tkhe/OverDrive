import os
import xml.etree.ElementTree as ET

import numpy as np

from overdrive.data.catalog import DatasetCatalog
from overdrive.data.catalog import MetadataCatalog
from overdrive.structures import BoxMode
from overdrive.utils.file_io import PathManager

__all__ = [
    "load_voc_instances",
    "load_voc_sem_seg",
    "register_pascal_voc",
    "register_pascal_voc_sem_seg",
]

CLASS_NAMES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


def load_voc_instances(dirname: str, split: str, class_names: list[str] | tuple[str, ...]):
    """
    Load Pascal VOC detection annotations to OverDrive format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=str)

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            #     continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instance = {
                "category_id": class_names.index(cls),
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
            }
            instances.append(instance)
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def load_voc_sem_seg(dirname: str, split: str):
    """
    Load Pascal VOC semantic segmentation annotations to ember format.

    Args:
        dirname: Contain "SegmentationClass", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Segmentation", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=str)

    dicts = []
    for fileid in fileids:
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")
        seg_file = os.path.join(dirname, "SegmentationClass", fileid + ".png")

        r = {
            "file_name": jpeg_file,
            "sem_seg_file_name": seg_file,
        }
        dicts.append(r)
    return dicts


def register_pascal_voc(name, dirname, split, year, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names),
        dirname=dirname,
        year=year,
        split=split,
    )


def register_pascal_voc_sem_seg(name, dirname, split, year, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_voc_sem_seg(dirname, split))
    MetadataCatalog.get(name).set(
        stuff_classes=["background"] + list(class_names),
        dirname=dirname,
        year=year,
        split=split,
        ignore_label=255,
    )
