import os

import imagesize

from overdrive.data.catalog import DatasetCatalog
from overdrive.data.catalog import MetadataCatalog
from overdrive.structures import BoxMode

__all__ = [
    "load_visdrone_instances",
    "register_visdrone",
]

CLASS_NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)

THING_COLORS = [
    [220, 20, 60],
    [0, 0, 142],
    [119, 11, 32],
    [0, 60, 100],
    [0, 0, 230],
    [0, 80, 100],
    [250, 170, 30],
    [153, 153, 153],
    [0, 60, 200],
    [220, 220, 0],
]


def load_visdrone_instances(dirname: str, split: str, class_names: list[str] | tuple[str, ...] = CLASS_NAMES):
    """
    Load VisDrone detection annotations.

    Args:
        dirname: Root directory of VisDrone dataset, containing split subdirectories.
        split: One of "VisDrone2019-DET-train", "VisDrone2019-DET-val".
        class_names: Tuple of class names (10 classes).
    """
    image_dir = os.path.join(dirname, split, "images")
    anno_dir = os.path.join(dirname, split, "annotations")

    image_files = sorted(f for f in os.listdir(image_dir) if f.endswith(".jpg"))

    dicts = []
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        stem = os.path.splitext(image_file)[0]
        anno_path = os.path.join(anno_dir, stem + ".txt")

        width, height = imagesize.get(image_path)

        r = {
            "file_name": image_path,
            "image_id": stem,
            "height": height,
            "width": width,
        }

        annotations = []
        with open(anno_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                score = int(parts[4])
                category = int(parts[5]) - 1

                if score == 0 or category >= len(class_names):
                    continue

                item = {
                    "category_id": category,
                    "bbox": [x, y, w, h],
                    "bbox_mode": BoxMode.XYWH_ABS,
                }
                annotations.append(item)

        r["annotations"] = annotations
        dicts.append(r)
    return dicts


def register_visdrone(name, dirname, split, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_visdrone_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names),
        thing_colors=THING_COLORS,
        dirname=dirname,
        split=split,
    )
