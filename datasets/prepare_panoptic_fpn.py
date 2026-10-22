import argparse
import json
import multiprocessing as mp
import os
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image

from overdrive.data.datasets.builtin_meta import COCO_CATEGORIES


def _process_panoptic_to_semantic(input_panoptic, output_semantic, segments, id_map):
    with Image.open(input_panoptic) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint32)
    panoptic = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    semantic = np.full(panoptic.shape, 255, dtype=np.uint8)
    for segment in segments:
        semantic[panoptic == segment["id"]] = id_map[segment["category_id"]]
    Path(output_semantic).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(semantic).save(output_semantic)


def separate_coco_semantic_from_panoptic(
    panoptic_json,
    panoptic_root,
    sem_seg_root,
    categories=COCO_CATEGORIES,
    workers=1,
):
    """
    Map things to 0, stuff to 1..N in metadata order, and void to 255.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")

    stuff_ids = [category["id"] for category in categories if not category["isthing"]]
    if len(stuff_ids) > 254:
        raise ValueError("At most 254 stuff categories can be encoded")

    id_map = {category_id: i + 1 for i, category_id in enumerate(stuff_ids)}
    id_map.update({category["id"]: 0 for category in categories if category["isthing"]})
    id_map[0] = 255
    with open(panoptic_json) as file:
        annotations = json.load(file)["annotations"]

    jobs = [
        (str(Path(panoptic_root) / ann["file_name"]), str(Path(sem_seg_root) / ann["file_name"]), ann["segments_info"])
        for ann in annotations
    ]
    process = partial(_process_panoptic_to_semantic, id_map=id_map)
    if workers == 1:
        for job in jobs:
            process(*job)
    else:
        with mp.Pool(workers) as pool:
            pool.starmap(process, jobs, chunksize=100)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default=os.path.join(os.getenv("OVERDRIVE_DATASETS", "datasets"), "coco"),
        help="COCO root containing annotations/ and panoptic_<split>/",
    )
    parser.add_argument("--split", nargs="+", choices=["train2017", "val2017"], default=["train2017", "val2017"])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    root = Path(args.dataset_root)
    for split in args.split:
        separate_coco_semantic_from_panoptic(
            root / "annotations" / f"panoptic_{split}.json",
            root / f"panoptic_{split}",
            root / f"panoptic_stuff_{split}",
            workers=args.workers,
        )
        print(f"Wrote {root / f'panoptic_stuff_{split}'}")


if __name__ == "__main__":
    main()
