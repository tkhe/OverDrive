import logging
import math
from typing import Any

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.utils.data as torchdata

from overdrive.data.catalog import MetadataCatalog
from overdrive.data.common import AspectRatioGroupedDataset
from overdrive.data.common import ToIterableDataset
from overdrive.data.common import _MapIterableDataset
from overdrive.structures import BitMasks
from overdrive.structures import Boxes
from overdrive.structures import BoxMode
from overdrive.structures import Instances
from overdrive.structures import Keypoints
from overdrive.structures import PolygonMasks
from overdrive.structures import polygons_to_bitmask
from overdrive.utils.comm import get_world_size


class SizeMismatchError(ValueError):
    pass


def convert_image_to_rgb(image: np.ndarray | torch.Tensor, format: str) -> np.ndarray:
    """
    Convert an image from the specified format to RGB.

    Args:
        image (Union[np.ndarray, torch.Tensor]): The image to convert.
        format (str): The current format of the input image (e.g., "RGB", "BGR", "YCrCb", "GRAY").

    Returns:
        np.ndarray: The converted image in RGB format.
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    if format == "RGB":
        return image
    elif format == "BGR":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif format == "GRAY":
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Unsupported image format: {format}")


def convert_bgr_to_image_format(image: np.ndarray, format: str) -> np.ndarray:
    """
    Convert an OpenCV BGR image to the requested model input format.
    """
    if format == "BGR":
        return image
    elif format == "RGB":
        return image[:, :, ::-1]
    else:
        raise ValueError(f"Unsupported image format '{format}'")


def read_image(file_name: str, format: str = "BGR"):
    image = cv2.imread(file_name)
    if image is None:
        raise ValueError(f"Fail to read image from {file_name}")
    if format == "BGR":
        return image
    elif format == "RGB":
        return image[:, :, ::-1]
    elif format == "GRAY":
        return np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), axis=-1)
    else:
        raise ValueError(f"Unsupported image format '{format}'")


def check_image_size(dataset_dict: dict[str, Any], image: np.ndarray):
    """
    Raise an error if the image does not match the size specified in the dict.
    """
    if "width" in dataset_dict or "height" in dataset_dict:
        image_wh = (image.shape[1], image.shape[0])
        expected_wh = (dataset_dict["width"], dataset_dict["height"])
        if image_wh != expected_wh:
            raise SizeMismatchError(
                f"Mismatched image shape{' for image' + dataset_dict['file_name'] if 'file_name' in dataset_dict else ''}, got {image_wh}, expect {expected_wh}."
                "Please check the width/height in your annotation."
            )

    # To ensure bbox always remap to original image size
    if "width" not in dataset_dict:
        dataset_dict["width"] = image.shape[1]
    if "height" not in dataset_dict:
        dataset_dict["height"] = image.shape[0]


def get_boxes(annotations):
    boxes = [BoxMode.convert(anno["bbox"], anno["bbox_mode"], BoxMode.XYXY_ABS) for anno in annotations if anno.get("iscrowd", 0) == 0]
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def get_masks(annotations):
    masks = []
    for anno in annotations:
        if anno.get("iscrowd", 0) == 0:
            if "segmentation" not in anno:
                raise ValueError("Expected 'segmentation' in anno")

            segm = anno["segmentation"]
            if isinstance(segm, list):
                # polygons
                polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
                masks.append(polygons)
            elif isinstance(segm, dict):
                # RLE
                masks.append(mask_util.decode(segm))
            else:
                raise ValueError(f"Unsupported segmentation format '{type(segm)}'")
    return masks


def get_keypoints(annotations):
    keypoints = []
    for anno in annotations:
        if anno.get("iscrowd", 0) == 0 and "keypoints" in anno:
            keypoints.append(np.asarray(anno["keypoints"], dtype="float64").reshape(-1, 3))
    keypoints = np.asarray(keypoints, dtype=np.float32)
    return keypoints


def get_labels(annotations):
    labels = [anno["category_id"] for anno in annotations if anno.get("iscrowd", 0) == 0]
    return labels


def annotation_to_instances(aug_input, image_size, mask_format="polygon"):
    instances = Instances(image_size)

    if aug_input.labels is not None:
        instances.gt_classes = torch.tensor(aug_input.labels, dtype=torch.int64)

    if aug_input.boxes is not None:
        instances.gt_boxes = Boxes(aug_input.boxes)

    if aug_input.masks is not None:
        masks = aug_input.masks
        if mask_format == "polygon":
            try:
                masks = PolygonMasks([[p.reshape(-1) for p in mask] for mask in masks])
            except ValueError as e:
                raise ValueError("Failed to use mask_format=='polygon' from the given annotations!") from e
        else:
            if mask_format != "bitmask":
                raise ValueError("Expected mask_format == 'bitmask'")

            new_masks = []
            for mask in masks:
                if isinstance(mask, list):
                    # Transforms use Nx2 coordinates; COCOAPI expects flat polygons.
                    polygons = [polygon.reshape(-1) for polygon in mask]
                    new_masks.append(polygons_to_bitmask(polygons, image_size[0], image_size[1]))
                elif isinstance(mask, np.ndarray):
                    new_masks.append(mask)
                elif isinstance(mask, dict):
                    new_masks.append(mask_util.decode(mask))
                else:
                    raise ValueError(f"Unsupported mask type {type(mask)}")

            masks = BitMasks(np.stack(new_masks) if new_masks else np.zeros((0, *image_size), dtype=bool))

        instances.gt_masks = masks

    if aug_input.keypoints is not None:
        keypoints = aug_input.keypoints
        keypoints_xy = keypoints[..., :2]
        inside = (keypoints_xy >= np.array([0, 0])) & (keypoints_xy <= np.array(image_size[::-1]))
        inside = inside.all(axis=-1)
        keypoints[..., :2] = keypoints_xy
        keypoints[..., 2][~inside] = 0
        keypoints[keypoints[..., 2] == 0] = 0
        instances.gt_keypoints = Keypoints(keypoints)

    return instances


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5, return_mask=False):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty
        return_mask (bool): whether to return boolean mask of filtered instances

    Returns:
        Instances: the filtered instances.
        tensor[bool], optional: boolean mask of filtered instances
    """
    if not (by_box or by_mask):
        raise ValueError("Expected by_box or by_mask")

    r = []

    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))

    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    # TODO: can also filter visible keypoints

    if not r:
        return instances

    m = r[0]
    for x in r[1:]:
        m = m & x
    if return_mask:
        return instances[m], m
    return instances[m]


def create_keypoint_hflip_indices(dataset_names: str | list[str]) -> list[int]:
    """
    Args:
        dataset_names: list of dataset names

    Returns:
        list[int]: a list of size=#keypoints, storing the
        horizontally-flipped keypoint indices.
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    check_metadata_consistency("keypoint_names", dataset_names)
    check_metadata_consistency("keypoint_flip_map", dataset_names)

    meta = MetadataCatalog.get(dataset_names[0])
    names = meta.keypoint_names
    # TODO flip -> hflip
    flip_map = dict(meta.keypoint_flip_map)
    flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in flip_map else flip_map[i] for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return flip_indices


def check_metadata_consistency(key, dataset_names):
    """
    Check that the datasets have consistent metadata.

    Args:
        key (str): a metadata key
        dataset_names (list[str]): a list of dataset names

    Raises:
        AttributeError: if the key does not exist in the metadata
        ValueError: if the given datasets do not have the same metadata values defined by key
    """
    if len(dataset_names) == 0:
        return

    logger = logging.getLogger(__name__)
    entries_per_dataset = [getattr(MetadataCatalog.get(d), key) for d in dataset_names]
    for idx, entry in enumerate(entries_per_dataset):
        if entry != entries_per_dataset[0]:
            logger.error(f"Metadata '{key}' for dataset '{dataset_names[idx]}' is '{str(entry)}'")
            logger.error(f"Metadata '{key}' for dataset '{dataset_names[0]}' is '{str(entries_per_dataset[0])}'")
            raise ValueError(f"Datasets have different metadata '{key}'!")


def compute_max_step_in_epoch(dataloader, gradient_accumulation_steps=1):
    """
    Compute the number of optimizer steps per epoch.

    This function unwraps the dataloader to extract the underlying dataset length
    and per-GPU batch size, then calculates:
        ceil(dataset_length / (batch_size_per_gpu * num_gpus * gradient_accumulation_steps))

    Args:
        dataloader: a dataloader built by ``build_batch_data_loader`` or
            ``build_detection_train_loader``. Can be a ``DataLoader``,
            ``AspectRatioGroupedDataset``, or ``_MapIterableDataset``.
        gradient_accumulation_steps (int): number of gradient accumulation steps.

    Returns:
        int: the number of optimizer steps in one epoch (rounded up).
    """
    obj = dataloader
    batch_size_per_gpu = None

    # Unwrap _MapIterableDataset (from MapDataset wrapping an IterableDataset)
    if isinstance(obj, _MapIterableDataset):
        obj = obj._dataset

    # Unwrap AspectRatioGroupedDataset and extract batch_size
    if isinstance(obj, AspectRatioGroupedDataset):
        batch_size_per_gpu = obj.batch_size
        obj = obj.dataset

    # Unwrap DataLoader
    if isinstance(obj, torchdata.DataLoader):
        if batch_size_per_gpu is None:
            batch_size_per_gpu = obj.batch_size
        obj = obj.dataset

    # Get dataset length
    if isinstance(obj, ToIterableDataset):
        dataset_length = len(obj.dataset)
    elif hasattr(obj, "__len__"):
        dataset_length = len(obj)
    else:
        raise TypeError(f"Cannot determine dataset length from dataloader of type {type(dataloader)}. Unwrapped to {type(obj)} which has no __len__.")

    if batch_size_per_gpu is None:
        raise ValueError(f"Cannot determine batch_size_per_gpu from dataloader of type {type(dataloader)}.")

    num_gpus = get_world_size()
    global_batch_size = batch_size_per_gpu * num_gpus * gradient_accumulation_steps
    return math.ceil(dataset_length / global_batch_size)
