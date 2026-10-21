import copy
import logging

import numpy as np
import torch

from . import transforms as T
from . import utils

__all__ = ["DatasetMapper"]


class DatasetMapper:
    """
    A callable that maps a dataset dictionary into a format consumable by
    the model.

    It loads the image, extracts annotations (boxes, masks, keypoints, labels,
    semantic segmentation), applies augmentations / transforms (in-place via
    :class:`AugInput`), and returns the processed dict. The behaviour differs
    between training and inference mode:

    - **Training**: full augmentation pipeline is applied. Enabled instance
      annotations are converted into an :class:`Instances` object (with empty
      instances filtered out), and semantic segmentation is returned separately.
    - **Inference**: augmentations are applied as deterministic transforms only;
      the returned dict contains ``"transforms"`` (a :class:`TransformList`) so
      that model outputs can be inversely transformed back to original image
      coordinates.

    Args:
        is_train (bool): Whether this mapper is used for training or inference.
        augmentations (list[Augmentation or Transform]): Augmentations /
            deterministic transforms to apply. Internally wrapped in an
            :class:`AugmentationList`.
        image_format (str): Final model input image format (``"BGR"`` or
            ``"RGB"``). Augmentations always receive BGR images.
        mask_format (str): Format to store instance masks in the output
            :class:`Instances`. ``"polygon"`` (default) produces
            :class:`PolygonMasks`; ``"bitmask"`` produces :class:`BitMasks`.
        use_box (bool): If ``True``, extract bounding boxes from annotations.
            Default ``True``.
        use_mask (bool): If ``True``, extract instance segmentation masks from
            annotations. Default ``False``.
        use_keypoint (bool): If ``True``, extract keypoints from annotations.
            Default ``False``.
        use_sem_seg (bool): If ``True`` during training, require
            ``"sem_seg_file_name"`` in the dataset dict and load semantic
            segmentation. Default ``False``.
        keypoint_hflip_indices (np.ndarray or None): Indices for horizontal
            flipping of keypoints (see
            :func:`utils.create_keypoint_hflip_indices`). Required when using
            keypoints together with horizontal flip augmentations.
        recompute_boxes (bool): If ``True``, recompute bounding boxes from
            instance masks after augmentations are applied. Requires
            ``use_mask=True``. Default ``False``.

    Raises:
        ValueError: If ``recompute_boxes=True`` but ``use_mask=False``, or a
            training sample lacks ``"sem_seg_file_name"`` when semantic
            segmentation is enabled.
    """

    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: list[T.Augmentation | T.Transform],
        image_format: str,
        mask_format: str = "polygon",
        use_box: bool = True,
        use_mask: bool = False,
        use_keypoint: bool = False,
        use_sem_seg: bool = False,
        keypoint_hflip_indices: np.ndarray | None = None,
        recompute_boxes: bool = False,
    ):
        if recompute_boxes and not use_mask:
            raise ValueError("recompute_boxes requires instance masks")

        self.is_train = is_train
        if image_format not in ("BGR", "RGB"):
            raise ValueError(f"Unsupported image_format '{image_format}'")

        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.mask_format = mask_format
        self.use_box = use_box
        self.use_mask = use_mask
        self.use_keypoint = use_keypoint
        self.use_sem_seg = use_sem_seg
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.recompute_boxes = recompute_boxes

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    def load_dataset_dict(self, dataset_dict):
        """
        Load and pre-process a single dataset dictionary into a format ready
        for augmentation.

        This method reads the image from disk and extracts annotations from the
        dataset dict depending on the configured flags (``use_box``,
        ``use_mask``, ``use_keypoint``, ``use_sem_seg``).

        In **inference** mode, only the image and metadata (file name, image
        ID, width, height) are loaded — annotations are skipped.

        In **training** mode, the following annotation fields may be extracted:

        - ``"boxes"``: bounding boxes in XYXY_ABS format (if ``use_box=True``)
        - ``"masks"``: instance masks as polygons or RLE-decoded bitmasks (if
          ``use_mask=True``)
        - ``"keypoints"``: keypoint arrays of shape (N, K, 3) (if
          ``use_keypoint=True``)
        - ``"keypoint_hflip_indices"``: horizontal flip index mapping (if
          ``use_keypoint=True``)
        - ``"labels"``: category ID list (if any of box/mask/keypoint is used)
        - ``"sem_seg"``: semantic segmentation map (if
          ``use_sem_seg=True``; ``"sem_seg_file_name"`` is then required)

        Args:
            dataset_dict (dict): A dataset dictionary containing at least
                ``"file_name"``. Training annotations are needed for enabled
                annotation types.

        Returns:
            dict: The processed dictionary with loaded image and extracted
            annotation arrays, ready to be consumed by :class:`AugInput`.
        """
        result = {"file_name": dataset_dict["file_name"]}
        if "image_id" in dataset_dict:
            result["image_id"] = dataset_dict["image_id"]
        if "width" in dataset_dict:
            result["width"] = dataset_dict["width"]
        if "height" in dataset_dict:
            result["height"] = dataset_dict["height"]

        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        utils.check_image_size(result, image)
        result["image"] = image

        if not self.is_train:
            return result

        if self.use_sem_seg:
            if "sem_seg_file_name" not in dataset_dict:
                raise ValueError(f"Missing 'sem_seg_file_name' for training image {dataset_dict['file_name']}.")
            sem_seg = utils.read_image(dataset_dict["sem_seg_file_name"], format="GRAY").squeeze(2)
            result["sem_seg"] = sem_seg

        annotations = dataset_dict.pop("annotations", [])

        if self.use_box:
            result["boxes"] = utils.get_boxes(annotations)

        if self.use_mask:
            result["masks"] = utils.get_masks(annotations)

        if self.use_keypoint:
            result["keypoints"] = utils.get_keypoints(annotations)
            result["keypoint_hflip_indices"] = self.keypoint_hflip_indices

        if self.use_box or self.use_mask or self.use_keypoint:
            result["labels"] = utils.get_labels(annotations)
        return result

    def __call__(self, dataset_dict):
        """
        Process a single dataset dictionary and return a model-ready dict.

        The processing pipeline is:

        1. Deep-copy the input dict (it will be mutated in place).
        2. Call :meth:`load_dataset_dict` to load the image and extract
           annotations.
        3. Build an :class:`AugInput` and apply the configured augmentations.
        4. Convert the augmented BGR image to the configured model input
           format, then to a ``CHW`` ``torch.Tensor``.
        5. In **inference** mode: store the applied transforms (a
           :class:`TransformList`) under ``"transforms"`` and return. The
           transforms can later be inverted to map model outputs back to the
           original image coordinates.
        6. In **training** mode: return enabled semantic segmentation as a
           tensor. If instance annotations are enabled, convert them to an
           :class:`Instances` object, optionally recompute boxes from masks,
           and filter out empty instances.

        Args:
            dataset_dict (dict): A dataset dictionary containing at least
                ``"file_name"``. During training, ``"sem_seg_file_name"`` is
                required if ``use_sem_seg=True``.

        Returns:
            dict: The processed dictionary with keys:

            - ``"file_name"`` (str)
            - ``"image_id"`` (optional)
            - ``"width"``, ``"height"``
            - ``"image"`` (``torch.Tensor``, CHW format)
            - ``"transforms"`` (inference only — :class:`TransformList`)
            - ``"instances"`` (training only, if instance annotations are
              enabled — :class:`Instances`)
            - ``"sem_seg"`` (training only, if ``use_sem_seg=True`` —
              ``torch.Tensor``)
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        dataset_dict = self.load_dataset_dict(dataset_dict)
        aug_input = T.AugInput.from_dataset_dict(dataset_dict)

        transforms = self.augmentations(aug_input)

        image = utils.convert_bgr_to_image_format(aug_input.image, self.image_format)
        image_shape = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).copy())

        if not self.is_train:
            dataset_dict["transforms"] = transforms
            return dataset_dict

        if self.use_sem_seg:
            dataset_dict["sem_seg"] = torch.as_tensor(aug_input.sem_seg.astype("long"))

        if self.use_box or self.use_mask or self.use_keypoint:
            instances = utils.annotation_to_instances(aug_input, image_shape, self.mask_format)

            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            dataset_dict["instances"] = utils.filter_empty_instances(instances, by_box=instances.has("gt_boxes"))

        return dataset_dict
