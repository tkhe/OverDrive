import numpy as np
from overdrive.data.transforms import Augmentation
from overdrive.data.transforms import CropTransform
from overdrive.data.transforms import NoOpTransform
from overdrive.data.transforms import RandomCrop
from overdrive.data.transforms import Transform

from .transform import LightingTransform

__all__ = [
    "RandomCropWithInstance",
    "RandomLighting",
]


class RandomCropWithInstance(RandomCrop):
    """
    Randomly crop an image around a selected instance's bounding box center.

    Args:
        crop_type: One of "relative", "relative_range", "absolute", or
            "absolute_range", with the same semantics as :class:`RandomCrop`.
        crop_size: Crop size in (height, width) order, or the size range for
            "absolute_range".
        prob: Probability of applying the crop. Defaults to 0.5.

    When cropping is applied, missing or empty boxes use a uniformly sampled
    crop position within the image. Instance-based origin bounds use floor/ceil rounding
    of the box center, allowing subpixel centers to lie less than one pixel
    outside the crop boundary.
    """

    def __init__(self, crop_type: str, crop_size: tuple[float, float], prob: float = 0.5):
        if not 0 <= prob <= 1:
            raise ValueError("prob must be between 0 and 1.")

        super().__init__(crop_type, crop_size, prob=prob)

    def get_transform(self, image: np.ndarray, boxes: np.ndarray | None = None) -> Transform:
        """
        Generate a crop from an image and (N, 4) XYXY_ABS boxes.
        """
        do = self._rand_range() < self.prob or self.force_apply
        if not do:
            return NoOpTransform()

        has_boxes = boxes is not None and len(boxes) > 0
        if has_boxes and (boxes.ndim != 2 or boxes.shape[1] != 4):
            raise ValueError(f"Expected boxes with shape (N, 4), got {boxes.shape}.")

        h, w = image.shape[:2]
        crop_size = np.asarray(self.get_crop_size((h, w)), dtype=np.int32)
        image_size = np.array([h, w], dtype=np.int32)

        if not np.all((crop_size > 0) & (crop_size <= image_size)):
            raise ValueError("Crop size must be positive and no larger than the image size.")

        min_yx = np.zeros(2, dtype=np.int32)
        max_yx = image_size - crop_size
        if has_boxes:
            # Select an instance to constrain the crop origin.
            bbox = boxes[np.random.choice(len(boxes))].astype(np.float64)
            center_yx = np.array([(bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5])
            if not np.all(np.isfinite(center_yx) & (center_yx >= 0) & (center_yx <= image_size)):
                raise ValueError("The selected annotation bounding box center is outside of the image or non-finite.")

            min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
            max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))
        y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
        x0 = np.random.randint(min_yx[1], max_yx[1] + 1)
        return CropTransform(x0, y0, int(crop_size[1]), int(crop_size[0]), origin_w=w, origin_h=h)


class RandomLighting(Augmentation):
    """
    Apply a random channel offset using fixed ImageNet color principal components.

    Args:
        scale: Nonnegative standard deviation of principal component weights.
        prob: Probability of applying the augmentation. Defaults to 0.5.

    Images must have three channels in BGR order. PCA offsets are added
    directly in input pixel units.
    """

    def __init__(self, scale: float, prob: float = 0.5):
        super().__init__()

        if not np.isfinite(scale) or scale < 0:
            raise ValueError("scale must be finite and nonnegative.")

        if not 0 <= prob <= 1:
            raise ValueError("prob must be between 0 and 1.")

        self.scale = scale
        self.prob = prob
        self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])
        self.eigen_vecs = np.array([[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]])

    def get_transform(self, image: np.ndarray) -> Transform:
        LightingTransform._check_lighting_image(image)
        do = self.prob == 1.0 or self._rand_range() < self.prob or self.force_apply
        if not do:
            return NoOpTransform()

        weights = np.random.normal(scale=self.scale, size=3)
        offset_rgb = self.eigen_vecs.dot(weights * self.eigen_vals)
        return LightingTransform(offset_rgb[::-1])
