import cv2
import numpy as np
from overdrive.data.transforms import NoOpTransform
from overdrive.data.transforms import Transform

__all__ = ["LightingTransform"]


class LightingTransform(Transform):
    """
    Add a fixed, normalized BGR channel offset to an image.

    Args:
        offset: Three finite channel offsets in BGR order, expressed in [0, 1]
            image units. Offsets may be negative.

    Floating point images use [0, 1]. For uint8 images, the offset is scaled
    by 255. Outputs are clipped to the corresponding range and retain the
    input dtype. Input arrays are not modified.
    """

    def __init__(self, offset: np.ndarray):
        super().__init__()

        self.offset = np.array(offset, dtype=np.float64, copy=True)
        if self.offset.shape != (3,) or not np.all(np.isfinite(self.offset)):
            raise ValueError("offset must contain three finite values in BGR order.")

    @staticmethod
    def _check_lighting_image(image: np.ndarray):
        if not isinstance(image, np.ndarray):
            raise TypeError("Expected a numpy array for the image.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected a three-channel HWC image, got shape {image.shape}.")

        if image.dtype != np.uint8 and not np.issubdtype(image.dtype, np.floating):
            raise TypeError(f"Expected a uint8 or floating point image, got {image.dtype}.")

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        self._check_lighting_image(image)
        if image.dtype == np.uint8:
            if image.size == 0:
                return image.copy()

            # Evaluate float64 offsets only for the 256 possible uint8 values.
            levels = np.arange(256, dtype=np.float64).reshape(256, 1, 1)
            lut = np.clip(levels + self.offset * 255, 0, 255).astype(np.uint8)
            return cv2.LUT(image, lut)

        return np.clip(image + self.offset, 0, 1.0).astype(image.dtype)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def inverse(self) -> Transform:
        return NoOpTransform()
