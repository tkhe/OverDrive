import random
import sys

import numpy as np

from .augmentation import Augmentation
from .augmentation import AugmentationList
from .augmentation import _transform_to_aug
from .transform import BlurTransform
from .transform import BrightnessTransform
from .transform import CLAHETransform
from .transform import ContrastTransform
from .transform import CropTransform
from .transform import HorizontalFlipTransform
from .transform import HueTransform
from .transform import MedianBlurTransform
from .transform import NoOpTransform
from .transform import PadTransform
from .transform import ResizeTransform
from .transform import RotationTransform
from .transform import SaturationTransform
from .transform import Transform
from .transform import TransformList
from .transform import VerticalFlipTransform

__all__ = [
    "AlwaysApply",
    "FixedSizeCrop",
    "OneOf",
    "RandomApply",
    "RandomBlur",
    "RandomBrightness",
    "RandomCLAHE",
    "RandomContrast",
    "RandomCrop",
    "RandomCrop_CategoryAreaConstraint",
    "RandomFlip",
    "RandomHue",
    "RandomMedianBlur",
    "RandomRotation",
    "RandomSaturation",
    "Resize",
    "ResizeScale",
    "ResizeShortestEdge",
]


class RandomApply(Augmentation):
    """
    Randomly apply an augmentation with a given probability.
    """

    def __init__(self, tfm_or_aug: Augmentation | Transform, prob: float = 0.5):
        """
        Args:
            tfm_or_aug (Transform, Augmentation): the transform or augmentation
                to be applied. It can either be a `Transform` or `Augmentation`
                instance.
            prob (float): probability between 0.0 and 1.0 that
                the wrapper transformation is applied
        """
        super().__init__()

        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"Probablity must be between 0.0 and 1.0 (given: {prob})")

        self.aug = _transform_to_aug(tfm_or_aug)
        self.prob = prob

    def get_transform(self, *args) -> Transform:
        do = self.force_apply or self._rand_range() < self.prob
        if do:
            return self.aug.get_transform(*args)
        else:
            return NoOpTransform()

    def __call__(self, aug_input) -> Transform:
        do = self.force_apply or self._rand_range() < self.prob
        if do:
            return self.aug(aug_input)
        else:
            return NoOpTransform()


class RandomFlip(Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob: float = 0.5, direction: str = "horizontal"):
        """
        Args:
            prob (float): probability of flip.
            direction (str): whether to apply horizontal or vertical flipping
        """
        super().__init__()

        if direction not in ("horizontal", "vertical"):
            raise ValueError("Expected direction in ('horizontal', 'vertical')")

        self.prob = prob
        self.direction = direction

    def get_transform(self, image: np.ndarray, keypoint_hflip_indices: np.ndarray) -> Transform:
        h, w = image.shape[:2]

        do = self._rand_range() < self.prob or self.force_apply
        if do:
            if self.direction == "horizontal":
                return HorizontalFlipTransform(w, keypoint_hflip_indices)
            elif self.direction == "vertical":
                return VerticalFlipTransform(h)
        else:
            return NoOpTransform()


class Resize(Augmentation):
    """
    Resize image to a fixed target size.
    """

    def __init__(self, shape: int | tuple[int, int], interp: str = "bilinear"):
        """
        Args:
            shape: (h, w) tuple or a int
            interp: interpolation method
        """
        super().__init__()

        if isinstance(shape, int):
            shape = (shape, shape)

        shape = tuple(shape)

        self.shape = shape
        self.interp = interp

    def get_transform(self, image: np.ndarray) -> Transform:
        return ResizeTransform(image.shape[1], image.shape[0], self.shape[1], self.shape[0], self.interp)


class ResizeShortestEdge(Augmentation):
    """
    Resize the image while keeping the aspect ratio unchanged.
    It attempts to scale the shorter edge to the given `short_edge_length`,
    as long as the longer edge does not exceed `max_size`.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(self, short_edge_length: list[int], max_size: int = sys.maxsize, sample_style: str = "choice", interp: str = "bilinear"):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        if sample_style not in ["range", "choice"]:
            raise ValueError(sample_style)

        self.is_range = sample_style == "range"

        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)

        if self.is_range:
            if not (len(short_edge_length) == 2):
                raise ValueError(f"short_edge_length must be two values using 'range' sample style. Got {short_edge_length}!")

        self.short_edge_length = short_edge_length
        self.max_size = max_size
        self.sample_style = sample_style
        self.interp = interp

    def get_transform(self, image: np.ndarray) -> Transform:
        h, w = image.shape[:2]

        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)

        if size == 0:
            return NoOpTransform()

        newh, neww = ResizeShortestEdge.get_output_shape(h, w, size, self.max_size)
        return ResizeTransform(w, h, neww, newh, self.interp)

    @staticmethod
    def get_output_shape(oldh: int, oldw: int, short_edge_length: int, max_size: int) -> tuple[int, int]:
        """
        Compute the output size given input size and target short edge length.
        """
        h, w = oldh, oldw
        size = short_edge_length * 1.0
        scale = size / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class RandomCrop(Augmentation):
    """
    Randomly crop a rectangle region out of an image.
    """

    def __init__(self, crop_type: str, crop_size: tuple[float, float], prob: float = 0.5):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
            crop_size (tuple[float, float]): two floats, explained below.

        - "relative": crop a (H * crop_size[0], W * crop_size[1]) region from an input image of
          size (H, W). crop size should be in (0, 1]
        - "relative_range": uniformly sample two values from [crop_size[0], 1]
          and [crop_size[1]], 1], and use them as in "relative" crop type.
        - "absolute" crop a (crop_size[0], crop_size[1]) region from input image.
          crop_size must be smaller than the input image size.
        - "absolute_range", for an input of size (H, W), uniformly sample H_crop in
          [crop_size[0], min(H, crop_size[1])] and W_crop in [crop_size[0], min(W, crop_size[1])].
          Then crop a region (H_crop, W_crop).
        """
        if crop_type not in ("relative_range", "relative", "absolute", "absolute_range"):
            raise ValueError("Expected crop_type in ('relative_range', 'relative', 'absolute', 'absolute_range')")

        self.crop_type = crop_type
        self.crop_size = crop_size
        self.prob = prob

    def get_transform(self, image: np.ndarray) -> Transform:
        do = self._rand_range() < self.prob or self.force_apply

        if do:
            h, w = image.shape[:2]
            croph, cropw = self.get_crop_size((h, w))

            if not (h >= croph and w >= cropw):
                raise ValueError(f"Shape computation in {self} has bugs.")

            h0 = np.random.randint(h - croph + 1)
            w0 = np.random.randint(w - cropw + 1)
            return CropTransform(w0, h0, cropw, croph)
        else:
            return NoOpTransform()

    def get_crop_size(self, image_size: tuple[int, int]):
        """
        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size

        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == "absolute_range":
            if not (self.crop_size[0] <= self.crop_size[1]):
                raise ValueError("Expected self.crop_size[0] <= self.crop_size[1]")

            ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return ch, cw
        else:
            raise NotImplementedError(f"Unknown crop type {self.crop_type}")


class ResizeScale(Augmentation):
    """
    Randomly resize an image by scaling target short/long edge constraints.

    The image is resized while preserving aspect ratio so that it fits inside
    the sampled short-edge and long-edge limits.
    """

    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        target_short_edge: int,
        target_long_edge: int,
        interp: str = "bilinear",
    ):
        """
        Args:
            min_scale: minimum image scale range.
            max_scale: maximum image scale range.
            target_short_edge: target short edge before random scaling.
            target_long_edge: target long edge before random scaling.
            interp: image interpolation method.
        """
        super().__init__()

        if not (min_scale <= max_scale):
            raise ValueError(f"min_scale must be less than or equal to max_scale, got min_scale={min_scale}, max_scale={max_scale}")
        if not (target_short_edge > 0 and target_long_edge > 0):
            raise ValueError(f"target edges must be positive, got target_short_edge={target_short_edge}, target_long_edge={target_long_edge}")
        if not (target_short_edge <= target_long_edge):
            raise ValueError(f"target_short_edge must be less than or equal to target_long_edge, got target_short_edge={target_short_edge}, target_long_edge={target_long_edge}")

        self.min_scale = min_scale
        self.max_scale = max_scale
        self.target_short_edge = target_short_edge
        self.target_long_edge = target_long_edge
        self.interp = interp

    def get_transform(self, image: np.ndarray) -> Transform:
        random_scale = np.random.uniform(self.min_scale, self.max_scale)
        h, w = image.shape[:2]

        target_short_edge = self.target_short_edge * random_scale
        target_long_edge = self.target_long_edge * random_scale

        output_scale = min(target_long_edge / max(h, w), target_short_edge / min(h, w))
        output_h = int(h * output_scale + 0.5)
        output_w = int(w * output_scale + 0.5)

        return ResizeTransform(w, h, output_w, output_h, self.interp)


class FixedSizeCrop(Augmentation):
    """
    If `crop_size` is smaller than the input image size, then it uses a random crop of
    the crop size. If `crop_size` is larger than the input image size, then it pads
    the right and the bottom of the image to the crop size if `pad` is True, otherwise
    it returns the smaller image.
    """

    def __init__(
        self,
        crop_size: int | tuple[int, int],
        pad: bool = True,
        pad_value: float = 114,
        seg_pad_value: int = 255,
    ):
        """
        Args:
            crop_size: target image (height, width).
            pad: if True, will pad images smaller than `crop_size` up to `crop_size`
            pad_value: the padding value to the image.
            seg_pad_value: the padding value to the segmentation mask.
        """
        super().__init__()

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        self.crop_size = crop_size
        self.pad = pad
        self.pad_value = pad_value
        self.seg_pad_value = seg_pad_value

    def _get_crop(self, image: np.ndarray) -> Transform:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = self.crop_size

        # Add random crop if the image is scaled up.
        max_offset = np.subtract(input_size, output_size)
        max_offset = np.maximum(max_offset, 0)
        offset = np.multiply(max_offset, np.random.uniform(0.0, 1.0))
        offset = np.round(offset).astype(int)
        return CropTransform(offset[1], offset[0], output_size[1], output_size[0], input_size[1], input_size[0])

    def _get_pad(self, image: np.ndarray) -> Transform:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = self.crop_size

        # Add padding if the image is scaled down.
        pad_size = np.subtract(output_size, input_size)
        pad_size = np.maximum(pad_size, 0)
        original_size = np.minimum(input_size, output_size)
        return PadTransform(
            0,
            0,
            pad_size[1],
            pad_size[0],
            original_size[1],
            original_size[0],
            self.pad_value,
            self.seg_pad_value,
        )

    def get_transform(self, image: np.ndarray) -> TransformList:
        transforms = [self._get_crop(image)]
        if self.pad:
            transforms.append(self._get_pad(image))
        return TransformList(transforms)


class RandomRotation(Augmentation):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around the given center.
    """

    def __init__(
        self,
        angle: list[float],
        expand: bool = True,
        center: list[tuple[float, float]] = None,
        sample_style: str = "range",
        interp: str = "bilinear",
    ):
        """
        Args:
            angle (list[float]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the angle (in degrees).
                If ``sample_style=="choice"``, a list of angles to sample from
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (list[[float, float]]):  If ``sample_style=="range"``,
                a [[minx, miny], [maxx, maxy]] relative interval from which to sample the center,
                [0, 0] being the top left of the image and [1, 1] the bottom right.
                If ``sample_style=="choice"``, a list of centers to sample from
                Default: None, which means that the center of rotation is the center of the image
                center has no effect if expand=True because it only affects shifting
        """
        super().__init__()

        if sample_style not in ["range", "choice"]:
            raise ValueError(sample_style)

        self.is_range = sample_style == "range"
        if isinstance(angle, (float, int)):
            angle = (angle, angle)

        if center is not None and isinstance(center[0], (float, int)):
            center = (center, center)

        self.angle = angle
        self.center = center
        self.expand = expand
        self.interp = interp

    def get_transform(self, image) -> Transform:
        h, w = image.shape[:2]
        center = None
        if self.is_range:
            angle = np.random.uniform(self.angle[0], self.angle[1])
            if self.center is not None:
                center = (
                    np.random.uniform(self.center[0][0], self.center[1][0]),
                    np.random.uniform(self.center[0][1], self.center[1][1]),
                )
        else:
            angle = np.random.choice(self.angle)
            if self.center is not None:
                center = np.random.choice(self.center)

        if center is not None:
            center = (w * center[0], h * center[1])  # Convert to absolute coordinates

        if angle % 360 == 0:
            return NoOpTransform()

        return RotationTransform(h, w, angle, expand=self.expand, center=center, interp=self.interp)


class RandomCrop_CategoryAreaConstraint(Augmentation):
    """
    Similar to :class:`RandomCrop`, but find a cropping window such that no single category
    occupies a ratio of more than `single_category_max_area` in semantic segmentation ground
    truth, which can cause unstability in training. The function attempts to find such a valid
    cropping window for at most 10 times.
    """

    def __init__(
        self,
        crop_type: str,
        crop_size,
        single_category_max_area: float = 1.0,
        ignored_category: int | None = None,
    ):
        """
        Args:
            crop_type, crop_size: same as in :class:`RandomCrop`
            single_category_max_area: the maximum allowed area ratio of a
                category. Set to 1.0 to disable
            ignored_category: allow this category in the semantic segmentation
                ground truth to exceed the area ratio. Usually set to the category
                that's ignored in training.
        """
        self.crop_aug = RandomCrop(crop_type, crop_size)

        self.single_category_max_area = single_category_max_area
        self.ignored_category = ignored_category

    def get_transform(self, image: np.ndarray, sem_seg: np.ndarray):
        if self.single_category_max_area >= 1.0:
            return self.crop_aug.get_transform(image)
        else:
            h, w = sem_seg.shape
            for _ in range(10):
                crop_size = self.crop_aug.get_crop_size((h, w))
                y0 = np.random.randint(h - crop_size[0] + 1)
                x0 = np.random.randint(w - crop_size[1] + 1)
                sem_seg_temp = sem_seg[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
                labels, cnt = np.unique(sem_seg_temp, return_counts=True)
                if self.ignored_category is not None:
                    cnt = cnt[labels != self.ignored_category]
                if len(cnt) > 1 and np.max(cnt) < np.sum(cnt) * self.single_category_max_area:
                    break
            crop_tfm = CropTransform(x0, y0, crop_size[1], crop_size[0])
            return crop_tfm


class RandomBrightness(Augmentation):
    def __init__(self, lower: float = 0.8, upper: float = 1.2, prob: float = 0.5):
        super().__init__()

        if not (lower <= upper):
            raise ValueError(f"lower must be less than or equal to upper, got lower={lower}, upper={upper}")

        self.lower = lower
        self.upper = upper
        self.prob = prob

    def get_transform(self, image):
        do = self._rand_range() < self.prob or self.force_apply
        if do:
            beta = random.uniform(self.lower, self.upper)
            return BrightnessTransform(beta)
        else:
            return NoOpTransform()


class RandomBlur(Augmentation):
    """
    Blur an image using a randomly sampled square box-filter kernel.
    """

    def __init__(self, blur_range: tuple[int, int] = (3, 7), prob: float = 0.5):
        super().__init__()

        if not isinstance(blur_range, tuple):
            raise TypeError(f"blur_range must be a tuple, got {type(blur_range)}.")

        if len(blur_range) != 2:
            raise ValueError(f"blur_range must contain exactly 2 values, got {len(blur_range)}.")

        if not all(isinstance(value, int) for value in blur_range):
            raise TypeError(f"blur_range values must be ints, got {blur_range}.")

        if blur_range[0] <= 0:
            raise ValueError(f"blur_range values must be positive, got {blur_range}.")

        if blur_range[0] > blur_range[1]:
            raise ValueError(f"blur_range must be ordered from minimum to maximum, got {blur_range}.")

        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"prob must be between 0.0 and 1.0, got {prob}.")

        self.blur_range = blur_range
        self.prob = prob

    def get_transform(self, image: np.ndarray) -> Transform:
        do = self._rand_range() < self.prob or self.force_apply
        if do:
            kernel_size = random.randint(self.blur_range[0], self.blur_range[1])
            return BlurTransform(kernel_size)
        else:
            return NoOpTransform()


class RandomMedianBlur(Augmentation):
    """
    Median-blur an image using a randomly sampled odd kernel size.
    """

    def __init__(self, blur_range: tuple[int, int] = (3, 7), prob: float = 0.5):
        super().__init__()

        if not isinstance(blur_range, tuple):
            raise TypeError(f"blur_range must be a tuple, got {type(blur_range)}.")

        if len(blur_range) != 2:
            raise ValueError(f"blur_range must contain exactly 2 values, got {len(blur_range)}.")

        if not all(isinstance(value, int) for value in blur_range):
            raise TypeError(f"blur_range values must be ints, got {blur_range}.")

        if blur_range[0] <= 0:
            raise ValueError(f"blur_range values must be positive, got {blur_range}.")

        if blur_range[0] > blur_range[1]:
            raise ValueError(f"blur_range must be ordered from minimum to maximum, got {blur_range}.")

        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"prob must be between 0.0 and 1.0, got {prob}.")

        kernel_sizes = tuple(size for size in range(blur_range[0], blur_range[1] + 1) if size > 1 and size % 2 == 1)
        if not kernel_sizes:
            raise ValueError(f"blur_range must contain at least one odd integer greater than 1, got {blur_range}.")

        self.blur_range = blur_range
        self.prob = prob
        self.kernel_sizes = kernel_sizes

    def get_transform(self, image: np.ndarray) -> Transform:
        do = self._rand_range() < self.prob or self.force_apply
        if do:
            kernel_size = random.choice(self.kernel_sizes)
            return MedianBlurTransform(kernel_size)
        else:
            return NoOpTransform()


class RandomCLAHE(Augmentation):
    """
    Apply CLAHE using a randomly sampled clip limit.
    """

    def __init__(
        self,
        clip_range: tuple[float, float] = (1.0, 4.0),
        tile_grid_size: tuple[int, int] = (8, 8),
        prob: float = 0.5,
    ):
        super().__init__()

        if not isinstance(clip_range, tuple):
            raise TypeError(f"clip_range must be a tuple, got {type(clip_range)}.")

        if len(clip_range) != 2:
            raise ValueError(f"clip_range must contain exactly 2 values, got {len(clip_range)}.")

        if not all(isinstance(value, (int, float)) for value in clip_range):
            raise TypeError(f"clip_range values must be numbers, got {clip_range}.")

        if clip_range[0] <= 0:
            raise ValueError(f"clip_range values must be positive, got {clip_range}.")

        if clip_range[0] > clip_range[1]:
            raise ValueError(f"clip_range must be ordered from minimum to maximum, got {clip_range}.")

        if not isinstance(tile_grid_size, tuple):
            raise TypeError(f"tile_grid_size must be a tuple, got {type(tile_grid_size)}.")

        if len(tile_grid_size) != 2:
            raise ValueError(f"tile_grid_size must contain exactly 2 values, got {len(tile_grid_size)}.")

        if not all(isinstance(value, int) for value in tile_grid_size):
            raise TypeError(f"tile_grid_size values must be ints, got {tile_grid_size}.")

        if not all(value > 0 for value in tile_grid_size):
            raise ValueError(f"tile_grid_size values must be positive, got {tile_grid_size}.")

        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"prob must be between 0.0 and 1.0, got {prob}.")

        self.clip_range = clip_range
        self.tile_grid_size = tile_grid_size
        self.prob = prob

    def get_transform(self, image: np.ndarray) -> Transform:
        do = self._rand_range() < self.prob or self.force_apply
        if do:
            clip_limit = random.uniform(self.clip_range[0], self.clip_range[1])
            return CLAHETransform(clip_limit, self.tile_grid_size)
        else:
            return NoOpTransform()


class RandomContrast(Augmentation):
    def __init__(self, lower: float = 0.5, upper: float = 1.5, prob: float = 0.5):
        super().__init__()

        if not (lower <= upper):
            raise ValueError(f"lower must be less than or equal to upper, got lower={lower}, upper={upper}")

        self.lower = lower
        self.upper = upper
        self.prob = prob

    def get_transform(self, image):
        do = self._rand_range() < self.prob or self.force_apply
        if do:
            alpha = random.uniform(self.lower, self.upper)
            return ContrastTransform(alpha)
        else:
            return NoOpTransform()


class RandomSaturation(Augmentation):
    def __init__(
        self,
        lower: float = 0.5,
        upper: float = 1.5,
        prob: float = 0.5,
    ):
        super().__init__()

        if not (lower <= upper):
            raise ValueError(f"lower must be less than or equal to upper, got lower={lower}, upper={upper}")

        self.lower = lower
        self.upper = upper
        self.prob = prob

    def get_transform(self, image):
        do = self._rand_range() < self.prob or self.force_apply
        if do:
            alpha = random.uniform(self.lower, self.upper)
            return SaturationTransform(alpha)
        else:
            return NoOpTransform()


class RandomHue(Augmentation):
    def __init__(
        self,
        lower: int = -18,
        upper: int = 18,
        prob: float = 0.5,
    ):
        super().__init__()

        if not (lower <= upper):
            raise ValueError(f"lower must be less than or equal to upper, got lower={lower}, upper={upper}")

        self.lower = lower
        self.upper = upper
        self.prob = prob

    def get_transform(self, image):
        do = self._rand_range() < self.prob or self.force_apply
        if do:
            alpha = random.uniform(self.lower, self.upper)
            return HueTransform(alpha)
        else:
            return NoOpTransform()


class AlwaysApply(Augmentation):
    def __init__(self, augmentations: Augmentation | Transform | list[Augmentation | Transform]):
        super().__init__()

        if not isinstance(augmentations, list):
            augmentations = [augmentations]

        # Recursively flatten nested AugmentationList and AlwaysApply to avoid circular nesting.
        def _flatten(augs):
            for aug in augs:
                if isinstance(aug, AugmentationList):
                    yield from _flatten(aug.augs)
                elif isinstance(aug, AlwaysApply):
                    yield from _flatten(aug.augs)
                else:
                    yield aug

        flat_augs = list(_flatten(_transform_to_aug(a) for a in augmentations))

        for aug in flat_augs:
            aug.force_apply = True

        self.augs = flat_augs

    def get_transform(self, *args) -> Transform:
        raise NotImplementedError

    def __call__(self, aug_input) -> Transform:
        tfms = []
        for x in self.augs:
            tfm = x(aug_input)
            tfms.append(tfm)
        return TransformList(tfms)


class OneOf(Augmentation):
    def __init__(self, augmentations: list[Augmentation | Transform], prob: float = 0.5):
        super().__init__()
        self.augmentations = [AlwaysApply(_transform_to_aug(aug)) for aug in augmentations]
        self.prob = prob

    def get_transform(self, *args) -> Transform:
        raise NotImplementedError

    def __call__(self, aug_input) -> Transform:
        do = self.force_apply or self._rand_range() < self.prob
        if do:
            aug = random.choice(self.augmentations)
            return aug(aug_input)
        else:
            return NoOpTransform()
