import inspect
import pprint
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from typing import TypeVar

import cv2
import numpy as np

__all__ = [
    "BlurTransform",
    "BrightnessTransform",
    "CLAHETransform",
    "ContrastTransform",
    "CropTransform",
    "GrayscaleTransform",
    "HorizontalFlipTransform",
    "HueTransform",
    "MedianBlurTransform",
    "NoOpTransform",
    "PadTransform",
    "ResizeTransform",
    "RotationTransform",
    "SaturationTransform",
    "Transform",
    "TransformList",
    "VerticalFlipTransform",
]

INTERPOLATION_METHOD = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
}


class Transform(ABC):
    @abstractmethod
    def apply_image(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return self.apply_image(segmentation)

    def apply_boxes(self, boxes: np.ndarray) -> np.ndarray:
        # Indexes of converting (x0, y0, x1, y1) box into 4 coordinates of
        # ([x0, y0], [x1, y0], [x0, y1], [x1, y1]).
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(boxes).reshape(-1, 4)[:, idxs].reshape(-1, 2)
        coords = self.apply_coords(coords).reshape((-1, 4, 2))
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        trans_boxes = np.concatenate((minxy, maxxy), axis=1)
        return trans_boxes

    def apply_polygons(self, polygons: list[np.ndarray]) -> list[np.ndarray]:
        return [self.apply_coords(p) for p in polygons]

    def apply_masks(self, masks: list[np.ndarray | list[np.ndarray]]) -> list[np.ndarray | list[np.ndarray]]:
        if not isinstance(masks, list):
            raise TypeError(f"Expected masks to be a list of instances, got {type(masks)}.")

        transformed_masks = []
        for mask in masks:
            if isinstance(mask, np.ndarray):
                if mask.ndim != 2:
                    raise ValueError(f"Expected bitmask instance with 2 dimensions, got shape {mask.shape}.")

                transformed_masks.append(self.apply_segmentation(mask))
            elif isinstance(mask, list):
                transformed_masks.append(self.apply_polygons(mask))
            else:
                raise TypeError(f"Expected each mask to be either a bitmask (ndarray) or a polygon (list of ndarray), got {type(mask)}.")

        return transformed_masks

    def apply_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        if keypoints.ndim != 3 or keypoints.shape[2] != 3:
            raise ValueError(f"Expected keypoints with shape (N, K, 3), got {keypoints.shape}.")

        N, K, _ = keypoints.shape
        keypoints = keypoints.astype("float64").reshape(-1, 3)
        keypoints[:, :2] = self.apply_coords(keypoints[:, :2])

        keypoints = keypoints.reshape(N, K, 3)
        keypoints[keypoints[:, :, 2] == 0] = 0
        return keypoints

    def apply_labels(self, labels: list[int]) -> list[int]:
        return labels

    @classmethod
    def register_type(cls, data_type: str, func: Callable | None = None):
        """
        Register the given function as a handler that this transform will use
        for a specific data type.

        Args:
            data_type (str): the name of the data type (e.g., box)
            func (callable): takes a transform and a data, returns the
                transformed data.

        Examples:
            >>> from overdrive.data.transforms.transform import HorizontalFlipTransform
            >>> def direct_handler(transform, data):
            ...     return transform.width, data
            >>> HorizontalFlipTransform.register_type("direct_voxel", direct_handler)
            >>> transform = HorizontalFlipTransform(width=10)
            >>> transform.apply_direct_voxel(["a", 2])
            (10, ['a', 2])

            >>> @HorizontalFlipTransform.register_type("decorated_voxel")
            ... def decorated_handler(transform, data):
            ...     return transform.width, data
            >>> transform.apply_decorated_voxel({"a": 2})
            (10, {'a': 2})
        """
        # the decorator style
        if func is None:

            def wrapper(decorated_func):
                if decorated_func is None:
                    raise ValueError("The decorated function passed to register_type must not be None.")

                cls.register_type(data_type, decorated_func)
                return decorated_func

            return wrapper

        if not callable(func):
            raise TypeError(f"You can only register a callable to a Transform. Got {func} instead.")

        argspec = inspect.getfullargspec(func)
        if len(argspec.args) != 2:
            raise TypeError(f"You can only register a function that takes two positional arguments to a Transform! Got a function with spec {str(argspec)}")

        setattr(cls, "apply_" + data_type, func)

    def inverse(self) -> "Transform":
        """
        Create a transform that inverts the geometric changes (i.e. change of
        coordinates) of this transform.

        Note that the inverse is meant for geometric changes only.
        The inverse of photometric transforms that do not change coordinates
        is defined to be a no-op, even if they may be invertible.

        Returns:
            Transform:
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        Produce something like:
        "MyTransform(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                if param.kind == param.VAR_POSITIONAL or param.kind == param.VAR_KEYWORD:
                    raise ValueError("The default __repr__ doesn't support *args or **kwargs")
                if not hasattr(self, name):
                    raise AttributeError(f"Attribute {name} not found! Default __repr__ only works if attributes match the constructor.")

                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue

                attr_str = pprint.pformat(attr)
                if "\n" in attr_str:
                    # don't show it if pformat decides to use >1 lines
                    attr_str = "..."
                argstr.append(f"{name}={attr_str}")
            return f"{classname}({', '.join(argstr)})"
        except (ValueError, AttributeError):
            return super().__repr__()


_T = TypeVar("_T")


class TransformList(Transform):
    """
    Maintain a list of transform operations which will be applied in sequence.

    Attributes:
        transforms (list[Transform])
    """

    def __init__(self, transforms: list[Transform]):
        """
        Args:
            transforms (list[Transform]): list of transforms to perform.
        """
        super().__init__()

        # "Flatten" the list so that TransformList do not recursively contain TransfomList.
        # The additional hierarchy does not change semantic of the class, but cause extra
        # complexities in e.g, telling whether a TransformList contains certain Transform
        def _flatten(tfms):
            for t in tfms:
                if not isinstance(t, Transform):
                    raise TypeError(f"TransformList requires a list of Transform. Got type {type(t)}!")
                if isinstance(t, TransformList):
                    yield from _flatten(t.transforms)
                else:
                    yield t

        self.transforms = list(_flatten(transforms))

    def _apply(self, x: _T, meth: str) -> _T:
        """
        Apply the transforms on the input.

        Args:
            x: input to apply the transform operations.
            meth (str): meth.

        Returns:
            x: after apply the transformation.
        """
        for t in self.transforms:
            x = getattr(t, meth)(x)
        return x

    def __getattribute__(self, name: str):
        # use __getattribute__ to win priority over any registered dtypes
        if name.startswith("apply_"):
            return lambda x: self._apply(x, name)

        return super().__getattribute__(name)

    def __add__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.

        Returns:
            TransformList: list of transforms.
        """
        others = other.transforms if isinstance(other, TransformList) else [other]
        return TransformList(self.transforms + others)

    def __iadd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.

        Returns:
            TransformList: list of transforms.
        """
        others = other.transforms if isinstance(other, TransformList) else [other]
        self.transforms.extend(others)
        return self

    def __radd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.

        Returns:
            TransformList: list of transforms.
        """
        others = other.transforms if isinstance(other, TransformList) else [other]
        return TransformList(others + self.transforms)

    def __len__(self) -> int:
        """
        Returns:
            Number of transforms contained in the TransformList.
        """
        return len(self.transforms)

    def __getitem__(self, idx) -> Transform:
        return self.transforms[idx]

    def inverse(self) -> "TransformList":
        """
        Invert each transform in reversed order.
        """
        return TransformList([x.inverse() for x in self.transforms[::-1]])

    def __repr__(self) -> str:
        msgs = [str(t) for t in self.transforms]
        return f"TransformList[{', '.join(msgs)}]"

    def __str__(self) -> str:
        return self.__repr__()

    # The actual implementations are provided in __getattribute__.
    # But abstract methods need to be declared here.
    def apply_coords(self, x):
        raise NotImplementedError

    def apply_image(self, x):
        raise NotImplementedError


class HorizontalFlipTransform(Transform):
    """
    Perform horizontal flip.
    """

    def __init__(self, width: int, keypoint_hflip_indices: np.ndarray | None = None):
        super().__init__()

        self.width = width
        self.keypoint_hflip_indices = keypoint_hflip_indices

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).

        Args:
            image (ndarray): of shape HxW or HxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the flipped image(s).
        """
        return image[:, ::-1]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).

        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, y)`, not
            `(W - 1 - x, H - 1 - y)`.
        """
        coords[:, 0] = self.width - coords[:, 0]
        return coords

    def apply_keypoints(self, keypoints):
        keypoints = super().apply_keypoints(keypoints)
        if self.keypoint_hflip_indices is None:
            raise ValueError("keypoint_hflip_indices is required for flipping keypoints!")
        if self.keypoint_hflip_indices is not None:
            keypoints = keypoints[:, self.keypoint_hflip_indices, :]
        return keypoints

    def inverse(self) -> Transform:
        """
        The inverse is to flip again
        """
        return self


class VerticalFlipTransform(Transform):
    """
    Perform vertical flip.
    """

    def __init__(self, height: int):
        super().__init__()

        self.height = height

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).

        Args:
            image (ndarray): of shape HxW or HxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: the flipped image(s).
        """
        return image[::-1, :]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).

        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(x, H - y)`, not
            `(W - 1 - x, H - 1 - y)`.
        """
        coords[:, 1] = self.height - coords[:, 1]
        return coords

    def inverse(self) -> Transform:
        """
        The inverse is to flip again
        """
        return self


class NoOpTransform(Transform):
    """
    A transform that does nothing.
    """

    def __init__(self):
        super().__init__()

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        return image

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def inverse(self) -> Transform:
        return self

    def __getattr__(self, name: str):
        if name.startswith("apply_"):
            return lambda x: x

        raise AttributeError(f"NoOpTransform object has no attribute {name}")


class PadTransform(Transform):
    def __init__(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
        origin_w: int | None = None,
        origin_h: int | None = None,
        pad_value: int = 0,
        seg_pad_value: int = 255,
    ):
        super().__init__()

        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.origin_w = origin_w
        self.origin_h = origin_h
        self.pad_value = pad_value
        self.seg_pad_value = seg_pad_value

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        pad_value = self.pad_value
        if np.isscalar(pad_value) and image.ndim > 2:
            pad_value = (pad_value,) * image.shape[2]

        ret = cv2.copyMakeBorder(
            image,
            self.top,
            self.bottom,
            self.left,
            self.right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_value,
        )
        if len(image.shape) > 2 and image.shape[2] == 1 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        coords[:, 0] += self.left
        coords[:, 1] += self.top
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        seg_pad_value = self.seg_pad_value
        if np.isscalar(seg_pad_value) and segmentation.ndim > 2:
            seg_pad_value = (seg_pad_value,) * segmentation.shape[2]

        ret = cv2.copyMakeBorder(
            segmentation,
            self.top,
            self.bottom,
            self.left,
            self.right,
            borderType=cv2.BORDER_CONSTANT,
            value=seg_pad_value,
        )
        if segmentation.ndim > 2 and segmentation.shape[2] == 1 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def inverse(self):
        if self.origin_h is None or self.origin_w is None:
            raise ValueError("origin_w, origin_h are required for PadTransform to be invertible!")

        neww = self.origin_w + self.left + self.right
        newh = self.origin_h + self.top + self.bottom
        return CropTransform(self.left, self.top, self.origin_w, self.origin_h, origin_w=neww, origin_h=newh)


class CropTransform(Transform):
    def __init__(
        self,
        x0: int,
        y0: int,
        w: int,
        h: int,
        origin_w: int | None = None,
        origin_h: int | None = None,
    ):
        super().__init__()

        self.x0 = x0
        self.y0 = y0
        self.w = w
        self.h = h
        self.origin_w = origin_w
        self.origin_h = origin_h

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(image[self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w])

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        coords[:, 0] -= self.x0
        coords[:, 1] -= self.y0
        return coords

    def apply_boxes(self, boxes: np.ndarray) -> np.ndarray:
        boxes = super().apply_boxes(boxes)

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.h)
        return boxes

    def apply_polygons(self, polygons: list[np.ndarray]) -> list[np.ndarray]:
        import shapely.geometry as geometry

        # Create a window that will be used to crop
        crop_box = geometry.box(self.x0, self.y0, self.x0 + self.w, self.y0 + self.h).buffer(0.0)

        cropped_polygons = []

        for polygon in polygons:
            polygon = geometry.Polygon(polygon).buffer(0.0)
            # polygon must be valid to perform intersection.
            if not polygon.is_valid:
                continue

            cropped = polygon.intersection(crop_box)
            if cropped.is_empty:
                continue

            if isinstance(cropped, geometry.collection.BaseMultipartGeometry):
                cropped = cropped.geoms
            else:
                cropped = [cropped]
            # one polygon may be cropped to multiple ones
            for poly in cropped:
                # It could produce lower dimensional objects like lines or
                # points, which we want to ignore
                if not isinstance(poly, geometry.Polygon) or not poly.is_valid:
                    continue
                coords = np.asarray(poly.exterior.coords)
                # NOTE This process will produce an extra identical vertex at
                # the end. So we remove it. This is tested by
                # `tests/test_data_transform.py`
                cropped_polygons.append(coords[:-1])
        return [self.apply_coords(p) for p in cropped_polygons]

    def inverse(self) -> Transform:
        if self.origin_w is None or self.origin_h is None:
            raise ValueError("origin_w, origin_h are required for CropTransform to be invertible!")

        return PadTransform(
            self.x0,
            self.y0,
            self.origin_w - self.x0 - self.w,
            self.origin_h - self.y0 - self.h,
            origin_w=self.w,
            origin_h=self.h,
        )


class ResizeTransform(Transform):
    def __init__(
        self,
        w,
        h,
        new_w,
        new_h,
        interpolation: str | None = "bilinear",
    ):
        super().__init__()

        self.w = w
        self.h = h
        self.new_w = new_w
        self.new_h = new_h
        self.interpolation = interpolation

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] != (self.h, self.w):
            raise ValueError(f"Expected image of shape ({self.h}, {self.w}), got {image.shape[:2]}.")

        interp = INTERPOLATION_METHOD[self.interpolation]
        ret = cv2.resize(image, (self.new_w, self.new_h), interpolation=interp)
        if len(image.shape) > 2 and image.shape[2] == 1 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        coords[:, 0] = coords[:, 0] * (self.new_w / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h / self.h)
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        if segmentation.shape[:2] != (self.h, self.w):
            raise ValueError(f"Expected segmentation of shape ({self.h}, {self.w}), got {segmentation.shape[:2]}.")

        ret = cv2.resize(segmentation, (self.new_w, self.new_h), interpolation=cv2.INTER_NEAREST)
        if len(segmentation.shape) > 2 and segmentation.shape[2] == 1 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def inverse(self) -> Transform:
        return ResizeTransform(self.new_w, self.new_h, self.w, self.h, self.interpolation)


class RotationTransform(Transform):
    """
    Rotate image and coordinates around the center of the image.
    """

    def __init__(
        self,
        h: int,
        w: int,
        angle: float,
        expand: bool = True,
        center: tuple | None = None,
        interpolation: str = "bilinear",
    ):
        """
        Args:
            h (int): Original image height.
            w (int): Original image width.
            angle (float): Rotation angle in degrees (counter-clockwise).
            expand (bool): If True, expands the output image to fit the entire
                rotated image. If False, keeps the output the same size as the
                input, cropping corners.
            center (tuple): (cx, cy) rotation center. Defaults to image center.
            interpolation (str): Interpolation method for image rotation.
        """
        super().__init__()

        self.h = h
        self.w = w
        self.angle = angle
        self.expand = expand
        self.center = center if center is not None else (w / 2.0, h / 2.0)
        self.interpolation = interpolation

        self.rot_mat = cv2.getRotationMatrix2D(self.center, angle, 1.0)

        if expand:
            cos = np.abs(self.rot_mat[0, 0])
            sin = np.abs(self.rot_mat[0, 1])
            self.new_w = int(np.ceil(h * sin + w * cos))
            self.new_h = int(np.ceil(h * cos + w * sin))
            self.rot_mat[0, 2] += (self.new_w - w) / 2.0
            self.rot_mat[1, 2] += (self.new_h - h) / 2.0
        else:
            self.new_w = w
            self.new_h = h

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] != (self.h, self.w):
            raise ValueError(f"Expected image of shape ({self.h}, {self.w}), got {image.shape[:2]}.")

        interp = INTERPOLATION_METHOD[self.interpolation]
        ret = cv2.warpAffine(image, self.rot_mat, (self.new_w, self.new_h), flags=interp)
        if len(image.shape) > 2 and image.shape[2] == 1 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        # rot_mat is 2x3, apply as: [x', y'] = rot_mat @ [x, y, 1]^T
        ones = np.ones((coords.shape[0], 1), dtype=coords.dtype)
        coords_h = np.hstack([coords, ones])  # (N, 3)
        coords_rot = coords_h @ self.rot_mat.T  # (N, 2)
        coords[:, :] = coords_rot
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        if segmentation.shape[:2] != (self.h, self.w):
            raise ValueError(f"Expected segmentation of shape ({self.h}, {self.w}), got {segmentation.shape[:2]}.")

        ret = cv2.warpAffine(segmentation, self.rot_mat, (self.new_w, self.new_h), flags=cv2.INTER_NEAREST)
        if len(segmentation.shape) > 2 and segmentation.shape[2] == 1 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def inverse(self) -> Transform:
        return RotationTransform(
            self.new_h,
            self.new_w,
            -self.angle,
            expand=self.expand,
            center=(self.new_w / 2.0, self.new_h / 2.0),
            interpolation=self.interpolation,
        )


class GrayscaleTransform(Transform):
    """
    Convert image to grayscale while preserving the number of channels.
    This is a photometric transform that does not alter coordinates.
    """

    def __init__(self):
        super().__init__()

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.shape[2] == 1:
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.stack([gray] * image.shape[2], axis=-1)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def inverse(self) -> Transform:
        return NoOpTransform()


class BlurTransform(Transform):
    """
    Blur an image with a normalized box filter.

    This is a photometric transform that does not alter coordinates or
    segmentation annotations.
    """

    def __init__(self, kernel_size: int):
        super().__init__()

        if not isinstance(kernel_size, int):
            raise TypeError(f"kernel_size must be an int, got {type(kernel_size)}.")

        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}.")

        self.kernel_size = kernel_size

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        ret = cv2.blur(image, (self.kernel_size, self.kernel_size))
        if image.ndim == 3 and image.shape[2] == 1 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def inverse(self) -> Transform:
        return NoOpTransform()


class MedianBlurTransform(Transform):
    """
    Blur an image with a median filter.

    This is a photometric transform that does not alter coordinates or
    segmentation annotations.
    """

    def __init__(self, kernel_size: int):
        super().__init__()

        if not isinstance(kernel_size, int):
            raise TypeError(f"kernel_size must be an int, got {type(kernel_size)}.")

        if kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be an odd integer greater than 1, got {kernel_size}.")

        self.kernel_size = kernel_size

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        ret = cv2.medianBlur(image, self.kernel_size)
        if image.ndim == 3 and image.shape[2] == 1 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def inverse(self) -> Transform:
        return NoOpTransform()


class CLAHETransform(Transform):
    """
    Apply contrast-limited adaptive histogram equalization to an image.

    This is a photometric transform that does not alter coordinates or
    segmentation annotations.
    """

    def __init__(self, clip_limit: float, tile_grid_size: tuple[int, int]):
        super().__init__()

        if not isinstance(clip_limit, (int, float)):
            raise TypeError(f"clip_limit must be a number, got {type(clip_limit)}.")

        if clip_limit <= 0:
            raise ValueError(f"clip_limit must be positive, got {clip_limit}.")

        if not isinstance(tile_grid_size, tuple):
            raise TypeError(f"tile_grid_size must be a tuple, got {type(tile_grid_size)}.")

        if len(tile_grid_size) != 2 or not all(isinstance(value, int) and value > 0 for value in tile_grid_size):
            raise ValueError(f"tile_grid_size must contain 2 positive ints, got {tile_grid_size}.")

        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

        if image.ndim == 2:
            return clahe.apply(image)
        if image.shape[2] == 1:
            return clahe.apply(image[:, :, 0])[:, :, np.newaxis]

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def inverse(self) -> Transform:
        return NoOpTransform()


class BrightnessTransform(Transform):
    """
    Adjust image brightness by a multiplicative factor.
    output = image * factor, clipped to valid range.
    """

    def __init__(self, factor: float):
        """
        Args:
            factor (float): Brightness multiplier. 1.0 means no change,
                <1.0 darkens, >1.0 brightens.
        """
        super().__init__()

        self.factor = factor

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            table = np.clip(np.arange(256) * self.factor, 0, 255).astype(np.uint8)
            ret = cv2.LUT(image, table)
            if len(image.shape) > 2 and image.shape[2] == 1 and ret.ndim == 2:
                ret = ret[:, :, np.newaxis]
            return ret

        return np.clip(image * self.factor, 0.0, 1.0).astype(image.dtype)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def inverse(self) -> Transform:
        return NoOpTransform()


class ContrastTransform(Transform):
    """
    Adjust image contrast by blending with the mean over all pixels and channels.
    output = mean * (1 - factor) + image * factor
    """

    def __init__(self, factor: float):
        """
        Args:
            factor (float): Contrast multiplier. 1.0 means no change,
                0.0 gives a uniform gray image, >1.0 increases contrast.
        """
        super().__init__()

        self.factor = factor

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        mean = image.mean()
        if image.dtype == np.uint8:
            table = np.clip(mean + (np.arange(256) - mean) * self.factor, 0, 255).astype(np.uint8)
            ret = cv2.LUT(image, table)
            if len(image.shape) > 2 and image.shape[2] == 1 and ret.ndim == 2:
                ret = ret[:, :, np.newaxis]
            return ret

        return np.clip(mean + (image - mean) * self.factor, 0.0, 1.0).astype(image.dtype)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def inverse(self) -> Transform:
        return NoOpTransform()


class SaturationTransform(Transform):
    """
    Adjust image saturation by blending with the grayscale version.
    output = gray * (1 - factor) + image * factor
    """

    def __init__(self, factor: float):
        """
        Args:
            factor (float): Saturation multiplier. 1.0 means no change,
                0.0 gives grayscale, >1.0 increases saturation.
        """
        super().__init__()

        self.factor = factor

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.stack([gray] * image.shape[2], axis=-1)

        if image.dtype == np.uint8:
            return cv2.addWeighted(image, self.factor, gray, 1.0 - self.factor, 0)

        return np.clip(gray * (1.0 - self.factor) + image * self.factor, 0.0, 1.0).astype(image.dtype)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def inverse(self) -> Transform:
        return NoOpTransform()


class HueTransform(Transform):
    """
    Shift the hue channel of the image by a given angle (in degrees).
    """

    def __init__(self, delta: float):
        """
        Args:
            delta (float): Hue shift in degrees, range [-180, 180].
        """
        super().__init__()

        if not (-180 <= delta <= 180):
            raise ValueError(f"delta must be in [-180, 180], got {delta}.")

        self.delta = delta

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            return image

        if image.dtype == np.uint8:
            h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
            table = ((np.arange(256) + int(self.delta / 2)) % 180).astype(np.uint8)
            return cv2.cvtColor(cv2.merge([cv2.LUT(h, table), s, v]), cv2.COLOR_HSV2BGR)

        # float image: H is [0, 360)
        hsv = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0] + self.delta) % 360.0
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(image.dtype)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def inverse(self) -> Transform:
        return NoOpTransform()
