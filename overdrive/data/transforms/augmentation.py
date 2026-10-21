import inspect
import pprint
from typing import Any

import numpy as np

from .transform import Transform
from .transform import TransformList

__all__ = [
    "AugInput",
    "Augmentation",
    "AugmentationList",
]


def _check_image_dtype(image: np.ndarray):
    if not isinstance(image, np.ndarray):
        raise TypeError(f"[Augmentation] Needs an numpy array, but got a {type(image)}")

    if isinstance(image.dtype, np.integer) and image.dtype != np.uint8:
        raise TypeError(f"[Augmentation] Got image of type {image.dtype}, use uint8 or floating points instead!")

    if image.ndim not in [2, 3]:
        raise ValueError(image.ndim)


def _get_aug_input_args(aug, aug_input) -> list[Any]:
    """
    Get the arguments to be passed to ``aug.get_transform`` from the input ``aug_input``.
    """
    if aug.input_args is None:
        # Decide what attributes are needed automatically
        params = list(inspect.signature(aug.get_transform).parameters.items())
        # The default behavior is: if there is one parameter, then its "image"
        # (work automatically for majority of use cases, and also avoid BC breaking),
        # Otherwise, use the argument names.
        if len(params) == 1:
            names = ("image",)
        else:
            names = []
            for name, param in params:
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    raise TypeError(
                        f"The default implementation of `{type(aug)}.__call__` does not allow `{type(aug)}.get_transform` to use variable-length arguments (*args, **kwargs)!"
                        "If arguments are unknown, reimplement `__call__` instead."
                    )
                names.append(name)
        aug.input_args = tuple(names)

    args = []
    for f in aug.input_args:
        try:
            args.append(getattr(aug_input, f))
        except AttributeError as e:
            # If the argument is not found, check if it has a default value in the signature
            params = dict(inspect.signature(aug.get_transform).parameters)
            if f in params and params[f].default is not inspect.Parameter.empty:
                args.append(params[f].default)
            else:
                raise AttributeError(f"{type(aug)}.get_transform needs input attribute '{f}', but it is not an attribute of {type(aug_input)}!") from e

    return args


class Augmentation:
    """
    Augmentation defines (often random) policies/strategies to generate :class:`Transform`
    from data. It is often used for pre-processing of input data.

    A "policy" that generates a :class:`Transform` may, in the most general case,
    need arbitrary information from input data in order to determine what transforms
    to apply. Therefore, each :class:`Augmentation` instance defines the arguments
    needed by its :meth:`get_transform` method. When called with the positional arguments,
    the :meth:`get_transform` method executes the policy.

    Note that :class:`Augmentation` defines the policies to create a :class:`Transform`,
    but not how to execute the actual transform operations to those data.
    Its :meth:`__call__` method will use :meth:`AugInput.transform` to execute the transform.

    The returned `Transform` object is meant to describe deterministic transformation, which means
    it can be re-applied on associated data, e.g. the geometry of an image and its segmentation
    masks need to be transformed together.
    (If such re-application is not needed, then determinism is not a crucial requirement.)
    """

    input_args: tuple[str] | None = None
    """
    Stores the attribute names needed by :meth:`get_transform`, e.g.  ``("image", "sem_seg")``.
    By default, it is just a tuple of argument names in :meth:`self.get_transform`, which often only
    contain "image". As long as the argument name convention is followed, there is no need for
    users to touch this attribute.
    """

    force_apply: bool = False

    def get_transform(self, *args) -> Transform:
        """
        Execute the policy based on input data, and decide what transform to apply to inputs.

        Args:
            args: Any fixed-length positional arguments. By default, the name of the arguments
                should exist in the :class:`AugInput` to be used.

        Returns:
            Transform: Returns the deterministic transform to apply to the input.

        Examples:

            >>> class MyAug:
            >>>     # if a policy needs to know both image and semantic segmentation
            >>>     def get_transform(image, sem_seg) -> T.Transform:
            >>>         pass
            >>> tfm: Transform = MyAug().get_transform(image, sem_seg)
            >>> new_image = tfm.apply_image(image)
        """
        raise NotImplementedError

    def __call__(self, aug_input) -> Transform:
        """
        Augment the given `aug_input` **in-place**, and return the transform that's used.

        This method will be called to apply the augmentation. In most augmentation, it
        is enough to use the default implementation, which calls :meth:`get_transform`
        using the inputs. But a subclass can overwrite it to have more complicated logic.

        Args:
            aug_input (AugInput): an object that has attributes needed by this augmentation
                (defined by ``self.get_transform``). Its ``transform`` method will be called
                to in-place transform it.

        Returns:
            Transform: the transform that is applied on the input.
        """
        args = _get_aug_input_args(self, aug_input)
        tfm = self.get_transform(*args)

        if not (isinstance(tfm, (Transform, TransformList))):
            raise TypeError(f"{type(self)}.get_transform must return an instance of Transform! Got {type(tfm)} instead.")

        aug_input.transform(tfm)
        return tfm

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self) -> str:
        """
        Produce something like:
        "MyAugmentation(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                if not (param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD):
                    raise ValueError("The default __repr__ doesn't support *args or **kwargs")
                if not (hasattr(self, name)):
                    raise ValueError(f"Attribute {name} not found! Default __repr__ only works if attributes match the constructor.")

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
        except ValueError:
            return super().__repr__()

    def __str__(self) -> str:
        return self.__repr__()


class _TransformToAug(Augmentation):
    def __init__(self, tfm: Transform):
        self.tfm = tfm

    def get_transform(self, *args):
        return self.tfm

    def __repr__(self):
        return repr(self.tfm)

    def __str__(self) -> str:
        return self.__repr__()


def _transform_to_aug(tfm_or_aug):
    """
    Wrap Transform into Augmentation.
    Private, used internally to implement augmentations.
    """
    if not (isinstance(tfm_or_aug, (Transform, Augmentation))):
        raise TypeError(tfm_or_aug)

    if isinstance(tfm_or_aug, Augmentation):
        return tfm_or_aug
    else:
        return _TransformToAug(tfm_or_aug)


class AugmentationList(Augmentation):
    """
    Apply a sequence of augmentations.

    It has ``__call__`` method to apply the augmentations.

    Note that :meth:`get_transform` method is impossible (will throw error if called)
    for :class:`AugmentationList`, because in order to apply a sequence of augmentations,
    the kth augmentation must be applied first, to provide inputs needed by the (k+1)th
    augmentation.
    """

    def __init__(self, augs: list[Augmentation | Transform]):
        """
        Args:
            augs (list[Augmentation or Transform]):
        """
        super().__init__()

        self.augs = [_transform_to_aug(x) for x in augs]

    def __call__(self, aug_input) -> Transform:
        tfms = []
        for x in self.augs:
            tfm = x(aug_input)
            tfms.append(tfm)
        return TransformList(tfms)

    def __repr__(self):
        msgs = [str(x) for x in self.augs]
        return f"AugmentationList[{', '.join(msgs)}]"

    def __str__(self) -> str:
        return self.__repr__()


class AugInput:
    def __init__(
        self,
        image: np.ndarray,
        *,
        boxes: np.ndarray | None = None,
        masks: list[np.ndarray | list[np.ndarray]] | None = None,
        keypoints: np.ndarray | None = None,
        labels: list[int] | None = None,
        keypoint_hflip_indices: np.ndarray | None = None,
        sem_seg: np.ndarray | None = None,
    ):
        _check_image_dtype(image)

        self.image = image
        self.boxes = boxes
        self.masks = masks
        self.keypoints = keypoints
        self.labels = labels
        self.sem_seg = sem_seg
        self.keypoint_hflip_indices = keypoint_hflip_indices

    def transform(self, tfm: Transform):
        """
        In-place transform all attributes of this class.

        By "in-place", it means after calling this method, accessing an attribute such
        as ``self.image`` will return transformed data.
        """
        self.image = tfm.apply_image(self.image)

        if self.boxes is not None:
            self.boxes = tfm.apply_boxes(self.boxes)

        if self.masks is not None:
            self.masks = tfm.apply_masks(self.masks)

        if self.keypoints is not None:
            self.keypoints = tfm.apply_keypoints(self.keypoints)

        if self.labels is not None:
            self.labels = tfm.apply_labels(self.labels)

        if self.sem_seg is not None:
            self.sem_seg = tfm.apply_segmentation(self.sem_seg)

    @classmethod
    def from_dataset_dict(cls, dataset_dict):
        image = dataset_dict["image"]
        boxes = dataset_dict.get("boxes", None)
        masks = dataset_dict.get("masks", None)
        keypoints = dataset_dict.get("keypoints", None)
        labels = dataset_dict.get("labels", None)
        sem_seg = dataset_dict.get("sem_seg", None)
        keypoint_hflip_indices = dataset_dict.get("keypoint_hflip_indices", None)
        return cls(
            image=image,
            boxes=boxes,
            masks=masks,
            keypoints=keypoints,
            labels=labels,
            sem_seg=sem_seg,
            keypoint_hflip_indices=keypoint_hflip_indices,
        )
