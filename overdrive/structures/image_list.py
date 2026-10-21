from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as to_2tuple

__all__ = ["ImageList"]


class ImageList:
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size.
    The original sizes of each image is stored in `image_sizes`.

    Attributes:
        image_sizes (list[tuple[int, int]]): each tuple is (h, w).
    """

    def __init__(self, tensor: torch.Tensor, image_sizes: list[tuple[int, int]]):
        """
        Args:
            tensor (Tensor): of shape (N, H, W) or (N, C_1, ..., C_K, H, W) where K >= 1
            image_sizes (list[tuple[int, int]]): Each tuple is (h, w). It can
                be smaller than (H, W) due to padding.
        """
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx) -> torch.Tensor:
        """
        Access the individual image in its original size.

        Args:
            idx: int or slice

        Returns:
            Tensor: an image of shape (H, W) or (C_1, ..., C_K, H, W) where K >= 1
        """
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1]]

    def to(self, *args: Any, **kwargs: Any) -> "ImageList":
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @staticmethod
    def from_tensors(
        tensors: list[torch.Tensor],
        padding_constraints: dict[str, int | tuple[int, int]] | None = None,
        pad_value: float = 0.0,
    ) -> "ImageList":
        """
        Args:
            tensors: a tuple or list of `torch.Tensor`, each of shape (Hi, Wi) or
                (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded
                to the same shape with `pad_value`.
            padding_constraints (optional[Dict]): If given, it would follow the format as
                {"size_divisibility": int, "pad_size": int or tuple[int, int]}. If `size_divisibility > 0`,
                add padding to ensure the common height and width is divisible by `size_divisibility`.
                `pad_size` sets a fixed (height, width), using the same value for both
                dimensions when an int is given. Fixed padding applies when both dimensions are positive.
            pad_value (float): value to pad.


        Returns:
            an `ImageList`.
        """
        if not isinstance(tensors, (tuple, list)):
            raise TypeError(f"ImageList.from_tensors expects a list or tuple of tensors. Got {type(tensors)}.")

        if len(tensors) == 0:
            raise ValueError("ImageList.from_tensors expects a non-empty list or tuple of tensors.")

        for t in tensors:
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"ImageList.from_tensors expects each item to be a torch.Tensor. Got {type(t)}.")
            if t.shape[:-2] != tensors[0].shape[:-2]:
                raise ValueError(f"All tensors must have matching leading dimensions. Got {t.shape} and {tensors[0].shape}.")

        image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]
        max_size = [max(size[dim] for size in image_sizes) for dim in range(2)]

        if padding_constraints is not None:
            pad_size = to_2tuple(padding_constraints.get("pad_size", 0))
            if pad_size[0] > 0 and pad_size[1] > 0:
                # pad to fixed size.
                max_size = list(pad_size)

            size_divisibility = padding_constraints.get("size_divisibility", 0)
            if size_divisibility > 1:
                stride = size_divisibility
                # the last two dims are H,W, both subject to divisibility requirement
                max_size = [(size + stride - 1) // stride * stride for size in max_size]

        if len(tensors) == 1:
            # F.pad is faster for a single image. For multiple images, preallocating
            # the batch and copying each valid image region is faster and uses less memory.
            image_size = image_sizes[0]
            padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
            batched_imgs = F.pad(tensors[0], padding_size, value=pad_value).unsqueeze_(0)
        else:
            batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + max_size
            batched_imgs = tensors[0].new_full(batch_shape, pad_value)
            batched_imgs = batched_imgs.to(tensors[0])
            for i, img in enumerate(tensors):
                batched_imgs[i, ..., : img.shape[-2], : img.shape[-1]].copy_(img)

        return ImageList(batched_imgs.contiguous(), image_sizes)
