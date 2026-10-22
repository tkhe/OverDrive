import math

import torch
import torch.nn as nn

from overdrive.layers import ROIAlign
from overdrive.layers import ROIPool
from overdrive.layers import cat
from overdrive.structures import Boxes

__all__ = ["ROIPooler"]


def assign_boxes_to_levels(
    box_lists: list[Boxes],
    min_level: int,
    max_level: int,
    canonical_box_size: int,
    canonical_level: int,
):
    """
    Map each box in `box_lists` to a feature map level index and return the assignment
    vector.

    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]): A list of N Boxes or N RotatedBoxes,
            where N is the number of images in the batch.
        min_level (int): Smallest feature map level index. The input is considered index 0,
            the output of stage 1 is index 1, and so.
        max_level (int): Largest feature map level index.
        canonical_box_size (int): A canonical box size in pixels (sqrt(box area)).
        canonical_level (int): The feature map level index on which a canonically-sized box
            should be placed.

    Returns:
        A tensor of length M, where M is the total number of boxes aggregated over all
            N batch images. The memory layout corresponds to the concatenation of boxes
            from all images. Each element is the feature map index, as an offset from
            `self.min_level`, for the corresponding box (so value i means the box is at
            `self.min_level + i`).
    """
    box_sizes = torch.sqrt(cat([boxes.area() for boxes in box_lists]))
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-8))
    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level


def _convert_boxes_to_pooler_format(boxes: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    sizes = sizes.to(device=boxes.device)
    indices = torch.repeat_interleave(torch.arange(len(sizes), dtype=boxes.dtype, device=boxes.device), sizes)
    return cat([indices[:, None], boxes], dim=1)


def convert_boxes_to_pooler_format(box_lists: list[Boxes]):
    """
    Convert all boxes in `box_lists` to the low-level format used by ROI pooling ops
    (see description under Returns).

    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]):
            A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.

    Returns:
        When input is list[Boxes]:
            A tensor of shape (M, 5), where M is the total number of boxes aggregated over all
            N batch images.
            The 5 columns are (batch index, x0, y0, x1, y1), where batch index
            is the index in [0, N) identifying which batch image the box with corners at
            (x0, y0, x1, y1) comes from.
        When input is list[RotatedBoxes]:
            A tensor of shape (M, 6), where M is the total number of boxes aggregated over all
            N batch images.
            The 6 columns are (batch index, x_ctr, y_ctr, width, height, angle_degrees),
            where batch index is the index in [0, N) identifying which batch image the
            rotated box (x_ctr, y_ctr, width, height, angle_degrees) comes from.
    """
    boxes = torch.cat([x.tensor for x in box_lists], dim=0)
    sizes = torch.as_tensor([len(x) for x in box_lists])
    return _convert_boxes_to_pooler_format(boxes, sizes)


def _create_zeros(
    batch_target: torch.Tensor | None,
    channels: int,
    height: int,
    width: int,
    like_tensor: torch.Tensor,
) -> torch.Tensor:
    batches = batch_target.shape[0] if batch_target is not None else 0
    sizes = (batches, channels, height, width)
    return torch.zeros(sizes, dtype=like_tensor.dtype, device=like_tensor.device)


class ROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
        self,
        output_size,
        scales,
        sampling_ratio,
        pooler_type,
        canonical_box_size=224,
        canonical_level=4,
    ):
        """
        Args:
            output_size (int, tuple[int] or list[int]): output size of the pooled region,
                e.g., 14 x 14. If tuple or list is given, the length must be 2.
            scales (list[float]): The scale for each low-level pooling op relative to
                the input image. For a feature map with stride s relative to the input
                image, scale is defined as 1/s. The stride must be power of 2.
                When there are multiple scales, they must form a pyramid, i.e. they must be
                a monotically decreasing geometric sequence with a factor of 1/2.
            sampling_ratio (int): The `sampling_ratio` parameter for the ROIAlign op.
            pooler_type (string): Name of the type of pooling operation that should be applied.
                For instance, "ROIPool" or "ROIAlignV2".
            canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
                is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
                pre-training).
            canonical_level (int): The feature map level index from which a canonically-sized box
                should be placed. The default is defined as level 4 (stride=16) in the FPN paper,
                i.e., a box of size 224x224 will be placed on the feature with stride=16.
                The box placement for all boxes will be determined from their sizes w.r.t
                canonical_box_size. For example, a box whose area is 4x that of a canonical box
                should be used to pool features from feature level ``canonical_level+1``.

                Note that the actual input feature maps given to this module may not have
                sufficiently many levels for the input boxes. If the boxes are too large or too
                small for the input feature maps, the closest level will be used.
        """
        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        if len(output_size) != 2:
            raise ValueError(f"Expected output_size to contain exactly 2 values, but got {len(output_size)}")
        if not isinstance(output_size[0], int) or not isinstance(output_size[1], int):
            raise TypeError(f"Expected both output_size values to be integers, but got {output_size}")

        self.output_size = output_size

        level_poolers = []
        if pooler_type == "ROIAlign":
            level_poolers = [ROIAlign(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False) for scale in scales]
        elif pooler_type == "ROIAlignV2":
            level_poolers = [ROIAlign(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True) for scale in scales]
        elif pooler_type == "ROIPool":
            level_poolers = [ROIPool(output_size, spatial_scale=scale) for scale in scales]
        else:
            raise ValueError(f"Unknown pooler type: {pooler_type}")

        self.level_poolers = nn.ModuleList(level_poolers)

        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.
        min_level = -(math.log2(scales[0]))
        max_level = -(math.log2(scales[-1]))

        if not math.isclose(min_level, int(min_level)) or not math.isclose(max_level, int(max_level)):
            raise ValueError(f"Expected feature map scales to correspond to power-of-two strides, but got {scales}")

        self.min_level = int(min_level)
        self.max_level = int(max_level)

        if len(scales) != self.max_level - self.min_level + 1:
            raise ValueError(f"Expected {self.max_level - self.min_level + 1} feature map scales, but got {len(scales)}: {scales}")
        if not (0 <= self.min_level <= self.max_level):
            raise ValueError(f"Expected 0 <= min_level <= max_level, but got min_level={self.min_level}, max_level={self.max_level}")
        if canonical_box_size <= 0:
            raise ValueError(f"Expected canonical_box_size to be positive, but got {canonical_box_size}")

        self.canonical_level = canonical_level
        self.canonical_box_size = canonical_box_size

    def forward(self, x: list[torch.Tensor], box_lists: list[Boxes]):
        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Boxes] | list[RotatedBoxes]):
                A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.

        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        num_level_assignments = len(self.level_poolers)

        if not isinstance(x, list) or not isinstance(box_lists, list):
            raise TypeError(f"Expected x and box_lists to be lists, but got {type(x)} and {type(box_lists)}")

        if len(x) != num_level_assignments:
            raise ValueError(f"Expected {num_level_assignments} input feature maps, but got {len(x)}")

        if len(box_lists) != x[0].size(0):
            raise ValueError(f"Expected {x[0].size(0)} box lists to match the batch size, but got {len(box_lists)}")

        if len(box_lists) == 0:
            return _create_zeros(None, x[0].shape[1], *self.output_size, x[0])

        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        level_assignments = assign_boxes_to_levels(box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level)

        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        output = _create_zeros(pooler_fmt_boxes, num_channels, output_size, output_size, x[0])

        for level, pooler in enumerate(self.level_poolers):
            inds = (level_assignments == level).nonzero(as_tuple=True)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            # Use index_put_ instead of advance indexing, to avoid pytorch/issues/49852
            output.index_put_((inds,), pooler(x[level], pooler_fmt_boxes_level))

        return output
