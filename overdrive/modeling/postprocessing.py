import numpy as np
import torch
import torch.nn.functional as F

from overdrive.data.transforms import Transform
from overdrive.structures import Instances, ROIMasks

__all__ = [
    "detector_postprocess",
    "sem_seg_postprocess",
]


def _get_scale_and_padding(
    transforms: Transform,
    output_height: int,
    output_width: int,
) -> tuple[float, float, float, float]:
    if not isinstance(transforms, Transform):
        raise TypeError("transforms must be a Transform or TransformList")

    points = np.asarray([[0.0, 0.0], [float(output_width), 0.0], [0.0, float(output_height)]], dtype=np.float64)
    mapped = np.asarray(transforms.apply_coords(points))
    if mapped.shape != (3, 2):
        raise ValueError(f"Transforms changed coordinate shape from (3, 2) to {mapped.shape}.")

    pad_x, pad_y = mapped[0]
    scale_x = (mapped[1, 0] - pad_x) / float(output_width)
    scale_y = (mapped[2, 1] - pad_y) / float(output_height)
    cross_terms = (mapped[1, 1] - pad_y, mapped[2, 0] - pad_x)
    if not np.allclose(cross_terms, 0.0, rtol=1e-6, atol=1e-6):
        raise ValueError("Only axis-aligned scale and padding transforms are supported.")

    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("Transform scales must be positive.")

    return tuple(float(v) for v in (scale_x, scale_y, pad_x, pad_y))


def detector_postprocess(
    results: Instances,
    output_height: int,
    output_width: int,
    mask_threshold: float = 0.5,
    transforms: Transform | None = None,
) -> Instances:
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function maps raw detector outputs to the desired output resolution.
    When preprocessing transforms are provided, three reference points are
    used to summarize them as axis-aligned scale and padding parameters. The
    parameters then map predictions back to the original image coordinates.
    Otherwise, the legacy width/height scaling behavior is used.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.
        mask_threshold (float): threshold used to convert ROI masks to bitmasks.
        transforms (Transform, optional): the complete preprocessing transform
            from the original image to the image seen by the detector. It must
            be expressible as positive x/y scaling followed by padding.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    input_image_size = results.image_size
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None

    if output_boxes is None:
        raise ValueError("Predictions must contain boxes!")

    if transforms is None:
        scale_x = input_image_size[1] / output_width
        scale_y = input_image_size[0] / output_height
        pad_x = pad_y = 0.0
    else:
        scale_x, scale_y, pad_x, pad_y = _get_scale_and_padding(transforms, output_height, output_width)

    output_boxes.tensor[:, 0::2] -= pad_x
    output_boxes.tensor[:, 1::2] -= pad_y
    output_boxes.scale(1.0 / scale_x, 1.0 / scale_y)

    if results.has("pred_keypoints"):
        keypoints = results.pred_keypoints
        if keypoints.ndim != 3 or keypoints.shape[2] != 3:
            raise ValueError(f"pred_keypoints must have shape (N, K, 3), got {tuple(keypoints.shape)}.")

        keypoints[:, :, 0] = (keypoints[:, :, 0] - pad_x) / scale_x
        keypoints[:, :, 1] = (keypoints[:, :, 1] - pad_y) / scale_y

    output_boxes.clip(results.image_size)

    keep = output_boxes.nonempty()
    results = results[keep]

    if results.has("pred_masks"):
        if isinstance(results.pred_masks, ROIMasks):
            roi_masks = results.pred_masks
        else:
            # pred_masks is a tensor of shape (N, 1, M, M)
            roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])

        results.pred_masks = roi_masks.to_bitmasks(results.pred_boxes, output_height, output_width, mask_threshold).tensor

    return results


def _validate_sem_seg_content_box(
    left: int,
    top: int,
    right: int,
    bottom: int,
    input_width: int,
    input_height: int,
) -> None:
    if left < 0 or top < 0 or right > input_width or bottom > input_height:
        raise ValueError("The transformed image content is outside the segmentor input size.")
    if right <= left or bottom <= top:
        raise ValueError("The transformed image content must be non-empty.")


def sem_seg_postprocess(
    result: torch.Tensor,
    img_size: tuple[int, int],
    output_height: int,
    output_width: int,
    transforms: Transform | None = None,
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.
        transforms (Transform, optional): the complete preprocessing transform
            from the original image to the image seen by the segmentor. It must
            be expressible as positive x/y scaling followed by padding.
        align_corners (bool): argument of F.interpolate.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]]

    if transforms is not None:
        scale_x, scale_y, pad_x, pad_y = _get_scale_and_padding(transforms, output_height, output_width)
        left = int(round(pad_x))
        top = int(round(pad_y))
        right = left + int(round(scale_x * float(output_width)))
        bottom = top + int(round(scale_y * float(output_height)))
        input_height, input_width = int(img_size[0]), int(img_size[1])
        _validate_sem_seg_content_box(left, top, right, bottom, input_width, input_height)
        result = result[:, top:bottom, left:right]

    result = result.expand(1, -1, -1, -1)
    result = F.interpolate(
        result,
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=align_corners,
    )[0]
    return result
