from .faster_rcnn import FasterRCNN

__all__ = ["CascadeMaskRCNN"]


class CascadeMaskRCNN(FasterRCNN):
    """
    support Cascade Mask R-CNN (https://arxiv.org/abs/1906.09756).
    """
