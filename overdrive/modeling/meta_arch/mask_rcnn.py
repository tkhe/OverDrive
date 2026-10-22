from .faster_rcnn import FasterRCNN

__all__ = ["MaskRCNN"]


class MaskRCNN(FasterRCNN):
    """
    support Mask R-CNN (https://arxiv.org/abs/1703.06870).
    """
