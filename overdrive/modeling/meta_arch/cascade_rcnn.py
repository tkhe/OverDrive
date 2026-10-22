from .faster_rcnn import FasterRCNN

__all__ = ["CascadeRCNN"]


class CascadeRCNN(FasterRCNN):
    """
    support Cascade R-CNN (https://arxiv.org/abs/1712.00726).
    """
