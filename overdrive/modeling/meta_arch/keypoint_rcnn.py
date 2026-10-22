from .faster_rcnn import FasterRCNN

__all__ = ["KeypointRCNN"]


class KeypointRCNN(FasterRCNN):
    """
    support Keypoint R-CNN (https://arxiv.org/abs/1703.06870).
    """
