from abc import ABC
from abc import abstractmethod

import torch.nn as nn

__all__ = ["ROIHead"]


class ROIHead(nn.Module, ABC):
    """
    Base class for all ROI heads.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        raise NotImplementedError

    @property
    def with_box(self) -> bool:
        return hasattr(self, "box_head") and self.box_head is not None

    @property
    def with_mask(self) -> bool:
        return hasattr(self, "mask_head") and self.mask_head is not None

    @property
    def with_keypoint(self) -> bool:
        return hasattr(self, "keypoint_head") and self.keypoint_head is not None
