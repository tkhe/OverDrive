from abc import ABC, abstractmethod

import torch

__all__ = ["BoxCoder"]


class BoxCoder(ABC):
    """
    Abstract base class for box coders.
    """

    @abstractmethod
    def get_deltas(self, src_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def apply_deltas(self, deltas: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
