from abc import ABC
from abc import abstractmethod

__all__ = ["Sampler"]


class Sampler(ABC):
    """
    Base class for all samplers.
    """

    @abstractmethod
    def sample(self):
        raise NotImplementedError
