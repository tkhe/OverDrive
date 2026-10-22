from abc import ABC
from abc import abstractmethod

__all__ = ["Matcher"]


class Matcher(ABC):
    """
    Abstract base class for matchers.
    """

    @abstractmethod
    def match(self):
        raise NotImplementedError
