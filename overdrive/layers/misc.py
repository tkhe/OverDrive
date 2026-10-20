import torch

from .norm import FrozenBatchNorm2d

__all__ = [
    "FreezableMixin",
    "cat",
    "inverse_sigmoid",
    "make_divisible",
    "permute_to_N_HWA_K",
]


def cat(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    if not isinstance(tensors, (list, tuple)):
        raise TypeError(f"Expected tensors to be a list or tuple, got {type(tensors)!r}")

    if len(tensors) == 1:
        return tensors[0]

    return torch.cat(tensors, dim)


def make_divisible(v: float, divisor: int = 8, min_value: int | None = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by divisor.
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


def permute_to_N_HWA_K(tensor: torch.Tensor, K: int) -> torch.Tensor:
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    if tensor.dim() != 4:
        raise ValueError(f"Expected a 4D tensor, got shape {tuple(tensor.shape)}")

    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)
    return tensor


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-3):
    """
    The inverse function for sigmoid activation function.

    NOTE:
        It might face numberical issues with fp16 small eps.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class FreezableMixin:
    """
    Mixin class that provides freeze() method for nn.Module subclasses.

    This mixin adds the ability to freeze a module by:
    1. Setting all parameters to requires_grad=False
    2. Converting BatchNorm layers to FrozenBatchNorm2d

    Examples::

        >>> import torch.nn as nn
        >>> from overdrive.layers.misc import FreezableMixin
        >>> from overdrive.layers.norm import FrozenBatchNorm2d
        >>> class MyModule(nn.Module, FreezableMixin):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv = nn.Conv2d(3, 8, kernel_size=3)
        ...         self.norm = nn.BatchNorm2d(8)
        >>> model = MyModule()
        >>> model.freeze() is model
        True
        >>> all(not p.requires_grad for p in model.parameters())
        True
        >>> isinstance(model.norm, FrozenBatchNorm2d)
        True
    """

    def freeze(self):
        """
        Make this module not trainable.
        This method sets all parameters to `requires_grad=False`,
        and convert all BatchNorm layers to FrozenBatchNorm

        Returns:
            the module itself
        """
        for p in self.parameters():
            p.requires_grad = False

        FrozenBatchNorm2d.convert_frozen_batch_norm(self)
        return self
