from functools import wraps

import torch.nn as nn

from .conv2d import Conv2d
from .conv2d import DeformConv2dV1
from .conv2d import DeformConv2dV2
from .conv2d import DepthwiseSeparableConv2d

__all__ = [
    "bias_fill",
    "c2_msra_fill",
    "c2_xavier_fill",
    "constant_fill",
    "dirac_fill",
    "eye_fill",
    "init_conv2d",
    "kaiming_normal_fill",
    "kaiming_uniform_fill",
    "normal_fill",
    "orthogonal_fill",
    "sparse_fill",
    "trunc_normal_fill",
    "uniform_fill",
    "xavier_normal_fill",
    "xavier_uniform_fill",
]


def init_conv2d(func):
    """
    Decorator that dispatches initialization functions to the actual convolutional
    layers inside wrapper modules (Conv2d, DepthwiseSeparableConv2d, DeformConv2d).

    For DeformConv modules, applies the initialization only to the main convolution
    (dcn). The offset/mask branches are initialized to zero in their __init__ methods.
    """

    @wraps(func)
    def deco(module, *args, **kwargs):
        """
        Apply the initializer to a module or its wrapped convolutional layers.
        """
        if isinstance(module, Conv2d):
            func(module.conv, *args, **kwargs)
        elif isinstance(module, DepthwiseSeparableConv2d):
            deco(module.dw_conv, *args, **kwargs)
            deco(module.pw_conv, *args, **kwargs)
        elif isinstance(module, (DeformConv2dV1, DeformConv2dV2)):
            # Only initialize the main deformable convolution
            func(module.dcn, *args, **kwargs)
        else:
            func(module, *args, **kwargs)

    return deco


@init_conv2d
def c2_msra_fill(module):
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    This corresponds to kaiming_normal with mode="fan_out" and nonlinearity="relu".
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, 0)


@init_conv2d
def c2_xavier_fill(module):
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Caffe2 implementation of XavierFill corresponds to kaiming_uniform with a=1.
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.kaiming_uniform_(module.weight, a=1)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, 0)


@init_conv2d
def kaiming_normal_fill(module, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"):
    """
    Initialize `module.weight` using Kaiming normal initialization.
    Also initializes `module.bias` to 0.

    Uses PyTorch default parameters: a=0, mode="fan_in", nonlinearity="leaky_relu".
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, 0)


@init_conv2d
def kaiming_uniform_fill(module, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"):
    """
    Initialize `module.weight` using Kaiming uniform initialization.
    Also initializes `module.bias` to 0.

    Uses PyTorch default parameters: a=0, mode="fan_in", nonlinearity="leaky_relu".
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, 0)


@init_conv2d
def normal_fill(module, mean: float = 0, std: float = 0.01):
    """
    Initialize `module.weight` from a normal distribution.
    Also initializes `module.bias` to 0.

    Args:
        module: Module whose weight and bias are initialized.
        mean: Mean of the normal distribution. Default: 0.
        std: Standard deviation of the normal distribution. Default: 0.01.
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean=mean, std=std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, 0)


@init_conv2d
def bias_fill(module, val: float = 0):
    """
    Initialize `module.bias` to a constant value.

    Args:
        module: Module whose bias is initialized.
        val: Value used to initialize the bias. Default: 0.
    """
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, val)


@init_conv2d
def constant_fill(module, val: float = 1, bias: float = 0):
    """
    Initialize `module.weight` and `module.bias` to constant values.

    Args:
        module: Module whose weight and bias are initialized.
        val: Value used to initialize the weight. Default: 1.
        bias: Value used to initialize the bias. Default: 0.
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val=val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


@init_conv2d
def dirac_fill(module, groups: int | None = None, bias: float = 0):
    """
    Initialize convolution weights to preserve as many input channels as possible.

    Args:
        module: Module whose convolutional weight is initialized.
        groups: Number of convolution groups. By default, uses ``module.groups``
            when available, otherwise 1.
        bias: Value used to initialize the bias. Default: 0.
    """
    if hasattr(module, "weight") and module.weight is not None:
        groups = getattr(module, "groups", 1) if groups is None else groups
        nn.init.dirac_(module.weight, groups=groups)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


@init_conv2d
def eye_fill(module, bias: float = 0):
    """
    Initialize a two-dimensional weight as the identity matrix.
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.eye_(module.weight)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


@init_conv2d
def orthogonal_fill(module, gain: float = 1, bias: float = 0):
    """
    Initialize weights with a (semi-)orthogonal matrix.
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.orthogonal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


@init_conv2d
def sparse_fill(module, sparsity: float, std: float = 0.01, bias: float = 0):
    """
    Initialize a two-dimensional weight with a fixed fraction of zeros per column.
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.sparse_(module.weight, sparsity=sparsity, std=std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


@init_conv2d
def trunc_normal_fill(module, mean: float = 0, std: float = 1, a: float = -2, b: float = 2):
    """
    Initialize `module.weight` from a truncated normal distribution.
    Also initializes `module.bias` to 0.

    Args:
        module: Module whose weight and bias are initialized.
        mean: Mean of the normal distribution before truncation. Default: 0.
        std: Standard deviation of the normal distribution before truncation. Default: 1.
        a: Lower bound of the truncation interval. Default: -2.
        b: Upper bound of the truncation interval. Default: 2.
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.trunc_normal_(module.weight, mean=mean, std=std, a=a, b=b)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, 0)


@init_conv2d
def uniform_fill(module, a: float = 0, b: float = 1, bias: float = 0):
    """
    Initialize weights from the uniform distribution :math:`U(a, b)`.
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.uniform_(module.weight, a=a, b=b)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


@init_conv2d
def xavier_uniform_fill(module, gain: float = 1, bias: float = 0):
    """
    Initialize `module.weight` using Xavier uniform initialization.
    Also initializes `module.bias` to the specified bias value.

    Args:
        module: Module whose weight and bias are initialized.
        gain: Scaling factor for the weight initialization. Default: 1.
        bias: Value used to initialize the bias. Default: 0.
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.xavier_uniform_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


@init_conv2d
def xavier_normal_fill(module, gain: float = 1, bias: float = 0):
    """
    Initialize `module.weight` using Xavier normal initialization.
    Also initializes `module.bias` to the specified bias value.

    Args:
        module: Module whose weight and bias are initialized.
        gain: Scaling factor for the weight initialization. Default: 1.
        bias: Value used to initialize the bias. Default: 0.
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)
