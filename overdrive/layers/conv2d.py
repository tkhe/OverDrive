from collections.abc import Mapping

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as to_2tuple
from torchvision.ops import DeformConv2d

from overdrive.utils.registry import Registry

from .activation import get_activation
from .misc import FreezableMixin
from .norm import get_norm

__all__ = [
    "CONV_REGISTRY",
    "Conv2d",
    "DeformConv2dV1",
    "DeformConv2dV2",
    "DepthwiseSeparableConv2d",
    "get_conv2d",
]

CONV_REGISTRY = Registry("Conv2d")

_size_2_t = int | tuple[int, int]


@CONV_REGISTRY.register()
class Conv2d(nn.Module, FreezableMixin):
    """
    Enhance `torch.nn.Conv2d` to support more features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t | str = "auto",
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool | str = "auto",
        padding_mode: str = "zeros",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        norm: str | dict | nn.Module | None = None,
        activation: str | dict | nn.Module | None = None,
    ):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution. Default: 1
            padding: Padding added to both sides of the input. If "auto", padding is
                calculated as (kernel_size - 1) * dilation // 2. Default: "auto"
            dilation: Spacing between kernel elements. Default: 1
            groups: Number of blocked connections from input channels to output channels. Default: 1
            bias: If True, adds a learnable bias to the output. If "auto", bias is disabled
                for BatchNorm2d, SyncBatchNorm, and InstanceNorm2d. It is enabled otherwise.
                Default: "auto"
            padding_mode: Padding mode. Default: "zeros"
            device: Device to place the module on.
            dtype: Data type of the module parameters.
            norm: Normalization layer specification (str, dict, or nn.Module).
                Falsy values such as None, "", and {} disable normalization. Default: None
            activation: Activation layer specification (str, dict, or nn.Module). Default: None
        """
        super().__init__()

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if padding == "auto":
            padding = tuple((k - 1) * d // 2 for k, d in zip(kernel_size, dilation, strict=True))
        else:
            padding = to_2tuple(padding)

        norm_layer = get_norm(norm, out_channels)

        if bias == "auto":
            bias = not isinstance(norm_layer, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.InstanceNorm2d))
        elif not isinstance(bias, bool):
            raise ValueError(f"bias must be bool or 'auto', got {bias}")

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.norm = norm_layer
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


@CONV_REGISTRY.register(alias="DWConv2d")
class DepthwiseSeparableConv2d(nn.Module, FreezableMixin):
    """
    Depthwise separable convolution.

    See https://arxiv.org/pdf/1704.04861.pdf for details.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t | str = "auto",
        dilation: _size_2_t = 1,
        bias: bool | str = "auto",
        norm: str | dict | nn.Module | None = None,
        activation: str | dict | nn.Module | None = None,
        dw_norm: str | dict | nn.Module | None = "default",
        dw_activation: str | dict | nn.Module | None = "default",
        pw_norm: str | dict | nn.Module | None = "default",
        pw_activation: str | dict | nn.Module | None = "default",
    ):
        """
        Args:
            bias: Bias setting for both convolutions. If "auto", each Conv2d determines
                its bias from its own normalization layer. Default: "auto"
            norm (str or dict): Default normalization layer for both depthwise conv
                and pointwise conv.
            activation (str or dict): Default activation layer for both depthwise conv
                and pointwise conv.
            dw_norm (str or dict): Normalization layer for depthwise conv. If it is `default`,
                it will be the same as `norm`.
            dw_activation (str or dict): Activation layer for depthwise conv. If it is `default`,
                it will be the same as `activation`.
            pw_norm (str or dict): Normalization layer for pointwise conv. If it is `default`,
                it will be the same as `norm`.
            pw_activation (str or dict): Activation layer for pointwise conv. If it is `default`,
                it will be the same as `activation`.
        """
        super().__init__()

        dw_norm = norm if dw_norm == "default" else dw_norm
        dw_activation = activation if dw_activation == "default" else dw_activation
        pw_norm = norm if pw_norm == "default" else pw_norm
        pw_activation = activation if pw_activation == "default" else pw_activation

        self.dw_conv = Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias, norm=dw_norm, activation=dw_activation)
        self.pw_conv = Conv2d(in_channels, out_channels, kernel_size=1, bias=bias, norm=pw_norm, activation=pw_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


@CONV_REGISTRY.register(alias=["DCNv1", "DeformConvV1"])
class DeformConv2dV1(nn.Module, FreezableMixin):
    """
    Deformable convolution v1 with a learned offset prediction branch.

    This module wraps :class:`torchvision.ops.DeformConv2d` and predicts the
    sampling offsets from the input feature map before applying optional
    normalization and activation layers.

    The offset prediction branch is initialized with zero weights and bias, so
    the module starts from regular convolution sampling locations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t | str = "auto",
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        norm: str | dict | nn.Module | None = None,
        activation: str | dict | nn.Module | None = None,
    ):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the deformable convolution kernel.
            stride: Stride of the deformable convolution. Default: 1
            padding: Padding added to both sides of the input. If "auto", padding is
                calculated as (kernel_size - 1) * dilation // 2. Default: "auto"
            dilation: Spacing between kernel elements. Default: 1
            groups: Number of blocked connections from input channels to output channels.
                Both ``in_channels`` and ``out_channels`` must be divisible by
                ``groups``. Default: 1
            bias: If True, adds a learnable bias to the deformable convolution output.
                Default: True
            norm: Normalization layer specification (str, dict, or nn.Module). Default: None
            activation: Activation layer specification (str, dict, or nn.Module). Default: None
        """
        super().__init__()

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if padding == "auto":
            padding = tuple((k - 1) * d // 2 for k, d in zip(kernel_size, dilation, strict=True))
        else:
            padding = to_2tuple(padding)

        self.dcn = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups,
            bias=bias,
        )
        self.offset = nn.Conv2d(in_channels, 2 * kernel_size[0] * kernel_size[1], kernel_size=3, stride=stride, padding=1, bias=True)
        self.norm = get_norm(norm, out_channels)
        self.activation = get_activation(activation)

        # Initialize offset prediction branch to zero for stable training
        nn.init.constant_(self.offset.weight, 0)
        nn.init.constant_(self.offset.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset(x)
        x = self.dcn(x, offset)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


@CONV_REGISTRY.register(alias=["DCNv2", "DeformConvV2"])
class DeformConv2dV2(nn.Module, FreezableMixin):
    """
    Modulated deformable convolution v2 with learned offset and mask branches.

    This module wraps :class:`torchvision.ops.DeformConv2d` and predicts both
    sampling offsets and modulation masks from the input feature map before
    applying optional normalization and activation layers.

    The offset and mask prediction branches are initialized with zero weights
    and bias. The initial offsets are therefore zero and the initial modulation
    masks are ``sigmoid(0) = 0.5``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t | str = "auto",
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        norm: str | dict | nn.Module | None = None,
        activation: str | dict | nn.Module | None = None,
    ):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the deformable convolution kernel.
            stride: Stride of the deformable convolution. Default: 1
            padding: Padding added to both sides of the input. If "auto", padding is
                calculated as (kernel_size - 1) * dilation // 2. Default: "auto"
            dilation: Spacing between kernel elements. Default: 1
            groups: Number of blocked connections from input channels to output channels.
                Both ``in_channels`` and ``out_channels`` must be divisible by
                ``groups``. Default: 1
            bias: If True, adds a learnable bias to the deformable convolution output.
                Default: True
            norm: Normalization layer specification (str, dict, or nn.Module). Default: None
            activation: Activation layer specification (str, dict, or nn.Module). Default: None
        """
        super().__init__()

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if padding == "auto":
            padding = tuple((k - 1) * d // 2 for k, d in zip(kernel_size, dilation, strict=True))
        else:
            padding = to_2tuple(padding)

        self.dcn = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups,
            bias=bias,
        )
        self.offset = nn.Conv2d(in_channels, 2 * kernel_size[0] * kernel_size[1], kernel_size=3, stride=stride, padding=1, bias=True)
        self.mask = nn.Conv2d(in_channels, kernel_size[0] * kernel_size[1], kernel_size=3, stride=stride, padding=1, bias=True)
        self.norm = get_norm(norm, out_channels)
        self.activation = get_activation(activation)

        # Initialize offset and mask prediction branches to zero for stable training
        nn.init.constant_(self.offset.weight, 0)
        nn.init.constant_(self.offset.bias, 0)
        nn.init.constant_(self.mask.weight, 0)
        nn.init.constant_(self.mask.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset(x)
        mask = self.mask(x).sigmoid()
        x = self.dcn(x, offset, mask)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def get_conv2d(conv2d: str | dict | nn.Module, *args, **kwargs) -> nn.Module | None:
    """
    Factory function to create or return a 2D convolution module.

    Provides a flexible interface for specifying convolution layers in multiple
    formats. Leverages ``CONV_REGISTRY`` to build convolution modules from
    string or dict specifications.

    Args:
        conv2d: The convolution specification in one of the following formats:
            - ``str``: Name or alias of a registered convolution (e.g.,
              ``"Conv2d"``, ``"DWConv2d"``).
            - ``nn.Module``: An already instantiated convolution module,
              returned as-is.
            - ``Mapping``: A config dict with ``name`` key and optional
              constructor args
              (e.g., ``{"name": "Conv2d", "kernel_size": 3}``).
            - Falsy value (``None``, ``""``, ``{}`` etc.): Returns ``None``,
              indicating no convolution.
        *args: Positional arguments forwarded to the convolution constructor.
        **kwargs: Keyword arguments forwarded to the convolution constructor.

    Returns:
        A convolution module instance, or ``None`` if *conv2d* is falsy.

    Raises:
        ValueError: If *conv2d* is not one of the supported types.

    Examples::

        >>> get_conv2d("Conv2d", 3, 64, kernel_size=3)
        Conv2d(...)
        >>> get_conv2d(nn.Conv2d(3, 64, 3))
        Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
        >>> get_conv2d({"name": "Conv2d", "in_channels": 3, "out_channels": 64, "kernel_size": 3})
        Conv2d(...)
        >>> get_conv2d(None) is None
        True
    """
    # Falsy values (None, "", {}, etc.) uniformly mean "no convolution"
    if not conv2d:
        return None

    if isinstance(conv2d, str):
        # Wrap the name into a registry-compatible config dict
        return CONV_REGISTRY.get(conv2d)(*args, **kwargs)
    elif isinstance(conv2d, nn.Module):
        return conv2d
    elif isinstance(conv2d, Mapping):
        # Config dict must contain a "name" key for registry lookup;
        # remaining items are forwarded as constructor kwargs.
        conv_cfg = dict(conv2d)
        if "name" not in conv_cfg:
            raise ValueError("conv2d config must contain 'name'")

        target = conv_cfg.pop("name")
        conv_cls = CONV_REGISTRY.get(target)
        conv_cfg.update(**kwargs)
        return conv_cls(*args, **conv_cfg)
    else:
        raise ValueError(f"Unsupported conv2d type: {type(conv2d)}")
