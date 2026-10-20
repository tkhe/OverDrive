from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from overdrive.utils.registry import Registry

__all__ = [
    "ACTIVATION_REGISTRY",
    "HardMish",
    "HardSigmoid",
    "HardSwish",
    "LeakyReLU",
    "MemoryEfficientHardMish",
    "MemoryEfficientHardSigmoid",
    "MemoryEfficientHardSwish",
    "MemoryEfficientMish",
    "MemoryEfficientSwish",
    "Mish",
    "QuickGELU",
    "ReLU",
    "ReLU6",
    "SiLU",
    "Sigmoid",
    "get_activation",
]

ACTIVATION_REGISTRY = Registry("Activation")


@ACTIVATION_REGISTRY.register()
class ReLU(nn.ReLU):
    """
    Same as `torch.nn.ReLU`, but `inplace` is set to True by default.
    """

    def __init__(self, inplace: bool = True):
        super().__init__(inplace=inplace)


@ACTIVATION_REGISTRY.register()
class ReLU6(nn.ReLU6):
    """
    Same as `torch.nn.ReLU6`, but `inplace` is set to True by default.
    """

    def __init__(self, inplace: bool = True):
        super().__init__(inplace=inplace)


@ACTIVATION_REGISTRY.register()
class Sigmoid(nn.Sigmoid):
    """
    Same as `torch.nn.Sigmoid`, with additional `inplace` support.
    """

    def __init__(self, inplace: bool = True):
        super().__init__()

        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sigmoid_() if self.inplace else x.sigmoid()


@ACTIVATION_REGISTRY.register(alias="Swish")
class SiLU(nn.SiLU):
    """
    Same as `torch.nn.SiLU`, but `inplace` is set to True by default.
    """

    def __init__(self, inplace: bool = True):
        super().__init__(inplace=inplace)


@ACTIVATION_REGISTRY.register()
class LeakyReLU(nn.LeakyReLU):
    """
    Same as `torch.nn.LeakyReLU`, but `negative_slope` is set to 0.1 and `inplace` is set to True by default.
    """

    def __init__(self, negative_slope: float = 0.1, inplace: bool = True):
        super().__init__(negative_slope=negative_slope, inplace=inplace)


@ACTIVATION_REGISTRY.register()
class HardSigmoid(nn.Hardsigmoid):
    """
    Same as `torch.nn.Hardsigmoid`, but `inplace` is set to True by default.
    """

    def __init__(self, inplace: bool = True):
        super().__init__(inplace=inplace)


@ACTIVATION_REGISTRY.register()
class HardSwish(nn.Hardswish):
    """
    Same as `torch.nn.Hardswish`, but `inplace` is set to True by default.
    """

    def __init__(self, inplace: bool = True):
        super().__init__(inplace=inplace)


@ACTIVATION_REGISTRY.register()
class Mish(nn.Mish):
    """
    Same as `torch.nn.Mish`, but `inplace` is set to True by default.
    """

    def __init__(self, inplace: bool = True):
        super().__init__(inplace=inplace)


@ACTIVATION_REGISTRY.register()
class HardMish(nn.Module):
    """
    Modified from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/activations.py
    """

    def __init__(self, inplace: bool = True):
        super().__init__()

        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace:
            return x.mul_(0.5 * (x + 2).clamp(min=0, max=2))
        else:
            return 0.5 * x * (x + 2).clamp(min=0, max=2)


@ACTIVATION_REGISTRY.register()
class QuickGELU(nn.Module):
    """
    Quick GELU approximation: x * sigmoid(1.702 * x).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class SwishMemoryEfficientFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x):
        return g.op("Mul", x, g.op("Sigmoid", x))

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x * x.sigmoid()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]
        sigmoid = x.sigmoid()
        return grad_output * (sigmoid * (1 + x * (1 - sigmoid)))


@ACTIVATION_REGISTRY.register(alias=["Swish-ME", "SiLU-ME", "Swish-MemoryEfficient", "SiLU-MemoryEfficient"])
class MemoryEfficientSwish(nn.Module):
    """
    Memory efficient Swish (SiLU) that recomputes intermediates in backward.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SwishMemoryEfficientFunction.apply(x)


class MishMemoryEfficientFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x):
        return g.op("Mul", x, g.op("Tanh", g.op("Softplus", x)))

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x * F.softplus(x).tanh()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]
        sigmoid = x.sigmoid()
        tanh_softplus = F.softplus(x).tanh()
        return grad_output * (tanh_softplus + x * sigmoid * (1 - tanh_softplus * tanh_softplus))


@ACTIVATION_REGISTRY.register(alias=["Mish-ME", "Mish-MemoryEfficient"])
class MemoryEfficientMish(nn.Module):
    """
    Memory efficient Mish that recomputes intermediates in backward.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return MishMemoryEfficientFunction.apply(x)


class HardSigmoidMemoryEfficientFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x):
        return g.op("HardSigmoid", x, alpha_f=1.0 / 6.0, beta_f=0.5)

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return (x + 3).clamp(min=0, max=6) / 6

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]
        slope = ((x >= -3) & (x <= 3)).to(dtype=x.dtype) / 6
        return grad_output * slope


@ACTIVATION_REGISTRY.register(alias=["HardSigmoid-ME", "HardSigmoid-MemoryEfficient"])
class MemoryEfficientHardSigmoid(nn.Module):
    """
    Memory efficient hard sigmoid that recomputes intermediates in backward.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HardSigmoidMemoryEfficientFunction.apply(x)


class HardSwishMemoryEfficientFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x):
        hard_sigmoid = g.op("HardSigmoid", x, alpha_f=1.0 / 6.0, beta_f=0.5)
        return g.op("Mul", x, hard_sigmoid)

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x * (x + 3).clamp(min=0, max=6).div(6)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]
        slope = torch.where((x >= -3) & (x <= 3), x / 3 + 0.5, (x >= 3).to(dtype=x.dtype))
        return grad_output * slope


@ACTIVATION_REGISTRY.register(alias=["HardSwish-ME", "HardSwish-MemoryEfficient"])
class MemoryEfficientHardSwish(nn.Module):
    """
    Memory efficient hard swish that recomputes intermediates in backward.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HardSwishMemoryEfficientFunction.apply(x)


class HardMishMemoryEfficientFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x):
        hard_sigmoid = g.op("HardSigmoid", x, alpha_f=0.5, beta_f=1.0)
        return g.op("Mul", x, hard_sigmoid)

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return 0.5 * x * (x + 2).clamp(min=0, max=2)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]
        slope = torch.where((x >= -2) & (x <= 0), x + 1, (x >= -2).to(dtype=x.dtype))
        return grad_output * slope


@ACTIVATION_REGISTRY.register(alias=["HardMish-ME", "HardMish-MemoryEfficient"])
class MemoryEfficientHardMish(nn.Module):
    """
    Memory efficient hard mish that recomputes intermediates in backward.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HardMishMemoryEfficientFunction.apply(x)


ACTIVATION_REGISTRY.register(name="ELU", obj=nn.ELU)
ACTIVATION_REGISTRY.register(name="CELU", obj=nn.CELU)
ACTIVATION_REGISTRY.register(name="SELU", obj=nn.SELU)
ACTIVATION_REGISTRY.register(name="PReLU", obj=nn.PReLU)
ACTIVATION_REGISTRY.register(name="GELU", obj=nn.GELU)
ACTIVATION_REGISTRY.register(name="Tanh", obj=nn.Tanh)


def get_activation(activation: str | nn.Module | Mapping) -> nn.Module | None:
    """
    Factory function to create or return an activation module.

    Provides a flexible interface for specifying activation functions in multiple
    formats. Leverages ``ACTIVATION_REGISTRY`` to build activation modules from
    string or dict specifications.

    Args:
        activation: The activation specification in one of the following formats:
            - ``str``: Name of a registered activation (e.g., ``"ReLU"``,
              ``"GELU"``, ``"Swish"``).
            - ``nn.Module``: An already instantiated activation module,
              returned as-is.
            - ``Mapping``: A config dict with ``name`` key and optional
              constructor args
              (e.g., ``{"name": "LeakyReLU", "negative_slope": 0.2}``).
            - Falsy value (``None``, ``""``, ``{}`` etc.): Returns ``None``,
              indicating no activation.

    Returns:
        An activation module instance, or ``None`` if *activation* is falsy.

    Raises:
        ValueError: If *activation* is not one of the supported types.

    Examples::

        >>> get_activation("ReLU")
        ReLU(inplace=True)
        >>> get_activation("QuickGELU")
        QuickGELU()
        >>> get_activation("Swish-ME")
        MemoryEfficientSwish()
        >>> get_activation("SiLU-ME")
        MemoryEfficientSwish()
        >>> get_activation(nn.GELU())
        GELU(approximate='none')
        >>> get_activation({"name": "LeakyReLU", "negative_slope": 0.2})
        LeakyReLU(negative_slope=0.2, inplace=True)
        >>> get_activation(None) is None
        True
    """
    # Falsy values (None, "", {}, etc.) uniformly mean "no activation"
    if not activation:
        return None

    if isinstance(activation, str):
        # Wrap the name into a registry-compatible config dict
        return ACTIVATION_REGISTRY.build(dict(name=activation))
    elif isinstance(activation, nn.Module):
        return activation
    elif isinstance(activation, Mapping):
        # Config dict must contain a "name" key for registry lookup;
        # remaining items are forwarded as constructor kwargs.
        return ACTIVATION_REGISTRY.build(activation)
    else:
        raise ValueError(f"Unsupported activation type: {type(activation)}")
