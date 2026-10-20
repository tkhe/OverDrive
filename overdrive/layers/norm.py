from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from overdrive.utils.registry import Registry

__all__ = [
    "BatchNorm2d",
    "FrozenBatchNorm2d",
    "GroupNorm",
    "InstanceNorm2d",
    "LayerNorm",
    "LayerNorm2d",
    "NORM_REGISTRY",
    "RMSNorm",
    "RMSNorm2d",
    "SyncBatchNorm",
    "get_norm",
]

NORM_REGISTRY = Registry("Norm")


@NORM_REGISTRY.register(alias="FrozenBN")
class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.

    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()

        self.num_features = channels
        self.eps = eps

        self.register_buffer("weight", torch.ones(channels))
        self.register_buffer("bias", torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels) - eps)
        self.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype  # may be half
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def __repr__(self):
        return f"FrozenBatchNorm2d(num_features={self.num_features}, eps={self.eps})"

    @classmethod
    def convert_frozen_batch_norm(cls, module: nn.Module) -> nn.Module:
        """
        Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
            res.num_batches_tracked = module.num_batches_tracked
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batch_norm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

    @classmethod
    def convert_frozenbatchnorm2d_to_batchnorm2d(cls, module: nn.Module) -> nn.Module:
        """
        Convert all FrozenBatchNorm2d to BatchNorm2d

        Args:
            module (torch.nn.Module):

        Returns:
            If module is FrozenBatchNorm2d, returns a new module.
            Otherwise, in-place convert module and return it.
        """
        res = module
        if isinstance(module, FrozenBatchNorm2d):
            res = nn.BatchNorm2d(module.num_features, module.eps)

            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data.clone().detach()
            res.running_var.data = module.running_var.data.clone().detach()
            res.eps = module.eps
            res.num_batches_tracked = module.num_batches_tracked
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozenbatchnorm2d_to_batchnorm2d(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


@NORM_REGISTRY.register(alias="BN")
class BatchNorm2d(nn.BatchNorm2d):
    """
    BatchNorm2d wrapper that standardizes the interface to use `channels` parameter.

    This wrapper provides a consistent API across all normalization layers in the registry,
    where the number of features is specified via `channels` instead of `num_features`.
    This design allows seamless switching between different normalization types through
    the registry without changing the calling code.
    """

    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ):
        """
        Args:
            channels: Number of channels in the input tensor (C from N, C, H, W).
            eps: Value added to denominator for numerical stability. Default: 1e-5.
            momentum: Value used for running mean and variance computation. Default: 0.1.
            affine: If True, has learnable affine parameters. Default: True.
            track_running_stats: If True, tracks running mean and variance. Default: True.
            device: Device to place the module on.
            dtype: Data type for parameters.
        """
        super().__init__(
            num_features=channels,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )


@NORM_REGISTRY.register(alias="SyncBN")
class SyncBatchNorm(nn.SyncBatchNorm):
    """
    SyncBatchNorm wrapper that standardizes the interface to use `channels` parameter.

    This wrapper provides a consistent API for synchronized batch normalization across
    distributed training processes. It synchronizes mean and variance statistics across
    all processes in the process group, ensuring consistent normalization in multi-GPU
    training scenarios.
    """

    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        process_group: Any | None = None,
        device=None,
        dtype=None,
    ):
        """
        Args:
            channels: Number of channels in the input tensor (C from N, C, H, W).
            eps: Value added to denominator for numerical stability. Default: 1e-5.
            momentum: Value used for running mean and variance computation. Default: 0.1.
            affine: If True, has learnable affine parameters. Default: True.
            track_running_stats: If True, tracks running mean and variance. Default: True.
            process_group: Process group for synchronization. If None, uses default group.
            device: Device to place the module on.
            dtype: Data type for parameters.
        """
        super().__init__(
            num_features=channels,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            process_group=process_group,
            device=device,
            dtype=dtype,
        )


@NORM_REGISTRY.register(alias="LN")
class LayerNorm(nn.LayerNorm):
    """
    LayerNorm wrapper that standardizes the interface to use `channels` parameter.

    Layer Normalization normalizes across the feature dimension for each sample
    independently. It's commonly used in transformers and NLP models, and is
    independent of batch size like GroupNorm.
    """

    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        """
        Args:
            channels: Number of features to normalize over (last dimension).
            eps: Value added to denominator for numerical stability. Default: 1e-5.
            elementwise_affine: If True, has learnable affine parameters. Default: True.
            bias: If True, adds a learnable bias. Default: True.
            device: Device to place the module on.
            dtype: Data type for parameters.
        """
        super().__init__(
            normalized_shape=channels,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
        )


@NORM_REGISTRY.register(alias="LN2d")
class LayerNorm2d(LayerNorm):
    """
    Layer Normalization adapted for 2D spatial data (images) with channel-first layout.

    Inherits from LayerNorm and overrides forward to handle (N, C, H, W) tensors by
    permuting to (N, H, W, C) before normalization and back afterward. This avoids
    reimplementing the normalization math while supporting the standard vision tensor
    layout where channels come before spatial dimensions.
    """

    def forward(self, x: torch.Tensor, channel_first: bool = True) -> torch.Tensor:
        """
        Args:
            x: Input tensor.
            channel_first: If True, expects (N, C, H, W) input and permutes internally.
                If False, expects (N, H, W, C) and applies LayerNorm directly.
        """
        if channel_first:
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            x = super().forward(x)
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        else:
            x = super().forward(x)
        return x


@NORM_REGISTRY.register()
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    A simplified variant of LayerNorm that omits mean centering and only normalizes
    by the root mean square statistic: x / RMS(x) * weight. This reduces computation
    while maintaining comparable performance, and is widely adopted in modern LLMs
    (e.g., LLaMA). The forward upcasts to float32 for numerical stability.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        """
        Args:
            channels: Number of features to normalize over (last dimension).
            eps: Value added to denominator for numerical stability. Default: 1e-5.
        """
        super().__init__()

        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


@NORM_REGISTRY.register()
class RMSNorm2d(RMSNorm):
    """
    RMS Normalization adapted for 2D spatial data (images) with channel-first layout.

    Inherits from RMSNorm and overrides forward to handle (N, C, H, W) tensors by
    permuting to (N, H, W, C) before normalization and back afterward. This mirrors
    the LayerNorm2d pattern, providing the computational benefits of RMSNorm for
    vision models.
    """

    def forward(self, x: torch.Tensor, channel_first: bool = True) -> torch.Tensor:
        """
        Args:
            x: Input tensor.
            channel_first: If True, expects (N, C, H, W) input and permutes internally.
                If False, expects (N, H, W, C) and applies RMSNorm directly.
        """
        if channel_first:
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            x = super().forward(x)
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        else:
            x = super().forward(x)
        return x


@NORM_REGISTRY.register(alias="GN")
class GroupNorm(nn.GroupNorm):
    """
    GroupNorm wrapper that standardizes the interface to use `channels` parameter.

    Group Normalization divides channels into groups and normalizes within each group.
    Unlike BatchNorm, it's independent of batch size, making it suitable for small
    batch training or tasks where batch statistics are unreliable (e.g., object detection).
    """

    def __init__(
        self,
        channels: int,
        num_groups: int = 32,
        eps: float = 1e-5,
        affine: bool = True,
        device=None,
        dtype=None,
    ):
        """
        Args:
            channels: Number of channels in the input tensor (C from N, C, H, W).
            num_groups: Number of groups to divide channels into. Default: 32.
            eps: Value added to denominator for numerical stability. Default: 1e-5.
            affine: If True, has learnable affine parameters. Default: True.
            device: Device to place the module on.
            dtype: Data type for parameters.
        """
        super().__init__(
            num_groups=num_groups,
            num_channels=channels,
            eps=eps,
            affine=affine,
            device=device,
            dtype=dtype,
        )


@NORM_REGISTRY.register(alias="IN")
class InstanceNorm2d(nn.InstanceNorm2d):
    """
    InstanceNorm2d wrapper that standardizes the interface to use `channels` parameter.

    Instance Normalization normalizes each channel independently for each sample,
    computing statistics over spatial dimensions (H, W) only. It's commonly used
    in style transfer and image generation tasks where normalizing across the batch
    would remove important style information.
    """

    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device=None,
        dtype=None,
    ):
        """
        Args:
            channels: Number of channels in the input tensor (C from N, C, H, W).
            eps: Value added to denominator for numerical stability. Default: 1e-5.
            momentum: Value used for running mean and variance computation. Default: 0.1.
            affine: If True, has learnable affine parameters. Default: False.
            track_running_stats: If True, tracks running mean and variance. Default: False.
            device: Device to place the module on.
            dtype: Data type for parameters.
        """
        super().__init__(
            num_features=channels,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )


def get_norm(norm: str | nn.Module | Mapping, channels: int) -> nn.Module | None:
    """
    Factory function to create or return a normalization module.

    Provides a flexible interface for specifying normalization layers in multiple
    formats. Leverages ``NORM_REGISTRY`` to build normalization modules from
    string or dict specifications.

    Args:
        norm: The normalization specification in one of the following formats:
            - ``str``: Name or alias of a registered normalization (e.g., ``"BN"``,
              ``"GN"``, ``"FrozenBN"``, ``"LN2d"``).
            - ``nn.Module``: An already instantiated normalization module,
              returned as-is.
            - ``Mapping``: A config dict with ``name`` key and optional
              constructor args (e.g., ``{"name": "GroupNorm", "num_groups": 16}``).
            - Falsy value (``None``, ``""``, ``{}`` etc.): Returns ``None``,
              indicating no normalization.

        channels: Number of channels/features (C) to pass to the normalization
            constructor when building from a string or mapping.

    Returns:
        A normalization module instance, or ``None`` if *norm* is falsy.

    Raises:
        ValueError: If *norm* is not one of the supported types.

    Examples::

        >>> get_norm("BN", channels=64)
        BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> get_norm(nn.BatchNorm2d(32), channels=32)
        BatchNorm2d(32, ...)
        >>> get_norm({"name": "GroupNorm", "num_groups": 8}, channels=32)
        GroupNorm(num_groups=8, num_channels=32, eps=1e-05, affine=True)
        >>> get_norm(None, channels=16) is None
        True
    """
    # Falsy values (None, "", {}, etc.) uniformly mean "no normalization"
    if not norm:
        return None

    if isinstance(norm, str):
        # Wrap the name into a registry-compatible config dict and pass channels
        return NORM_REGISTRY.build(dict(name=norm, channels=channels))
    elif isinstance(norm, nn.Module):
        # Already instantiated module: return as-is
        return norm
    elif isinstance(norm, Mapping):
        # Config dict must contain a "name" key for registry lookup;
        # remaining items are forwarded as constructor kwargs.
        return NORM_REGISTRY.build(norm, channels=channels)
    else:
        raise ValueError(f"Unsupported norm type: {type(norm)}")
