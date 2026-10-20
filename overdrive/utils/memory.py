import logging
from contextlib import contextmanager
from functools import wraps

import torch

__all__ = ["retry_if_cuda_oom"]


@contextmanager
def _ignore_torch_cuda_oom():
    """
    A context which ignores CUDA OOM exception from pytorch.
    """
    try:
        yield
    except torch.cuda.OutOfMemoryError:
        pass


def retry_if_cuda_oom(func):
    """
    Makes a function retry itself after encountering
    pytorch's CUDA OOM error.
    It will first retry after calling `torch.cuda.empty_cache()`.

    If that still fails, it will then retry by trying to convert inputs to CPUs.
    In this case, it expects the function to dispatch to CPU implementation.
    The return values may become CPU tensors as well and it's user's
    responsibility to convert it back to CUDA tensor if needed.

    Args:
        func: a stateless callable that takes tensor-like objects as arguments

    Returns:
        a callable which retries `func` if OOM is encountered.

    Examples:

        >>> import torch
        >>> from overdrive.utils.memory import retry_if_cuda_oom
        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> x = torch.randn(32, 32, device=device)
        >>> y = torch.randn(32, 32, device=device)
        >>> # output may be on CPU if CUDA OOM triggers CPU fallback.
        >>> output = retry_if_cuda_oom(torch.matmul)(x, y)

    Note:
        1. When converting inputs to CPU, it will only look at each argument and check
           if it has `.device` and `.to` for conversion. Nested structures of tensors
           are not supported.

        2. Since the function might be called more than once, it has to be
           stateless.
    """

    def maybe_to_cpu(x):
        try:
            like_gpu_tensor = x.device.type == "cuda" and hasattr(x, "to")
        except AttributeError:
            like_gpu_tensor = False

        if like_gpu_tensor:
            return x.to(device="cpu")
        else:
            return x

    @wraps(func)
    def wrapped(*args, **kwargs):
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Clear cache and retry
        torch.cuda.empty_cache()
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Try on CPU. This slows down the code significantly, therefore print a notice.
        logger = logging.getLogger(__name__)
        logger.info(f"Attempting to copy inputs of {str(func)} to CPU due to CUDA OOM")
        new_args = (maybe_to_cpu(x) for x in args)
        new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapped
