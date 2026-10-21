import copy
import itertools
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch

__all__ = ["get_default_optimizer_params"]


def get_default_optimizer_params(
    model: torch.nn.Module,
    base_lr: float | None = None,
    weight_decay: float | None = None,
    weight_decay_norm: float | None = None,
    bias_lr_factor: float | None = 1.0,
    weight_decay_bias: float | None = None,
    lr_factor_func: Callable | None = None,
    overrides: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    """
    Get default param list for optimizer, with support for a few types of
    overrides. If no overrides needed, this is equivalent to `model.parameters()`.

    Args:
        base_lr: lr for every group by default. Can be omitted to use the one in optimizer.
        weight_decay: weight decay for every group by default. Can be omitted to use the one
            in optimizer.
        weight_decay_norm: override weight decay for params in normalization layers
        bias_lr_factor: multiplier of lr for bias parameters.
        weight_decay_bias: override weight decay for bias parameters.
        lr_factor_func: function to calculate lr decay rate by mapping the parameter names to
            corresponding lr decay rate. Note that setting this option requires
            also setting ``base_lr``.
        overrides: if not `None`, provides values for optimizer hyperparameters
            (LR, weight decay) for module parameters with a given name; e.g.
            ``{"embedding": {"lr": 0.01, "weight_decay": 0.1}}`` will set the LR and
            weight decay values for all module parameters named `embedding`.

    For common detection models, ``weight_decay_norm`` is the only option
    needed to be set. ``bias_lr_factor,weight_decay_bias`` are legacy settings
    from Detectron1 that are not found useful.

    Example:

        >>> torch.optim.SGD(get_default_optimizer_params(model, weight_decay_norm=0),
                            lr=0.01,
                            weight_decay=1e-4,
                            momentum=0.9)
    """
    if overrides is None:
        overrides = {}

    defaults = {}

    if base_lr is not None:
        defaults["lr"] = base_lr

    if weight_decay is not None:
        defaults["weight_decay"] = weight_decay

    bias_overrides = {}
    if bias_lr_factor is not None and bias_lr_factor != 1.0:
        # NOTE: we now by default make bias hyperparameters
        # exactly the same as regular weights.
        if base_lr is None:
            raise ValueError("bias_lr_factor requires base_lr")
        bias_overrides["lr"] = base_lr * bias_lr_factor

    if weight_decay_bias is not None:
        bias_overrides["weight_decay"] = weight_decay_bias

    if len(bias_overrides):
        if "bias" in overrides:
            raise ValueError("Conflicting overrides for 'bias'")
        overrides["bias"] = bias_overrides

    if lr_factor_func is not None:
        if base_lr is None:
            raise ValueError("lr_factor_func requires base_lr")

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: list[dict[str, Any]] = []
    memo: set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue

            # Avoid duplicating parameters
            if value in memo:
                continue

            memo.add(value)

            hyperparams = copy.copy(defaults)
            if isinstance(module, norm_module_types) and weight_decay_norm is not None:
                hyperparams["weight_decay"] = weight_decay_norm

            if lr_factor_func is not None:
                hyperparams["lr"] *= lr_factor_func(f"{module_name}.{module_param_name}")

            hyperparams.update(overrides.get(module_param_name, {}))
            params.append({"params": [value], **hyperparams})
    return reduce_param_groups(params)


def _expand_param_groups(params: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Transform parameter groups into per-parameter structure.
    # Later items in `params` can overwrite parameters set in previous items.
    ret = defaultdict(dict)
    for item in params:
        if "params" not in item:
            raise KeyError("Each parameter group must contain a 'params' entry")

        cur_params = {x: y for x, y in item.items() if x != "params" and x != "param_names"}
        if "param_names" in item:
            for param_name, param in zip(item["param_names"], item["params"], strict=True):
                ret[param].update({"param_names": [param_name], "params": [param], **cur_params})
        else:
            for param in item["params"]:
                ret[param].update({"params": [param], **cur_params})
    return list(ret.values())


def reduce_param_groups(params: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Reorganize the parameter groups and merge duplicated groups.
    # The number of parameter groups needs to be as small as possible in order
    # to efficiently use the PyTorch multi-tensor optimizer. Therefore instead
    # of using a parameter_group per single parameter, we reorganize the
    # parameter groups and merge duplicated groups. This approach speeds
    # up multi-tensor optimizer significantly.
    params = _expand_param_groups(params)
    groups = defaultdict(list)  # re-group all parameter groups by their hyperparams
    for item in params:
        cur_params = tuple((x, y) for x, y in item.items() if x != "params" and x != "param_names")
        groups[cur_params].append({"params": item["params"]})
        if "param_names" in item:
            groups[cur_params][-1]["param_names"] = item["param_names"]

    ret = []
    for param_keys, param_values in groups.items():
        cur = {kv[0]: kv[1] for kv in param_keys}
        cur["params"] = list(itertools.chain.from_iterable([params["params"] for params in param_values]))
        if len(param_values) > 0 and "param_names" in param_values[0]:
            cur["param_names"] = list(itertools.chain.from_iterable([params["param_names"] for params in param_values]))
        ret.append(cur)
    return ret
