import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as to_2tuple

from .activation import get_activation
from .norm import get_norm

__all__ = [
    "DropPath",
    "FFN",
    "MLP",
    "SwiGLUFFN",
]


class DropPath(nn.Module):
    """
    Apply stochastic depth independently to each sample during training.

    A single Bernoulli mask value is broadcast across all non-batch dimensions
    of each sample. In evaluation mode, the module acts as the identity.

    Args:
        drop_prob: Probability of dropping a sample's entire path, in [0, 1].
            A value of 0.0 disables dropping. Default: 0.0.
        scale_by_keep: If True, scale retained samples by the reciprocal of the
            keep probability to preserve their expected magnitude. Default: True.

    Shape:
        Input: (N, ...), where N is the batch size.
        Output: A tensor with the same shape as the input.

    Examples::

        >>> import torch
        >>> from overdrive.layers.transformer import DropPath
        >>> layer = DropPath(drop_prob=0.1)
        >>> x = torch.ones(2, 10, 16)
        >>> layer(x).shape
        torch.Size([2, 10, 16])
        >>> _ = layer.eval()
        >>> layer(x) is x
        True
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()

        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class FFN(nn.Module):
    """
    Apply two linear projections with optional activation, dropout, and normalization.

    The computation order is ``fc1 -> activation -> drop1 -> norm -> fc2 -> drop2``,
    with optional stages enabled by their arguments.

    Args:
        in_features: Number of features in the last input dimension.
        hidden_features: Number of features produced by the first linear
            projection. Defaults to in_features when None.
        out_features: Number of features in the last output dimension.
            Defaults to in_features when None.
        norm: Normalization specification accepted by :func:`get_norm`, using
            hidden_features channels. The selected layer must support the hidden
            tensor's layout. Falsy values disable normalization. Default: None.
        activation: Activation specification accepted by :func:`get_activation`.
            Falsy values disable activation. Default: None.
        drop: Dropout probability shared by both stages, or a pair of
            probabilities for the hidden and output stages, respectively.
            A probability of 0.0 disables the corresponding stage. Default: 0.0.
        bias: Whether both linear projections use bias, or a pair of booleans
            controlling the first and second projections separately. Default: True.

    Shape:
        Input: (*, in_features), where * denotes any number of leading dimensions.
        Output: (*, out_features), with the same leading dimensions as the input.

    Examples::

        >>> import torch
        >>> from overdrive.layers.transformer import FFN
        >>> layer = FFN(
        ...     in_features=16, hidden_features=32, out_features=8,
        ...     norm="LayerNorm", activation="GELU", drop=(0.1, 0.2),
        ... )
        >>> layer(torch.randn(2, 10, 16)).shape
        torch.Size([2, 10, 8])
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        norm: str | dict | nn.Module = None,
        activation: str | dict | nn.Module = None,
        drop: float = 0.0,
        bias: bool | tuple[bool, bool] = True,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        bias = to_2tuple(bias)
        drop = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.activation = get_activation(activation)
        self.drop1 = nn.Dropout(drop[0]) if drop[0] > 0.0 else None
        self.norm = get_norm(norm, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop[1]) if drop[1] > 0.0 else None

    def forward(self, x):
        x = self.fc1(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.drop1 is not None:
            x = self.drop1(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.fc2(x)
        if self.drop2 is not None:
            x = self.drop2(x)
        return x


class MLP(FFN):
    """
    Provide an MLP interface with the same computation as :class:`FFN`.

    This subclass inherits the constructor and forward method from :class:`FFN`.
    See :class:`FFN` for argument descriptions and input/output shapes.

    Examples::

        >>> import torch
        >>> from overdrive.layers.transformer import MLP
        >>> layer = MLP(in_features=16, hidden_features=32, activation="GELU")
        >>> layer(torch.randn(2, 10, 16)).shape
        torch.Size([2, 10, 16])
    """

    pass


class SwiGLUFFN(nn.Module):
    """
    Apply a gated feed-forward network using SiLU activation by default.

    The computation is ``down_proj(activation(gate_proj(x)) * up_proj(x))``.
    The gate and value projections have hidden_dim features, and their
    elementwise product is projected back to embed_dim features.

    Args:
        embed_dim: Number of features in the last input and output dimensions.
        hidden_dim: Number of features in both the gate and value projections.
        bias: Whether all three linear projections use bias. Default: True.
        activation: Gate activation specification accepted by
            :func:`get_activation`. Falsy values use the linear gate projection
            directly. Default: "SiLU".

    Shape:
        Input: (*, embed_dim), where * denotes any number of leading dimensions.
        Output: A tensor with the same shape as the input.

    Examples::

        >>> import torch
        >>> from overdrive.layers.transformer import SwiGLUFFN
        >>> layer = SwiGLUFFN(embed_dim=16, hidden_dim=32)
        >>> layer(torch.randn(2, 10, 16)).shape
        torch.Size([2, 10, 16])
    """

    def __init__(self, embed_dim: int, hidden_dim: int, bias: bool = True, activation: str | dict | nn.Module | None = "SiLU"):
        super().__init__()

        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        if self.activation is not None:
            gate = self.activation(gate)
        return self.down_proj(gate * self.up_proj(x))
