import torch
import torch.nn as nn

__all__ = [
    "LayerScale",
    "Scale",
]


class Scale(nn.Module):
    """
    Multiply every input element by a shared learnable scalar.

    Args:
        scale: Initial value of the learnable scalar. Default: 1.0.

    Shape:
        Input: A tensor of any shape.
        Output: A tensor with the same shape as the input.

    Examples::

        >>> import torch
        >>> from overdrive.layers.scale import Scale
        >>> layer = Scale(scale=2.0)
        >>> layer(torch.tensor([1.0, 2.0])).tolist()
        [2.0, 4.0]
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


class LayerScale(nn.Module):
    """
    Multiply each feature in the last input dimension by a learnable scale.

    The scaling parameters are shared across all leading dimensions.

    Args:
        embed_dim: Number of features in the last input dimension.
        scale: Initial value of each learnable scaling parameter. Default: 1e-5.
        inplace: If True, modifies and returns the input tensor in place. Default: True.

    Shape:
        Input: (*, embed_dim), where * denotes any number of leading dimensions.
        Output: A tensor with the same shape as the input.

    Examples::

        >>> import torch
        >>> from overdrive.layers.scale import LayerScale
        >>> layer = LayerScale(embed_dim=3, scale=0.5)
        >>> x = torch.ones(2, 3)
        >>> layer(x) is x
        True
        >>> x.tolist()
        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    """

    def __init__(self, embed_dim: int, scale: float = 1e-5, inplace: bool = True):
        super().__init__()

        self.inplace = inplace
        self.gamma = nn.Parameter(scale * torch.ones(embed_dim))

    def forward(self, x: torch.Tensor):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
