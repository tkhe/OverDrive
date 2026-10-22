import math

import torch
import torch.nn as nn

__all__ = ["SinePositionEmbedding"]


class SinePositionEmbedding(nn.Module):
    """
    2D sine-cosine position embedding for image feature maps.

    Generates position embeddings by applying sinusoidal functions to spatial
    coordinates. The output channels are split equally between y-axis and x-axis
    embeddings, each encoded with interleaved sin/cos pairs.

    Note:
        ``embed_dim`` must be an even number because it is split into two halves:
        the first ``embed_dim // 2`` channels encode the y-axis position and the
        remaining ``embed_dim // 2`` channels encode the x-axis position. Within
        each half, adjacent pairs of channels form (sin, cos) pairs at increasing
        frequencies. Therefore, if your Transformer expects a hidden dimension of
        ``d_model``, you should set ``embed_dim = d_model`` (not ``d_model // 2``).

    Example::

        >>> embed = SinePositionEmbedding(embed_dim=256)
        >>> mask = torch.zeros(1, 32, 32, dtype=torch.bool)
        >>> pos = embed(mask)  # shape: [1, 256, 32, 32]
    """

    def __init__(
        self,
        embed_dim: int = 256,
        temperature: int = 1000,
        scale: float | None = None,
        eps: float = 1e-6,
        offset: float = 0.0,
        normalize: bool = False,
    ):
        """
        Args:
            embed_dim (int): The total feature dimension of the output position embedding.
                Must be even. The output is split into two halves (y and x), each of size
                ``embed_dim // 2``. Set this equal to your model's hidden dimension
                (e.g., ``d_model``), not half of it.
            temperature (int, optional): The temperature used for scaling
                the position embedding. Higher values spread frequencies more
                uniformly across dimensions. Default: 1000.
            scale (float, optional): A scale factor that scales the position
                embedding. The scale will be used only when `normalize` is True.
                Default: 2*pi.
            eps (float, optional): A value added to the denominator for numerical
                stability. Default: 1e-6.
            offset (float): An offset added to embed when doing normalization.
            normalize (bool, optional): Whether to normalize the position embedding.
                Default: False.
        """
        super().__init__()

        if scale is None:
            scale = 2 * math.pi

        if normalize and not isinstance(scale, (float, int)):
            raise TypeError(f"Expected scale to be a float or int when normalize=True, but got {type(scale).__name__}.")

        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, but got {embed_dim}.")

        self.embed_dim = embed_dim
        self.half_dim = embed_dim // 2
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: torch.Tensor):
        """
        Args:
            mask (Tensor): ByteTensor mask of shape [batch_size, height, width]. Non-zero values representing ignored positions,
                while zero values means valid positions for each image in the batch.

        Returns:
            pos (Tensor): The position embedding of shape [batch_size, embed_dim, height, width].
                The output channel dimension matches `embed_dim` directly, with half of the
                channels assigned to y-axis embeddings and the other half to x-axis embeddings.
        """
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = torch.arange(self.half_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.half_dim)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        B, H, W = mask.size()
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).view(B, H, W, -1)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
