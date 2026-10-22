import math

import torch
import torch.nn as nn
from overdrive.layers import ShapeSpec
from overdrive.layers import get_conv2d
from overdrive.layers import weight_init
from overdrive.modeling.neck import Neck

__all__ = ["CenterNetDLANeck"]


def _fill_up_weights(up: nn.ConvTranspose2d):
    """
    Initialize depthwise transposed convolution with bilinear filters.
    """
    f = math.ceil(up.kernel_size[0] / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    kernel = 1 - (torch.arange(up.kernel_size[0], device=up.weight.device, dtype=up.weight.dtype) / f - c).abs()
    with torch.no_grad():
        up.weight.copy_(kernel[:, None] * kernel[None, :])


class IDAUp(nn.Module):
    """
    Project, upsample and iteratively aggregate a range of feature levels.
    """

    def __init__(
        self,
        out_channels: int,
        channels: list[int],
        up_factors: list[int],
        conv_op: str = "DCNv2",
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
    ):
        super().__init__()

        for i, (in_channels, factor) in enumerate(zip(channels[1:], up_factors[1:], strict=True), start=1):
            proj = get_conv2d(
                conv_op,
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                bias=True,
                norm=norm,
                activation=activation,
            )
            weight_init.bias_fill(proj, 0)
            node = get_conv2d(
                conv_op,
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                bias=True,
                norm=norm,
                activation=activation,
            )
            weight_init.bias_fill(node, 0)

            up = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=2 * factor,
                stride=factor,
                padding=factor // 2,
                groups=out_channels,
                bias=False,
            )
            _fill_up_weights(up)

            self.add_module(f"proj_{i}", proj)
            self.add_module(f"up_{i}", up)
            self.add_module(f"node_{i}", node)

    def forward(self, layers: list[torch.Tensor], startp: int, endp: int) -> list[torch.Tensor]:
        for i in range(startp + 1, endp):
            project = getattr(self, f"proj_{i - startp}")
            upsample = getattr(self, f"up_{i - startp}")
            node = getattr(self, f"node_{i - startp}")
            layers[i] = upsample(project(layers[i]))
            layers[i] = node(layers[i] + layers[i - 1])
        return layers


class DLAUp(nn.Module):
    """
    Aggregate feature levels progressively from the deepest to the shallowest.
    """

    def __init__(
        self,
        channels: list[int],
        scales: list[int],
        conv_op: str = "DCNv2",
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
    ):
        super().__init__()

        in_channels = list(channels)
        scales = list(scales)
        self.num_stages = len(channels) - 1

        for i in range(self.num_stages):
            startp = len(channels) - i - 2
            up_factors = [scale // scales[startp] for scale in scales[startp:]]
            ida = IDAUp(
                channels[startp],
                in_channels[startp:],
                up_factors,
                conv_op=conv_op,
                norm=norm,
                activation=activation,
            )
            self.add_module(f"ida_{i}", ida)
            scales[startp + 1 :] = [scales[startp]] * (len(channels) - startp - 1)
            in_channels[startp + 1 :] = [channels[startp]] * (len(channels) - startp - 1)

    def forward(self, layers: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = [layers[-1]]
        for i in range(self.num_stages):
            ida = getattr(self, f"ida_{i}")
            layers = ida(layers, len(layers) - i - 2, len(layers))
            outputs.insert(0, layers[-1])
        return outputs


class CenterNetDLANeck(Neck):
    """
    CenterNet DLA aggregation with input features ordered by increasing stride.
    """

    def __init__(
        self,
        input_shape: dict[str, ShapeSpec],
        in_features: list[str],
        conv_op: str = "DCNv2",
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
    ):
        super().__init__()

        self.in_features = in_features
        if len(self.in_features) < 2:
            raise ValueError("CenterNetDLANeck requires at least two input features.")

        channels = [input_shape[name].channels for name in self.in_features]
        strides = [input_shape[name].stride for name in self.in_features]
        if any(current != previous * 2 for previous, current in zip(strides[:-1], strides[1:], strict=True)):
            raise ValueError("Input features must be ordered by increasing stride with adjacent strides doubling.")

        scales = [stride // strides[0] for stride in strides]
        self.dla_up = DLAUp(channels, scales, conv_op=conv_op, norm=norm, activation=activation)
        self.ida_up = IDAUp(channels[0], channels[:-1], scales[:-1], conv_op=conv_op, norm=norm, activation=activation)
        self._size_divisibility = strides[-1]

        out_feature = f"p{int(math.log2(strides[0]))}"
        self._out_features = [out_feature]
        self._out_feature_channels[out_feature] = channels[0]
        self._out_feature_strides[out_feature] = strides[0]

    def forward(self, bottom_up_features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        layers = self.dla_up([bottom_up_features[name] for name in self.in_features])[:-1]
        layers = self.ida_up(layers, 0, len(layers))
        return {self._out_features[0]: layers[-1]}

    @property
    def padding_constraints(self):
        return {
            "size_divisibility": self._size_divisibility,
            "pad_size": 0,
        }
