import math

import torch
import torch.nn as nn
from overdrive.layers import ShapeSpec
from overdrive.layers import get_activation
from overdrive.layers import get_conv2d
from overdrive.layers import get_norm
from overdrive.modeling.neck import Neck

__all__ = ["CenterNetResNetNeck"]


class CenterNetResNetNeck(Neck):
    """
    support neck proposed in CenterNet (https://arxiv.org/abs/1904.07850).
    """

    def __init__(
        self,
        input_shape: dict[str, ShapeSpec],
        in_features: list[str],
        num_deconv_kernel_sizes: list[int],
        num_deconv_channels: list[int],
        conv_op: str = "DCNv2",
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
    ):
        super().__init__()

        if len(num_deconv_kernel_sizes) != len(num_deconv_channels):
            raise ValueError(
                "Each deconvolution layer requires one kernel size and one channel count, "
                "so num_deconv_kernel_sizes and num_deconv_channels must have the same length. "
                f"Got num_deconv_kernel_sizes={num_deconv_kernel_sizes!r} (length={len(num_deconv_kernel_sizes)}), "
                f"num_deconv_channels={num_deconv_channels!r} (length={len(num_deconv_channels)})."
            )

        in_channels = input_shape[in_features[0]].channels
        cur_stride = input_shape[in_features[0]].stride
        self.in_features = in_features
        self._size_divisibility = cur_stride

        layers = []
        for i in range(len(num_deconv_kernel_sizes)):
            conv = get_conv2d(
                conv2d=conv_op,
                in_channels=in_channels,
                out_channels=num_deconv_channels[i],
                kernel_size=3,
                stride=1,
                norm=norm,
                activation=activation,
            )
            layers.append(conv)
            deconv = nn.ConvTranspose2d(
                num_deconv_channels[i],
                num_deconv_channels[i],
                kernel_size=num_deconv_kernel_sizes[i],
                stride=2,
                padding=1,
                bias=False,
            )
            layers.append(deconv)
            layers.append(get_norm(norm, num_deconv_channels[i]))
            layers.append(get_activation(activation))

            in_channels = num_deconv_channels[i]
            cur_stride = cur_stride // 2

        layers = [layer for layer in layers if layer is not None]
        self.layers = nn.Sequential(*layers)

        stage = int(math.log2(cur_stride))
        self._out_feature_strides[f"p{stage}"] = cur_stride
        self._out_feature_channels[f"p{stage}"] = num_deconv_channels[i]
        self._out_features = [f"p{stage}"]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.reset_parameters()
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2.0 * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, bottom_up_features: dict[str, torch.Tensor]):
        features = bottom_up_features[self.in_features[0]]
        features = self.layers(features)

        return {f: res for f, res in zip(self._out_features, [features], strict=True)}

    @property
    def padding_constraints(self):
        return {
            "size_divisibility": self._size_divisibility,
            "pad_size": 0,
        }
