import torch
import torch.nn as nn

from overdrive.layers import CNNBlock
from overdrive.layers import Conv2d
from overdrive.layers import ShapeSpec
from overdrive.layers import weight_init

from .base import Backbone
from .base import validate_factory_kwargs

__all__ = ["ShuffleNetV2"]


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """
    N, C, H, W = x.size()
    if C % groups != 0:
        raise ValueError(f"`num_channels` should be divisible by `groups`, but got num_channels={C}, groups={groups}")

    channels_per_group = C // groups
    x = x.view(N, groups, channels_per_group, H, W)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(N, -1, H, W)
    return x


class InvertedResidual(CNNBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
    ):
        super().__init__(in_channels, out_channels, stride)

        hidden_dim = out_channels // 2
        if (self.stride == 1) and (in_channels != hidden_dim << 1):
            raise ValueError(
                f"Invalid combination of stride {stride}, in_channels {in_channels} and out_channels {out_channels}. " f"If stride == 1 then in_channels should be equal to out_channels // 2 << 1."
            )

        self.branch1 = (
            nn.Sequential(
                Conv2d(in_channels, in_channels, kernel_size=3, stride=self.stride, groups=in_channels, norm=norm),
                Conv2d(in_channels, hidden_dim, kernel_size=1, norm=norm, activation=activation),
            )
            if self.stride > 1
            else None
        )

        in_channels = in_channels if self.stride > 1 else hidden_dim
        self.branch2 = nn.Sequential(
            Conv2d(in_channels, hidden_dim, kernel_size=1, norm=norm, activation=activation),
            Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=self.stride, groups=hidden_dim, norm=norm),
            Conv2d(hidden_dim, hidden_dim, kernel_size=1, norm=norm, activation=activation),
        )

    def forward(self, x: torch.Tensor):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)
        return out


class Stem(CNNBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
    ):
        super().__init__(in_channels, out_channels, 4)

        self.conv = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, norm=norm, activation=activation)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        weight_init.c2_msra_fill(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x


class ShuffleNetV2(Backbone):
    """
    support ShuffleNet v2 (https://arxiv.org/abs/1807.11164).
    """

    def __init__(
        self,
        stages_repeats: list[int],
        stages_out_channels: list[int],
        stem_out_channels: int,
        conv5_out_channels: int,
        input_shape: ShapeSpec | None = None,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
        out_features: list[str] | None = None,
        freeze_at: int = 0,
        num_classes: int | None = None,
    ):
        super().__init__()

        if input_shape is None:
            input_shape = ShapeSpec(channels=3)

        self.num_classes = num_classes

        if out_features is not None:
            num_stages = max([{"stage2": 1, "stage3": 2, "stage4": 3, "conv5": 3}.get(f, 0) for f in out_features])
            stages_repeats = stages_repeats[:num_stages]
            stages_out_channels = stages_out_channels[:num_stages]

        in_channels = input_shape.channels
        out_channels = stem_out_channels
        self.stem = Stem(in_channels, out_channels, norm, activation)
        current_stride = self.stem.stride
        self._out_feature_strides["stem"] = current_stride
        self._out_feature_channels["stem"] = out_channels
        in_channels = out_channels

        self.stages = []
        for idx, (repeats, out_channels) in enumerate(zip(stages_repeats, stages_out_channels, strict=True)):
            name = f"stage{idx + 2}"
            seq = [InvertedResidual(in_channels, out_channels, 2, norm, activation)]
            for _ in range(repeats - 1):
                seq.append(InvertedResidual(out_channels, out_channels, 1, norm, activation))
            self.add_module(name, nn.Sequential(*seq))
            self.stages.append(name)
            current_stride *= 2
            self._out_feature_channels[name] = out_channels
            self._out_feature_strides[name] = current_stride
            in_channels = out_channels

        if out_features is not None and "conv5" in out_features:
            name = "conv5"
            self.conv5 = Conv2d(in_channels, conv5_out_channels, kernel_size=1, norm=norm, activation=activation)
            self.stages.append(name)
            self._out_feature_strides[name] = current_stride
            self._out_feature_channels[name] = conv5_out_channels
            in_channels = conv5_out_channels

        if num_classes is not None:
            self.linear = nn.Linear(in_channels, num_classes)
            weight_init.normal_fill(self.linear, std=0.01)

        if out_features is None:
            out_features = [self.stages[-1]]

        self._out_features = out_features
        if len(self._out_features) == 0:
            raise ValueError("out_features must contain at least one feature name.")

        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            if out_feature not in children:
                raise ValueError(f"Available children: {', '.join(children)}")

        self.freeze(freeze_at)

    @staticmethod
    def build(name, **kwargs):
        predefined = {
            "ShuffleNet-V2-x0.5": shufflenet_v2_x0_5,
            "ShuffleNet-V2-x1.0": shufflenet_v2_x1_0,
            "ShuffleNet-V2-x1.5": shufflenet_v2_x1_5,
            "ShuffleNet-V2-x2.0": shufflenet_v2_x2_0,
        }

        if name not in predefined:
            raise ValueError(f"`{name}` is not predefined for ShuffleNetV2. Available names: {', '.join(predefined)}.")

        return predefined[name](**kwargs)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}

        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x

        for stage in self.stages:
            x = getattr(self, stage)(x)
            if stage in self._out_features:
                outputs[stage] = x

        if self.num_classes is not None:
            x = x.mean([2, 3])
            x = self.linear(x)
            outputs["linear"] = x

        return outputs

    def freeze(self, freeze_at=0):
        if freeze_at >= 1:
            self.stem.freeze()

        for idx, stage_name in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                module = getattr(self, stage_name)
                if stage_name == "conv5":
                    module.freeze()
                else:
                    for block in module.children():
                        block.freeze()


@validate_factory_kwargs("stages_repeats", "stages_out_channels", "stem_out_channels", "conv5_out_channels")
def shufflenet_v2_x0_5(**kwargs):
    return ShuffleNetV2(
        stages_repeats=[4, 8, 4],
        stages_out_channels=[48, 96, 192],
        stem_out_channels=24,
        conv5_out_channels=1024,
        **kwargs,
    )


@validate_factory_kwargs("stages_repeats", "stages_out_channels", "stem_out_channels", "conv5_out_channels")
def shufflenet_v2_x1_0(**kwargs):
    return ShuffleNetV2(
        stages_repeats=[4, 8, 4],
        stages_out_channels=[116, 232, 464],
        stem_out_channels=24,
        conv5_out_channels=1024,
        **kwargs,
    )


@validate_factory_kwargs("stages_repeats", "stages_out_channels", "stem_out_channels", "conv5_out_channels")
def shufflenet_v2_x1_5(**kwargs):
    return ShuffleNetV2(
        stages_repeats=[4, 8, 4],
        stages_out_channels=[176, 352, 704],
        stem_out_channels=24,
        conv5_out_channels=1024,
        **kwargs,
    )


@validate_factory_kwargs("stages_repeats", "stages_out_channels", "stem_out_channels", "conv5_out_channels")
def shufflenet_v2_x2_0(**kwargs):
    return ShuffleNetV2(
        stages_repeats=[4, 8, 4],
        stages_out_channels=[244, 488, 976],
        stem_out_channels=24,
        conv5_out_channels=2048,
        **kwargs,
    )
