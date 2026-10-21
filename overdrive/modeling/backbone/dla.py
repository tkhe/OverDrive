import math

import torch
import torch.nn as nn
from torch.utils import checkpoint

from overdrive.layers import CNNBlock
from overdrive.layers import Conv2d
from overdrive.layers import ShapeSpec
from overdrive.layers import get_activation
from overdrive.layers import weight_init

from .base import Backbone
from .base import validate_factory_kwargs

__all__ = ["DLA"]


class DLABasic(CNNBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
        cardinality: int = 1,
        base_width: int = 64,
    ):
        super().__init__(in_channels, out_channels, stride)

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, dilation=dilation, norm=norm, activation=activation)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, dilation=dilation, norm=norm)
        self.activation = get_activation(activation)

        for layer in [self.conv1, self.conv2]:
            weight_init.c2_msra_fill(layer)

    def forward(self, x: torch.Tensor, shortcut: torch.Tensor | None = None, children: list[torch.Tensor] | None = None):
        if shortcut is None:
            shortcut = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += shortcut
        out = self.activation(out)

        return out


class DLABottleneck(CNNBlock):
    expansion: int = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        cardinality: int = 1,
        base_width: int = 64,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
    ):
        super().__init__(in_channels, out_channels, stride)

        mid_channels = int(math.floor(out_channels * (base_width / 64)) * cardinality)
        mid_channels = mid_channels // self.expansion

        self.conv1 = Conv2d(in_channels, mid_channels, kernel_size=1, norm=norm, activation=activation)
        self.conv2 = Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, dilation=dilation, groups=cardinality, norm=norm, activation=activation)
        self.conv3 = Conv2d(mid_channels, out_channels, kernel_size=1, norm=norm)
        self.activation = get_activation(activation)

        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)

    def forward(self, x: torch.Tensor, shortcut: torch.Tensor | None = None, children: list[torch.Tensor] | None = None):
        if shortcut is None:
            shortcut = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += shortcut
        out = self.activation(out)

        return out


class DLABottle2neck(CNNBlock):
    expansion = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        scale: int = 4,
        cardinality: int = 8,
        base_width: int = 4,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
    ):
        super().__init__(in_channels, out_channels, stride)

        self.is_first = stride > 1
        self.scale = scale
        mid_channels = int(math.floor(out_channels * (base_width / 64)) * cardinality)
        mid_channels = mid_channels // self.expansion
        self.width = mid_channels

        self.conv1 = Conv2d(in_channels, mid_channels * scale, kernel_size=1, norm=norm, activation=activation)

        num_scale_convs = max(1, scale - 1)
        self.convs = nn.ModuleList()
        for _ in range(num_scale_convs):
            self.convs.append(Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, dilation=dilation, groups=cardinality, norm=norm, activation=activation))

        self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1) if self.is_first else None
        self.conv3 = Conv2d(mid_channels * scale, out_channels, kernel_size=1, norm=norm)
        self.activation = get_activation(activation)

        weight_init.c2_msra_fill(self.conv1)
        for conv in self.convs:
            weight_init.c2_msra_fill(conv)
        weight_init.c2_msra_fill(self.conv3)

    def forward(self, x: torch.Tensor, shortcut: torch.Tensor | None = None, children: list[torch.Tensor] | None = None):
        if shortcut is None:
            shortcut = x

        out = self.conv1(x)

        spx = torch.split(out, self.width, 1)
        spo = []
        sp = spx[0]
        for i, conv in enumerate(self.convs):
            if i == 0 or self.is_first:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = conv(sp)
            spo.append(sp)
        if self.scale > 1:
            if self.pool is not None:
                spo.append(self.pool(spx[-1]))
            else:
                spo.append(spx[-1])
        out = torch.cat(spo, 1)

        out = self.conv3(out)

        out += shortcut
        out = self.activation(out)

        return out


class DLARoot(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        shortcut: bool,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
    ):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size=1, padding=(kernel_size - 1) // 2, norm=norm)
        self.activation = get_activation(activation)
        self.shortcut = shortcut

        weight_init.c2_msra_fill(self.conv)

    def forward(self, x_children: list[torch.Tensor]):
        x = self.conv(torch.cat(x_children, 1))
        if self.shortcut:
            x += x_children[0]
        x = self.activation(x)
        return x


class DLATree(nn.Module):
    def __init__(
        self,
        levels: int,
        block: type[CNNBlock],
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        cardinality: int = 1,
        base_width: int = 64,
        level_root: bool = False,
        root_dim: int = 0,
        root_kernel_size: int = 1,
        root_shortcut: bool = False,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
    ):
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels

        self.downsample = nn.MaxPool2d(stride, stride=stride) if stride > 1 else nn.Identity()
        self.project = nn.Identity()

        if levels == 1:
            self.tree1 = block(
                in_channels,
                out_channels,
                stride,
                dilation=dilation,
                cardinality=cardinality,
                base_width=base_width,
                norm=norm,
                activation=activation,
            )
            self.tree2 = block(
                out_channels,
                out_channels,
                1,
                dilation=dilation,
                cardinality=cardinality,
                base_width=base_width,
                norm=norm,
                activation=activation,
            )
            if in_channels != out_channels:
                self.project = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, norm=norm)
                weight_init.c2_msra_fill(self.project)
            self.root = DLARoot(root_dim, out_channels, root_kernel_size, root_shortcut, norm=norm, activation=activation)
        else:
            self.tree1 = DLATree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                dilation=dilation,
                cardinality=cardinality,
                base_width=base_width,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                root_shortcut=root_shortcut,
                norm=norm,
                activation=activation,
            )
            self.tree2 = DLATree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                dilation=dilation,
                cardinality=cardinality,
                base_width=base_width,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                root_shortcut=root_shortcut,
                norm=norm,
                activation=activation,
            )
            self.root = None

        self.level_root = level_root
        self.root_dim = root_dim
        self.levels = levels

    def forward(self, x: torch.Tensor, shortcut: torch.Tensor | None = None, children: list[torch.Tensor] | None = None):
        if children is None:
            children = []
        bottom = self.downsample(x)
        shortcut = self.project(bottom)
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, shortcut)
        if self.root is not None:
            x2 = self.tree2(x1)
            x = self.root([x2, x1] + children)
        else:
            children.append(x1)
            x = self.tree2(x1, None, children)
        return x


class DLA(Backbone):
    """
    support Deep Layer Aggregation (https://arxiv.org/abs/1707.06484).
    """

    def __init__(
        self,
        levels: list[int],
        channels: list[int],
        block_class: type[CNNBlock],
        input_shape: ShapeSpec | None = None,
        cardinality: int = 1,
        base_width: int = 64,
        shortcut_root: bool = False,
        use_checkpoint: bool = False,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
        out_features: list[str] | None = None,
        freeze_at: int = 0,
        num_classes: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        if input_shape is None:
            input_shape = ShapeSpec(channels=3)

        self.num_classes = num_classes

        if out_features is not None:
            num_stages = max([{"level2": 1, "level3": 2, "level4": 3, "level5": 4}.get(f, 0) for f in out_features])
            levels = levels[: num_stages + 2]
            channels = channels[: num_stages + 2]

        self.use_checkpoint = use_checkpoint

        self.stem = Conv2d(input_shape.channels, channels[0], kernel_size=7, stride=1, norm=norm, activation=activation)
        weight_init.c2_msra_fill(self.stem)
        self._out_feature_strides["stem"] = 1
        self._out_feature_channels["stem"] = channels[0]
        in_channels = channels[0]

        self.stages = []
        cur_stride = 1
        for i, (num_level, out_channels) in enumerate(zip(levels, channels, strict=True)):
            stride = 1 if i == 0 else 2
            cur_stride *= stride
            if i < 2:
                level = self._make_conv_level(in_channels, out_channels, num_level, stride=stride, norm=norm, activation=activation)
            else:
                level = DLATree(
                    num_level,
                    block_class,
                    in_channels,
                    out_channels,
                    stride=stride,
                    cardinality=cardinality,
                    base_width=base_width,
                    level_root=i > 2,
                    root_shortcut=shortcut_root,
                    norm=norm,
                    activation=activation,
                )
            name = f"level{i}"
            self.add_module(name, level)
            self.stages.append(name)
            self._out_feature_channels[name] = out_channels
            self._out_feature_strides[name] = cur_stride
            in_channels = out_channels

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(p=dropout)
            self.linear = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True)
            weight_init.c2_msra_fill(self.linear)

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
            "DLA-34": dla34,
            "DLA-46-C": dla46_c,
            "DLA-60": dla60,
            "DLA-102": dla102,
            "DLA-169": dla169,
            "DLA-X-46-C": dla_x_46_c,
            "DLA-X-60-C": dla_x_60_c,
            "DLA-X-60": dla_x_60,
            "DLA-X-102": dla_x_102,
        }

        if name not in predefined:
            raise ValueError(f"`{name}` is not predefined for DLA. Available names: {', '.join(predefined)}.")

        return predefined[name](**kwargs)

    @staticmethod
    def _make_conv_level(
        in_channels: int,
        out_channels: int,
        convs: int,
        stride: int = 1,
        dilation: int = 1,
        norm: str | dict = "BN",
        activation: str | dict = "ReLU",
    ):
        modules = []
        for i in range(convs):
            conv = Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride if i == 0 else 1,
                dilation=dilation,
                norm=norm,
                activation=activation,
            )
            weight_init.c2_msra_fill(conv)
            modules.append(conv)
            in_channels = out_channels
        return nn.Sequential(*modules)

    def freeze(self, freeze_at: int = 0):
        if freeze_at >= 1:
            self.stem.freeze()

        for idx, stage_name in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                stage = getattr(self, stage_name)
                for p in stage.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}

        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x

        for stage_name in self.stages:
            if self.use_checkpoint and x.requires_grad:
                x = checkpoint.checkpoint(getattr(self, stage_name), x, use_reentrant=False)
            else:
                x = getattr(self, stage_name)(x)
            if stage_name in self._out_features:
                outputs[stage_name] = x

        if self.num_classes is not None:
            x = self.avgpool(x)
            x = self.dropout(x)
            x = self.linear(x)
            x = torch.flatten(x, 1)
            outputs["linear"] = x

        return outputs


@validate_factory_kwargs("levels", "channels", "block_class")
def dla34(**kwargs):
    return DLA(
        levels=[1, 1, 1, 2, 2, 1],
        channels=[16, 32, 64, 128, 256, 512],
        block_class=DLABasic,
        **kwargs,
    )


@validate_factory_kwargs("levels", "channels", "block_class")
def dla46_c(**kwargs):
    return DLA(
        levels=[1, 1, 1, 2, 2, 1],
        channels=[16, 32, 64, 64, 128, 256],
        block_class=DLABottleneck,
        **kwargs,
    )


@validate_factory_kwargs("levels", "channels", "block_class", "cardinality", "base_width")
def dla_x_46_c(**kwargs):
    return DLA(
        levels=[1, 1, 1, 2, 2, 1],
        channels=[16, 32, 64, 64, 128, 256],
        block_class=DLABottleneck,
        cardinality=32,
        base_width=4,
        **kwargs,
    )


@validate_factory_kwargs("levels", "channels", "block_class", "cardinality", "base_width")
def dla_x_60_c(**kwargs):
    return DLA(
        levels=[1, 1, 1, 2, 3, 1],
        channels=[16, 32, 64, 64, 128, 256],
        block_class=DLABottleneck,
        cardinality=32,
        base_width=4,
        **kwargs,
    )


@validate_factory_kwargs("levels", "channels", "block_class")
def dla60(**kwargs):
    return DLA(
        levels=[1, 1, 1, 2, 3, 1],
        channels=[16, 32, 128, 256, 512, 1024],
        block_class=DLABottleneck,
        **kwargs,
    )


@validate_factory_kwargs("levels", "channels", "block_class", "cardinality", "base_width")
def dla_x_60(**kwargs):
    return DLA(
        levels=[1, 1, 1, 2, 3, 1],
        channels=[16, 32, 128, 256, 512, 1024],
        block_class=DLABottleneck,
        cardinality=32,
        base_width=4,
        **kwargs,
    )


@validate_factory_kwargs("levels", "channels", "block_class", "shortcut_root")
def dla102(**kwargs):
    return DLA(
        levels=[1, 1, 1, 3, 4, 1],
        channels=[16, 32, 128, 256, 512, 1024],
        block_class=DLABottleneck,
        shortcut_root=True,
        **kwargs,
    )


@validate_factory_kwargs("levels", "channels", "block_class", "shortcut_root", "cardinality", "base_width")
def dla_x_102(**kwargs):
    return DLA(
        levels=[1, 1, 1, 3, 4, 1],
        channels=[16, 32, 128, 256, 512, 1024],
        block_class=DLABottleneck,
        shortcut_root=True,
        cardinality=32,
        base_width=4,
        **kwargs,
    )


@validate_factory_kwargs("levels", "channels", "block_class", "shortcut_root")
def dla169(**kwargs):
    return DLA(
        levels=[1, 1, 2, 3, 5, 1],
        channels=[16, 32, 128, 256, 512, 1024],
        block_class=DLABottleneck,
        shortcut_root=True,
        **kwargs,
    )
