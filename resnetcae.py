import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint_sequential

import torchvision.transforms as transforms
import preprocessing
import os
import pandas as pd

from torch import Tensor
from torchvision.models.resnet import ResNet
import torchvision.models.resnet as resnet

from typing import Any, Callable, List, Optional, Type, Union


def convtrans3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, output_padding: int = 0) -> nn.ConvTranspose2d:
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        output_padding=output_padding,
    )


def convtrans1x1(in_planes: int, out_planes: int, stride: int = 1, padding: int = 0, output_padding: int = 0) -> nn.ConvTranspose2d:
    """1x1 transposed convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, padding=padding, output_padding=output_padding)


class BasicBlockTranspose(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers upsample the input when stride != 1
        self.conv1 = convtrans3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convtrans3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.id_in = None

    def set_id(self, id: Tensor) -> None:
        self.id_in = id

    def forward(self, x: Tensor) -> Tensor:
        identity = self.id_in
        if self.downsample is not None:
            identity = self.downsample(self.id_in)

        out = x - identity
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class BottleneckTranspose(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv3 = convtrans1x1(planes * self.expansion, width)
        self.bn3 = norm_layer(width)
        self.conv2 = convtrans3x3(width, width, stride, groups, dilation, output_padding=stride-1)
        self.bn2 = norm_layer(width)
        self.conv1 = convtrans1x1(width, inplanes)
        self.bn1 = norm_layer(inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv3(x)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetCAE(ResNet):
    def __init__(
            self,
            block: Type[Union[resnet.BasicBlock, resnet.Bottleneck]],
            layers: List[int],
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        # This will do all the initializing for the normal ResNet model. We'll make some adjustments after.
        super().__init__(
            block=block,
            layers=layers,
            num_classes=1,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer)
        # Original class uses color images, we're using grayscale. Thus, we changed in_channels from 3 to 1.
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        # Need to change the block types to the custom ones above for the decoding step.
        if type(block) == resnet.BasicBlock:
            block = BasicBlockTranspose
        else:
            block = BottleneckTranspose

        self.layer4trans = self._make_layer_transpose(block, 512, 1024, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.layer3trans = self._make_layer_transpose(block, 256, 512, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer2trans = self._make_layer_transpose(block, 128, 256, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer1trans = self._make_layer_transpose(block, 64, 64, layers[0])
        self.maxunpool = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans1 = nn.ConvTranspose2d(self.inplanes, 1, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False)
        self.bntrans1 = norm_layer(1)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer_transpose(
            self,
            block: Type[Union[BasicBlockTranspose, BottleneckTranspose]],
            planes: int,
            outplanes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or outplanes == planes:
            upsample = nn.Sequential(
                convtrans1x1(self.inplanes, outplanes, stride, output_padding=stride-1),
                norm_layer(outplanes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        self.inplanes = outplanes
        layers.append(
            block(
                outplanes, planes, stride, upsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )

        return nn.Sequential(*layers)

    def _forward_impl_encoder(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def _forward_impl_decoder(self, x: Tensor) -> Tensor:

        x = self.layer4trans(x)
        x = self.layer3trans(x)
        x = self.layer2trans(x)
        x = self.layer1trans(x)

        x = self.maxunpool(x)
        x = self.convtrans1(x)
        x = self.bntrans1(x)
        x = self.relu(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._forward_impl_encoder(x)
        x = self._forward_impl_decoder(x)
        return x
