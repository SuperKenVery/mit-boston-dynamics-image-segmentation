#!/usr/bin/env python3

# Modified from https://github.com/Lornatang/VGG-PyTorch/blob/main/model.py
# Apache License

from typing import cast, Dict, List, Union

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

vgg_cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
    "vgg19s": [64, 64, "M", 128, 128, "M", 256, 256, "S", 256, 256, "M", 512, 512, "S", 512, 512, "M", 256, 256, "S", 256, 256, "M", 40],
    "vgg11s": [64, "M", 128, "M", 256, "S", 256, "M", 512, "S", 512, "M", 256, "S", 256, "M", 40],
    "vggtiny": [64, "M", 128, "M", 256, "S", 40],
}


def _make_layers(vgg_cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers = nn.Sequential()
    in_channels = 3
    for v in vgg_cfg:
        if v == "M":    # MaxPooling
            layers.append(nn.MaxPool2d((2, 2), (2, 2)))
        elif v=="S":    # PixelShuffle
            upscale_factor = 4
            layers.append(nn.PixelShuffle(upscale_factor))
            in_channels = in_channels // upscale_factor**2
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, (3, 3), (1, 1), (1, 1))
            layers.append(conv2d)
            if batch_norm: layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(True))
            in_channels = v

    return layers


class Segmenter(nn.Module):
    """An image segment model derived from VGG

    Reference paper: Multi-view Self-supervised Deep Learning for 6D Pose Estimation in the Amazon Picking Challenge
    We don't aim to 1:1 reproduce the original model, just modify a VGG and see what we get~

    Input: (Batch, Channels, Height, Width) RGB image
    Output: (Batch, 40 (number of classes), Height, Weight)
        Classification and segmentation for the image. For each pixel, it outputs a classification. 39 for objects and 1 for background.
    """

    def __init__(self, variant: str = "19s"):
        super().__init__()
        vgg_cfg = vgg_cfgs["vgg"+variant]

        self.feature_extractor = _make_layers(vgg_cfg, True)

        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        out = self.feature_extractor(x)
        out = F.adaptive_avg_pool2d(out, (H, W))    # (B, 40, H, W)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

if __name__ == '__main__':
    model = Segmenter(variant="tiny").to("mps")

    B, H, W = 1, 100, 80
    fake_data = torch.randn(B, 3, H, W).to("mps")
    output = model(fake_data)
    assert output.shape==(B, 40, H, W)
    print("Test passed")
