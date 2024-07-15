from typing import Tuple
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransposeConv2d(torch.nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super(TransposeConv2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        height, width = x.size()[-2:]
        return super(TransposeConv2d, self).forward(x, output_size=(2*height, 2*width))

def create_upscaling_layer(in_channels: int, scale_factor: int, layer_name: str='upconv', interpolation: str='bilinear', activation: nn.Module=nn.ReLU):

    if layer_name == 'transposeconv':
        if scale_factor > 2:
            upscale_blocks = []
            for i in range(scale_factor // 2):
                block = nn.Sequential(*[
                    TransposeConv2d(in_channels, in_channels, 
                                    kernel_size=3, 
                                    stride=2, 
                                    padding=1, 
                                    groups=in_channels,
                                    bias=False),
                    nn.BatchNorm2d(in_channels),
                    activation(inplace=True)
                ])
                upscale_blocks.append(block)
            return nn.Sequential(*upscale_blocks)

        else:
            block = nn.Sequential(*[
                TransposeConv2d(in_channels, in_channels, 
                                kernel_size=3, 
                                stride=2, 
                                padding=1, 
                                groups=in_channels,
                                bias=False),
                nn.BatchNorm2d(in_channels),
                activation(inplace=True)
            ])
            return block

    elif layer_name == 'transposeconv_nogroup':
        if scale_factor > 2:
            upscale_blocks = []
            for i in range(scale_factor // 2):
                block = nn.Sequential(*[
                    TransposeConv2d(in_channels, in_channels, 
                                    kernel_size=3, 
                                    stride=2, 
                                    padding=1, 
                                    bias=False),
                    nn.BatchNorm2d(in_channels),
                    activation(inplace=True)
                ])
                upscale_blocks.append(block)
            return nn.Sequential(*upscale_blocks)

        else:
            block = nn.Sequential(*[
                TransposeConv2d(in_channels, in_channels, 
                                kernel_size=3, 
                                stride=2, 
                                padding=1, 
                                bias=False),
                nn.BatchNorm2d(in_channels),
                activation(inplace=True)
            ])
            return block

    elif layer_name == 'interpolation':
        return nn.Upsample(scale_factor=scale_factor, mode=interpolation, align_corners=True)
    
    else:
        
        raise NotImplementedError()
