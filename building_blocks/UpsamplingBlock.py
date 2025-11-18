from typing import Tuple

import torch
from torch import nn


class UpsamplingBlock(nn.Module):
    """
    Upsampling Block (upsampling layer) as specified by the paper
    This is composed of a 2d bilinear upsampling layer followed by a convolutional layer, BatchNorm layer, and ReLU activation
    """

    def __init__(self, in_channels: int, out_channels: int, size: Tuple):
        """
        Create the layers for the upsampling block

        :param in_channels:   number of input features to the block
        :param out_channels:  number of output features from this entire block
        :param scale_factor:  tuple to determine how to scale the dimensions
        :param residual:      residual from the opposite dense block to add before upsampling
        """
        super().__init__()
        # blinear vs trilinear kernel size and padding
        if size[0] == 2:
            d_kernel_size = 3
            d_padding = 1
        else:
            d_kernel_size = 1
            d_padding = 0

        self.upsample = nn.Upsample(
            scale_factor=size, mode="trilinear", align_corners=True
        )
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(d_kernel_size, 3, 3),
                padding=(d_padding, 1, 1),
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, projected_residual):
        """
        Forward pass through the block

        :param x:  image tensor
        :return:   output of the forward pass
        """
        residual = torch.cat(
            (self.upsample(x), self.upsample(projected_residual)),
            dim=1,
        )
        return self.conv(residual)
