import torch
from torch import nn


class TransitionBlock(nn.Module):
    """
    Transition Block (transition layer) as specified by the paper
    This is composed of a pointwise convolution followed by a pointwise convolution with higher stride to reduce the image size
    We use BatchNorm and ReLU after the first convolution, but not the second

    Some notes on architecture based on the paper:
      - The number of channels is always 32
      - The depth is always 3
    """

    def __init__(self, channels: int):
        """
        Create the layers for the transition block

        :param channels:  number of input and output channels, which should be equal
        """
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            # This conv layer is analogous to H-Dense-UNet's 1x2x2 average pool
            nn.Conv3d(channels, channels, kernel_size=(1, 1, 1), stride=(1, 2, 2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the block

        :param x:  image tensor
        :return:   output of the forward pass
        """
        return self.convs(x)
