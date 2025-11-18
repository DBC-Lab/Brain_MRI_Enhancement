import torch
from torch import nn


class DenseBlock(nn.Module):
    """
    Repeatable Dense block as specified by the paper
    This is composed of a pointwise convolution followed by a depthwise separable convolution
    After each convolution is a BatchNorm followed by a ReLU

    Some notes on architecture based on the paper:
      - The first block uses an input channel of 96, and the remaining input channels are 32
      - The hidden channels is always 128
      - The output channels is always 32
      - The depth is always 3
    """

    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, count: int
    ):
        """
        Create the layers for the dense block

        :param in_channels:      number of input features to the block
        :param hidden_channels:  number of output features from the first convolutional layer
        :param out_channels:     number of output features from this entire block
        :param count:            number of times to repeat
        """
        super().__init__()

        # First iteration takes different number of input channels and does not repeat
        first_block = [
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),
            nn.Conv3d(
                hidden_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        ]

        # Remaining repeats are identical blocks
        repeating_block = [
            nn.Conv3d(out_channels, hidden_channels, kernel_size=(3, 1, 1)),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),
            nn.Conv3d(
                hidden_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        ]

        self.convs = nn.Sequential(
            *first_block,
            *repeating_block * (count - 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the block

        :param x:  image tensor
        :return:   output of the forward pass
        """
        return self.convs(x)
