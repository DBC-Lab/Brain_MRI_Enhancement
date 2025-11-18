import torch
from torch import nn

from models.building_blocks.DenseBlock import DenseBlock
from models.building_blocks.TransitionBlock import TransitionBlock
from models.building_blocks.UpsamplingBlock import UpsamplingBlock

class DenseUNet3d(nn.Module):
    def __init__(self):
        """
        Create the layers for the model
        """
        super().__init__()
        # Initial Layers
        self.conv1 = nn.Conv3d(
            1, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))

        # Dense Layers
        self.transition = TransitionBlock(32)
        self.dense1 = DenseBlock(64, 128, 32, 1)
        self.dense2 = DenseBlock(32, 128, 32, 1)
        self.dense3 = DenseBlock(32, 128, 32, 1)
        self.dense4 = DenseBlock(32, 32, 32, 1)

        # Upsampling Layers
        self.upsample1 = UpsamplingBlock(32 + 32, 504, size=(1, 2, 2))
        self.upsample2 = UpsamplingBlock(504 + 32, 224, size=(1, 2, 2))
        self.upsample3 = UpsamplingBlock(224 + 32, 192, size=(1, 2, 2))
        self.upsample4 = UpsamplingBlock(192 + 32, 96, size=(2, 2, 2))
        self.upsample5 = UpsamplingBlock(96 + 64, 64, size=(2, 2, 2))

        # Final output layer
        # Typo in the paper? Says stride = 0 but that's impossible
        self.conv_classifier = nn.Conv3d(64, 4, kernel_size=1, stride=1)

    def name(self):
        return 'DUNet_Segmentation'
    
    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model

        :param x:  image tensor
        :return:   output of the forward pass
        """
        #x=self.conv1(x)
        residual1 = self.relu(self.bn1(self.conv1(x)))
        #residual2 = self.dense1(self.maxpool1(residual1))
        #residual3 = self.dense2(self.transition(residual2))
        #residual4 = self.dense3(self.transition(residual3))
        #output = self.dense4(self.transition(residual4))

        #output = self.upsample1(residual2, output)
        #output = self.upsample2(output, residual4)
        ##output = self.upsample3(output, residual3)
        #output = self.upsample4(output, residual2)
        #output = self.upsample5(output, residual1)

        output = self.conv_classifier(residual1)

        return output