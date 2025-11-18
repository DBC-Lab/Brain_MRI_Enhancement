import torch
from torch import nn

from BME_X.models.building_blocks.DenseBlock import DenseBlock
from BME_X.models.building_blocks.TransitionBlock import TransitionBlock
from BME_X.models.building_blocks.UpsamplingBlock import UpsamplingBlock


class DenseUNet3d(nn.Module):
    def __init__(self):
        """
        Create the layers for the model
        """
        super().__init__()
        # Segmentation model
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv3d(64, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout3 = nn.Dropout(p=0.1)
        self.bn3 = nn.BatchNorm3d(80)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv3d(80, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout4 = nn.Dropout(p=0.1)
        self.bn4 = nn.BatchNorm3d(96)
        self.relu4 = nn.ReLU()

        self.conv4_1 = nn.Conv3d(96, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout4_1 = nn.Dropout(p=0.1)
        self.bn4_1 = nn.BatchNorm3d(112)
        self.relu4_1 = nn.ReLU()

        self.conv5 = nn.Conv3d(112, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn5 = nn.BatchNorm3d(64)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))

        self.conv6 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn6 = nn.BatchNorm3d(64)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv3d(64, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout7 = nn.Dropout(p=0.1)
        self.bn7 = nn.BatchNorm3d(96)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv3d(96, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout8 = nn.Dropout(p=0.1)
        self.bn8 = nn.BatchNorm3d(128)
        self.relu8 = nn.ReLU()

        self.conv9 = nn.Conv3d(128, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout9 = nn.Dropout(p=0.1)
        self.bn9 = nn.BatchNorm3d(160)
        self.relu9 = nn.ReLU()

        self.conv10 = nn.Conv3d(160, 128, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn10 = nn.BatchNorm3d(128)
        self.relu10 = nn.ReLU()
        self.maxpool10 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))

        self.conv11 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn11 = nn.BatchNorm3d(128)
        self.relu11 = nn.ReLU()
        self.maxpool11 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))
        self.deconv11 = nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, padding=1)

        self.conv12 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn12 = nn.BatchNorm3d(256)
        self.relu12 = nn.ReLU()

        self.conv13 = nn.Conv3d(256, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout13 = nn.Dropout(p=0.1)
        self.bn13 = nn.BatchNorm3d(288)
        self.relu13 = nn.ReLU()

        self.conv14 = nn.Conv3d(288, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout14 = nn.Dropout(p=0.1)
        self.bn14 = nn.BatchNorm3d(320)
        self.relu14 = nn.ReLU()

        self.conv15 = nn.Conv3d(320, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout15 = nn.Dropout(p=0.1)
        self.bn15 = nn.BatchNorm3d(352)
        self.relu15 = nn.ReLU()
        self.deconv15 = nn.ConvTranspose3d(352, 128, kernel_size=4, stride=2, padding=1)

        self.conv16 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn16 = nn.BatchNorm3d(256)
        self.relu16 = nn.ReLU()

        self.conv17 = nn.Conv3d(256, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout17 = nn.Dropout(p=0.1)
        self.bn17 = nn.BatchNorm3d(288)
        self.relu17 = nn.ReLU()

        self.conv18 = nn.Conv3d(288, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout18 = nn.Dropout(p=0.1)
        self.bn18 = nn.BatchNorm3d(320)
        self.relu18 = nn.ReLU()

        self.conv19 = nn.Conv3d(320, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout19 = nn.Dropout(p=0.1)
        self.bn19 = nn.BatchNorm3d(352)
        self.relu19 = nn.ReLU()

        self.conv20 = nn.Conv3d(352, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn20 = nn.BatchNorm3d(256)
        self.relu20 = nn.ReLU()

        self.conv21 = nn.Conv3d(256, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout21 = nn.Dropout(p=0.1)
        self.bn21 = nn.BatchNorm3d(288)
        self.relu21 = nn.ReLU()

        self.conv22 = nn.Conv3d(288, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout22 = nn.Dropout(p=0.1)
        self.bn22 = nn.BatchNorm3d(320)
        self.relu22 = nn.ReLU()

        self.conv23 = nn.Conv3d(320, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout23 = nn.Dropout(p=0.1)
        self.bn23 = nn.BatchNorm3d(352)
        self.relu23 = nn.ReLU()
        self.deconv23 = nn.ConvTranspose3d(352, 64, kernel_size=4, stride=2, padding=1)

        self.conv24 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn24 = nn.BatchNorm3d(128)
        self.relu24 = nn.ReLU()

        self.conv25 = nn.Conv3d(128, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout25 = nn.Dropout(p=0.1)
        self.bn25 = nn.BatchNorm3d(144)
        self.relu25 = nn.ReLU()

        self.conv26 = nn.Conv3d(144, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout26 = nn.Dropout(p=0.1)
        self.bn26 = nn.BatchNorm3d(160)
        self.relu26 = nn.ReLU()

        self.conv27 = nn.Conv3d(160, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout27 = nn.Dropout(p=0.1)
        self.bn27 = nn.BatchNorm3d(176)
        self.relu27 = nn.ReLU()

        self.conv28 = nn.Conv3d(176, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout28 = nn.Dropout(p=0.1)
        self.bn28 = nn.BatchNorm3d(64)
        self.relu28 = nn.ReLU()

        self.conv29 = nn.Conv3d(64, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout29 = nn.Dropout(p=0.1)
        self.bn29 = nn.BatchNorm3d(80)
        self.relu29 = nn.ReLU()

        self.conv30 = nn.Conv3d(80, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout30 = nn.Dropout(p=0.1)
        self.bn30 = nn.BatchNorm3d(96)
        self.relu30 = nn.ReLU()

        self.conv31 = nn.Conv3d(96, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout31 = nn.Dropout(p=0.1)
        self.bn31 = nn.BatchNorm3d(112)
        self.relu31 = nn.ReLU()

        self.conv32 = nn.Conv3d(112, 4, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn32 = nn.BatchNorm3d(4)
        self.relu32 = nn.ReLU()

        # Connection
        self.conv1_con = nn.Conv3d(4, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn1_con = nn.BatchNorm3d(32)
        self.relu1_con = nn.ReLU()

        self.conv2_con = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn2_con = nn.BatchNorm3d(32)
        self.relu2_con = nn.ReLU()

        # Reconstruction model
        self.conv1_recon = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn1_recon = nn.BatchNorm3d(64)
        self.relu1_recon = nn.ReLU()

        self.conv2_recon = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn2_recon = nn.BatchNorm3d(64)
        self.relu2_recon = nn.ReLU()

        self.conv3_recon = nn.Conv3d(64, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout3_recon = nn.Dropout(p=0.1)
        self.bn3_recon = nn.BatchNorm3d(80)
        self.relu3_recon = nn.ReLU()

        self.conv4_recon = nn.Conv3d(80, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout4_recon = nn.Dropout(p=0.1)
        self.bn4_recon = nn.BatchNorm3d(96)
        self.relu4_recon = nn.ReLU()

        self.conv4_1_recon = nn.Conv3d(96, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout4_1_recon = nn.Dropout(p=0.1)
        self.bn4_1_recon = nn.BatchNorm3d(112)
        self.relu4_1_recon = nn.ReLU()

        self.conv5_recon = nn.Conv3d(112, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn5_recon = nn.BatchNorm3d(64)
        self.relu5_recon = nn.ReLU()
        self.maxpool5_recon = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))

        self.conv6_recon = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn6_recon = nn.BatchNorm3d(64)
        self.relu6_recon = nn.ReLU()

        self.conv7_recon = nn.Conv3d(64, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout7_recon = nn.Dropout(p=0.1)
        self.bn7_recon = nn.BatchNorm3d(96)
        self.relu7_recon = nn.ReLU()

        self.conv8_recon = nn.Conv3d(96, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout8_recon = nn.Dropout(p=0.1)
        self.bn8_recon = nn.BatchNorm3d(128)
        self.relu8_recon = nn.ReLU()

        self.conv9_recon = nn.Conv3d(128, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout9_recon = nn.Dropout(p=0.1)
        self.bn9_recon = nn.BatchNorm3d(160)
        self.relu9_recon = nn.ReLU()

        self.conv10_recon = nn.Conv3d(160, 128, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn10_recon = nn.BatchNorm3d(128)
        self.relu10_recon = nn.ReLU()
        self.maxpool10_recon = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))

        self.conv11_recon = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn11_recon = nn.BatchNorm3d(128)
        self.relu11_recon = nn.ReLU()
        self.maxpool11_recon = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))
        self.deconv11_recon = nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, padding=1)

        self.conv12_recon = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn12_recon = nn.BatchNorm3d(256)
        self.relu12_recon = nn.ReLU()

        self.conv13_recon = nn.Conv3d(256, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout13_recon = nn.Dropout(p=0.1)
        self.bn13_recon = nn.BatchNorm3d(288)
        self.relu13_recon = nn.ReLU()

        self.conv14_recon = nn.Conv3d(288, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout14_recon = nn.Dropout(p=0.1)
        self.bn14_recon = nn.BatchNorm3d(320)
        self.relu14_recon = nn.ReLU()

        self.conv15_recon = nn.Conv3d(320, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout15_recon = nn.Dropout(p=0.1)
        self.bn15_recon = nn.BatchNorm3d(352)
        self.relu15_recon = nn.ReLU()
        self.deconv15_recon = nn.ConvTranspose3d(352, 128, kernel_size=4, stride=2, padding=1)

        self.conv16_recon = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn16_recon = nn.BatchNorm3d(256)
        self.relu16_recon = nn.ReLU()

        self.conv17_recon = nn.Conv3d(256, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout17_recon = nn.Dropout(p=0.1)
        self.bn17_recon = nn.BatchNorm3d(288)
        self.relu17_recon = nn.ReLU()

        self.conv18_recon = nn.Conv3d(288, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout18_recon = nn.Dropout(p=0.1)
        self.bn18_recon = nn.BatchNorm3d(320)
        self.relu18_recon = nn.ReLU()

        self.conv19_recon = nn.Conv3d(320, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout19_recon = nn.Dropout(p=0.1)
        self.bn19_recon = nn.BatchNorm3d(352)
        self.relu19_recon = nn.ReLU()

        self.conv20_recon = nn.Conv3d(352, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn20_recon = nn.BatchNorm3d(256)
        self.relu20_recon = nn.ReLU()

        self.conv21_recon = nn.Conv3d(256, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout21_recon = nn.Dropout(p=0.1)
        self.bn21_recon = nn.BatchNorm3d(288)
        self.relu21_recon = nn.ReLU()

        self.conv22_recon = nn.Conv3d(288, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout22_recon = nn.Dropout(p=0.1)
        self.bn22_recon = nn.BatchNorm3d(320)
        self.relu22_recon = nn.ReLU()

        self.conv23_recon = nn.Conv3d(320, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout23_recon = nn.Dropout(p=0.1)
        self.bn23_recon = nn.BatchNorm3d(352)
        self.relu23_recon = nn.ReLU()
        self.deconv23_recon = nn.ConvTranspose3d(352, 64, kernel_size=4, stride=2, padding=1)

        self.conv24_recon = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn24_recon = nn.BatchNorm3d(128)
        self.relu24_recon = nn.ReLU()

        self.conv25_recon = nn.Conv3d(128, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout25_recon = nn.Dropout(p=0.1)
        self.bn25_recon = nn.BatchNorm3d(144)
        self.relu25_recon = nn.ReLU()

        self.conv26_recon = nn.Conv3d(144, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout26_recon = nn.Dropout(p=0.1)
        self.bn26_recon = nn.BatchNorm3d(160)
        self.relu26_recon = nn.ReLU()

        self.conv27_recon = nn.Conv3d(160, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout27_recon = nn.Dropout(p=0.1)
        self.bn27_recon = nn.BatchNorm3d(176)
        self.relu27_recon = nn.ReLU()

        self.conv28_recon = nn.Conv3d(176, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout28_recon = nn.Dropout(p=0.1)
        self.bn28_recon = nn.BatchNorm3d(64)
        self.relu28_recon = nn.ReLU()

        self.conv29_recon = nn.Conv3d(64, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout29_recon = nn.Dropout(p=0.1)
        self.bn29_recon = nn.BatchNorm3d(80)
        self.relu29_recon = nn.ReLU()

        self.conv30_recon = nn.Conv3d(80, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout30_recon = nn.Dropout(p=0.1)
        self.bn30_recon = nn.BatchNorm3d(96)
        self.relu30_recon = nn.ReLU()

        self.conv31_recon = nn.Conv3d(96, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout31_recon = nn.Dropout(p=0.1)
        self.bn31_recon = nn.BatchNorm3d(112)
        self.relu31_recon = nn.ReLU()

        self.conv32_recon = nn.Conv3d(112, 1, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn32_recon = nn.BatchNorm3d(1)
        self.relu32_recon = nn.ReLU()
        
        self.softmax = nn.Softmax(dim=1)
        
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
        conv1a = self.conv1(x)  # x: 9 1 32 32 32
        conv1a_bn = self.bn1(conv1a)
        conv1a_bn = self.relu1(conv1a_bn)

        convolution1 = self.conv2(conv1a_bn)  # 3 64 32 32 32
        batchnorm1 = self.bn2(convolution1)
        batchnorm1 = self.relu2(batchnorm1)

        convolution2 = self.conv3(batchnorm1)
        dropout1 = self.dropout3(convolution2)
        concat1 = torch.cat([convolution1, dropout1], dim=1)
        batchnorm2 = self.bn3(concat1)  # 3 80 32 32 32
        batchnorm2 = self.relu3(batchnorm2)

        convolution3 = self.conv4(batchnorm2)
        dropout2 = self.dropout4(convolution3)
        concat2 = torch.cat([concat1, dropout2], dim=1)
        batchnorm3 = self.bn4(concat2)  # 3 96 32 32 32
        batchnorm3 = self.relu4(batchnorm3)

        convolution4 = self.conv4_1(batchnorm3)
        dropout3 = self.dropout4_1(convolution4)
        concat3 = torch.cat([concat2, dropout3], dim=1)
        conv1b_bn = self.bn4_1(concat3)  # 3 112 32 32 32
        conv1b_bn = self.relu4_1(conv1b_bn)

        conv1c = self.conv5(conv1b_bn)  # 3 64 32 32 32
        conv1c_bn = self.bn5(conv1c)
        conv1c_bn = self.relu5(conv1c_bn)
        pool1 = self.maxpool5(conv1c_bn)

        conv2a_Convolution1 = self.conv6(pool1)
        conv2a_BatchNorm1 = self.bn6(conv2a_Convolution1)
        conv2a_BatchNorm1 = self.relu6(conv2a_BatchNorm1)

        conv2a_Convolution2 = self.conv7(conv2a_BatchNorm1)  # 3 32 16 16 16
        conv2a_Dropout1 = self.dropout7(conv2a_Convolution2)
        conv2a_Concat1 = torch.cat([conv2a_Dropout1, conv2a_Convolution1], dim=1)
        conv2a_BatchNorm2 = self.bn7(conv2a_Concat1)
        conv2a_BatchNorm2 = self.relu7(conv2a_BatchNorm2)

        conv2a_Convolution3 = self.conv8(conv2a_BatchNorm2)
        conv2a_Dropout2 = self.dropout8(conv2a_Convolution3)
        conv2a_Concat2 = torch.cat([conv2a_Concat1, conv2a_Dropout2], dim=1)
        conv2a_BatchNorm3 = self.bn8(conv2a_Concat2)
        conv2a_BatchNorm3 = self.relu8(conv2a_BatchNorm3)

        conv2a_Convolution4 = self.conv9(conv2a_BatchNorm3)  # 3 32 16 16 16
        conv2a_Dropout3 = self.dropout9(conv2a_Convolution4)
        conv2a = torch.cat([conv2a_Concat2, conv2a_Dropout3], dim=1)
        conv2a_bn = self.bn9(conv2a)
        conv2a_bn = self.relu9(conv2a_bn)

        conv2b = self.conv10(conv2a_bn)
        conv2b_bn = self.bn10(conv2b)
        conv2b_bn = self.relu10(conv2b_bn)
        pool2 = self.maxpool10(conv2b_bn)

        conv3a = self.conv11(pool2)
        conv3a_bn = self.bn11(conv3a)
        conv3a_bn = self.relu11(conv3a_bn)
        pool3 = self.maxpool11(conv3a_bn)
        deconv4 = self.deconv11(pool3)  # 3 128 8 8 8
        concat8 = torch.cat([conv3a, deconv4], dim=1)

        conv4_Convolution1 = self.conv12(concat8)
        conv4_BatchNorm1 = self.bn12(conv4_Convolution1)
        conv4_BatchNorm1 = self.relu12(conv4_BatchNorm1)

        conv4_Convolution2 = self.conv13(conv4_BatchNorm1)
        conv4_Dropout1 = self.dropout13(conv4_Convolution2)
        conv4_Concat1 = torch.cat([conv4_Convolution1, conv4_Dropout1], dim=1)
        conv4_BatchNorm2 = self.bn13(conv4_Concat1)
        conv4_BatchNorm2 = self.relu13(conv4_BatchNorm2)

        conv4_Convolution3 = self.conv14(conv4_BatchNorm2)
        conv4_Dropout2 = self.dropout14(conv4_Convolution3)
        conv4_Concat2 = torch.cat([conv4_Concat1, conv4_Dropout2], dim=1)
        conv4_BatchNorm3 = self.bn14(conv4_Concat2)
        conv4_BatchNorm3 = self.relu14(conv4_BatchNorm3)

        conv4_Convolution4 = self.conv15(conv4_BatchNorm3)  # 3 32 8 8 8
        conv4_Dropout3 = self.dropout15(conv4_Convolution4)
        conv4 = torch.cat([conv4_Concat2, conv4_Dropout3], dim=1)
        conv4_bn = self.bn15(conv4)
        conv4_bn = self.relu15(conv4_bn)
        deconv5 = self.deconv15(conv4_bn)  # 3 128 16 16 16
        concat16 = torch.cat([conv2b, deconv5], dim=1)

        conv5_Convolution1 = self.conv16(concat16)
        conv5_BatchNorm1 = self.bn16(conv5_Convolution1)
        conv5_BatchNorm1 = self.relu16(conv5_BatchNorm1)

        conv5_Convolution2 = self.conv17(conv5_BatchNorm1)
        conv5_Dropout1 = self.dropout17(conv5_Convolution2)
        conv5_Concat1 = torch.cat([conv5_Convolution1, conv5_Dropout1], dim=1)
        conv5_BatchNorm2 = self.bn17(conv5_Concat1)
        conv5_BatchNorm2 = self.relu17(conv5_BatchNorm2)

        conv5_Convolution3 = self.conv18(conv5_BatchNorm2)
        conv5_Dropout2 = self.dropout18(conv5_Convolution3)
        conv5_Concat2 = torch.cat([conv5_Concat1, conv5_Dropout2], dim=1)
        conv5_Concat2 = self.bn18(conv5_Concat2)
        conv5_Concat2 = self.relu18(conv5_Concat2)

        conv5_Convolution4 = self.conv19(conv5_Concat2)  # 3 32 16 16 16
        conv5_Dropout3 = self.dropout19(conv5_Convolution4)
        conv5 = torch.cat([conv5_Concat2, conv5_Dropout3], dim=1)
        conv5_bn = self.bn19(conv5)
        conv5_bn = self.relu19(conv5_bn)

        conv5_2_Convolution1 = self.conv20(conv5_bn)
        conv5_2_BatchNorm1 = self.bn20(conv5_2_Convolution1)
        conv5_2_BatchNorm1 = self.relu20(conv5_2_BatchNorm1)

        conv5_2_Convolution2 = self.conv21(conv5_2_BatchNorm1)
        conv5_2_Dropout1 = self.dropout21(conv5_2_Convolution2)
        conv5_2_Concat1 = torch.cat([conv5_2_Convolution1, conv5_2_Dropout1], dim=1)
        conv5_2_BatchNorm2 = self.bn21(conv5_2_Concat1)
        conv5_2_BatchNorm2 = self.relu21(conv5_2_BatchNorm2)

        conv5_2_Convolution3 = self.conv22(conv5_2_BatchNorm2)
        conv5_2_Dropout2 = self.dropout22(conv5_2_Convolution3)  # 3 32 16 16 16
        conv5_2_Concat2 = torch.cat([conv5_2_Concat1, conv5_2_Dropout2], dim=1)
        conv5_2_BatchNorm3 = self.bn22(conv5_2_Concat2)
        conv5_2_BatchNorm3 = self.relu22(conv5_2_BatchNorm3)

        conv5_2_Convolution4 = self.conv23(conv5_2_BatchNorm3)
        conv5_2_Dropout3 = self.dropout23(conv5_2_Convolution4)
        conv5_2 = torch.cat([conv5_2_Concat2, conv5_2_Dropout3], dim=1)
        conv5_2_bn = self.bn23(conv5_2)
        conv5_2_bn = self.relu23(conv5_2_bn)
        deconv6 = self.deconv23(conv5_2_bn)
        concat32 = torch.cat([conv1c, deconv6], dim=1)

        conv6_Convolution1 = self.conv24(concat32)
        conv6_BatchNorm1 = self.bn24(conv6_Convolution1)
        conv6_BatchNorm1 = self.relu24(conv6_BatchNorm1)

        conv6_Convolution2 = self.conv25(conv6_BatchNorm1)  # 3 16 32 32 32
        conv6_Dropout1 = self.dropout25(conv6_Convolution2)
        conv6_Concat1 = torch.cat([conv6_Convolution1, conv6_Dropout1], dim=1)
        conv6_BatchNorm2 = self.bn25(conv6_Concat1)
        conv6_BatchNorm2 = self.relu25(conv6_BatchNorm2)

        conv6_Convolution3 = self.conv26(conv6_BatchNorm2)
        conv6_Dropout2 = self.dropout26(conv6_Convolution3)
        conv6_Concat2 = torch.cat([conv6_Concat1, conv6_Dropout2], dim=1)
        conv6_BatchNorm3 = self.bn26(conv6_Concat2)
        conv6_BatchNorm3 = self.relu26(conv6_BatchNorm3)

        conv6_Convolution4 = self.conv27(conv6_BatchNorm3)
        conv6_Dropout3 = self.dropout27(conv6_Convolution4)
        conv6 = torch.cat([conv6_Concat2, conv6_Dropout3], dim=1)
        conv6_bn = self.bn27(conv6)
        conv6_bn = self.relu27(conv6_bn)

        conv6_2_Convolution1 = self.conv28(conv6_bn)  # 3 16 32 32 32
        conv6_2_BatchNorm1 = self.bn28(conv6_2_Convolution1)
        conv6_2_BatchNorm1 = self.relu28(conv6_2_BatchNorm1)

        conv6_2_Convolution2 = self.conv29(conv6_2_BatchNorm1)
        conv6_2_Dropout1 = self.dropout29(conv6_2_Convolution2)
        conv6_2_Concat1 = torch.cat([conv6_2_Convolution1, conv6_2_Dropout1], dim=1) # 3 80 32 32
        conv6_2_BatchNorm2 = self.bn29(conv6_2_Concat1)
        conv6_2_BatchNorm2 = self.relu29(conv6_2_BatchNorm2)

        conv6_2_Convolution3 = self.conv30(conv6_2_BatchNorm2)  # 3 16 32 32 32
        conv6_2_Dropout2 = self.dropout30(conv6_2_Convolution3)
        conv6_2_Concat2 = torch.cat([conv6_2_Concat1, conv6_2_Dropout2], dim=1)
        conv6_2_BatchNorm3 = self.bn30(conv6_2_Concat2)
        conv6_2_BatchNorm3 = self.relu30(conv6_2_BatchNorm3)

        conv6_2_Convolution4 = self.conv31(conv6_2_BatchNorm3)
        conv6_2_Dropout3 = self.dropout31(conv6_2_Convolution4)
        conv6_2 = torch.cat([conv6_2_Concat2, conv6_2_Dropout3], dim=1)
        conv6_2_bn = self.bn31(conv6_2)
        conv6_2_bn = self.relu31(conv6_2_bn)

        conv6_3_Convolution1 = self.conv32(conv6_2_bn)
        conv6_3_BatchNorm1 = self.bn32(conv6_3_Convolution1)
        output = self.relu32(conv6_3_BatchNorm1)

        # Connection
        output_soft = self.softmax(output)
        output_con = self.conv1_con(output_soft)
        output_bn_con = self.bn1_con(output_con)
        output_bn_con = self.relu1_con(output_bn_con)

        volMR_con = self.conv2_con(x)
        volMR_bn_con = self.bn2_con(volMR_con)
        volMR_bn_con = self.relu2_con(volMR_bn_con)
        concat1_recon = torch.cat([output_bn_con, volMR_bn_con], dim=1)

        # Reconstruction model
        conv1a_recon = self.conv1_recon(concat1_recon)  # x: 9 1 32 32 32
        conv1a_bn_recon = self.bn1_recon(conv1a_recon)
        conv1a_bn_recon = self.relu1_recon(conv1a_bn_recon)

        convolution1_recon = self.conv2_recon(conv1a_bn_recon)  # 3 64 32 32 32
        batchnorm1_recon = self.bn2_recon(convolution1_recon)
        batchnorm1_recon = self.relu2_recon(batchnorm1_recon)

        convolution2_recon = self.conv3_recon(batchnorm1_recon)
        dropout1_recon = self.dropout3_recon(convolution2_recon)
        concat1_recon = torch.cat([convolution1_recon, dropout1_recon], dim=1)
        batchnorm2_recon = self.bn3_recon(concat1_recon)  # 3 80 32 32 32
        batchnorm2_recon = self.relu3_recon(batchnorm2_recon)

        convolution3_recon = self.conv4_recon(batchnorm2_recon)
        dropout2_recon = self.dropout4_recon(convolution3_recon)
        concat2_recon = torch.cat([concat1_recon, dropout2_recon], dim=1)
        batchnorm3_recon = self.bn4_recon(concat2_recon)  # 3 96 32 32 32
        batchnorm3_recon = self.relu4_recon(batchnorm3_recon)

        convolution4_recon = self.conv4_1_recon(batchnorm3_recon)
        dropout3_recon = self.dropout4_1_recon(convolution4_recon)
        concat3_recon = torch.cat([concat2_recon, dropout3_recon], dim=1)
        conv1b_bn_recon = self.bn4_1_recon(concat3_recon)  # 3 112 32 32 32
        conv1b_bn_recon = self.relu4_1_recon(conv1b_bn_recon)

        conv1c_recon = self.conv5_recon(conv1b_bn_recon)  # 3 64 32 32 32
        conv1c_bn_recon = self.bn5_recon(conv1c_recon)
        conv1c_bn_recon = self.relu5_recon(conv1c_bn_recon)
        pool1_recon = self.maxpool5_recon(conv1c_bn_recon)

        conv2a_Convolution1_recon = self.conv6_recon(pool1_recon)
        conv2a_BatchNorm1_recon = self.bn6_recon(conv2a_Convolution1_recon)
        conv2a_BatchNorm1_recon = self.relu6_recon(conv2a_BatchNorm1_recon)

        conv2a_Convolution2_recon = self.conv7_recon(conv2a_BatchNorm1_recon)  # 3 32 16 16 16
        conv2a_Dropout1_recon = self.dropout7_recon(conv2a_Convolution2_recon)
        conv2a_Concat1_recon = torch.cat([conv2a_Dropout1_recon, conv2a_Convolution1_recon], dim=1)
        conv2a_BatchNorm2_recon = self.bn7_recon(conv2a_Concat1_recon)
        conv2a_BatchNorm2_recon = self.relu7_recon(conv2a_BatchNorm2_recon)

        conv2a_Convolution3_recon = self.conv8_recon(conv2a_BatchNorm2_recon)
        conv2a_Dropout2_recon = self.dropout8_recon(conv2a_Convolution3_recon)
        conv2a_Concat2_recon = torch.cat([conv2a_Concat1_recon, conv2a_Dropout2_recon], dim=1)
        conv2a_BatchNorm3_recon = self.bn8_recon(conv2a_Concat2_recon)
        conv2a_BatchNorm3_recon = self.relu8_recon(conv2a_BatchNorm3_recon)

        conv2a_Convolution4_recon = self.conv9_recon(conv2a_BatchNorm3_recon)  # 3 32 16 16 16
        conv2a_Dropout3_recon = self.dropout9_recon(conv2a_Convolution4_recon)
        conv2a_recon = torch.cat([conv2a_Concat2_recon, conv2a_Dropout3_recon], dim=1)
        conv2a_bn_recon = self.bn9_recon(conv2a_recon)
        conv2a_bn_recon = self.relu9_recon(conv2a_bn_recon)

        conv2b_recon = self.conv10_recon(conv2a_bn_recon)
        conv2b_bn_recon = self.bn10_recon(conv2b_recon)
        conv2b_bn_recon = self.relu10_recon(conv2b_bn_recon)
        pool2_recon = self.maxpool10_recon(conv2b_bn_recon)

        conv3a_recon = self.conv11_recon(pool2_recon)
        conv3a_bn_recon = self.bn11_recon(conv3a_recon)
        conv3a_bn_recon = self.relu11_recon(conv3a_bn_recon)
        pool3_recon = self.maxpool11_recon(conv3a_bn_recon)
        deconv4_recon = self.deconv11_recon(pool3_recon)  # 3 128 8 8 8
        concat8_recon = torch.cat([conv3a_recon, deconv4_recon], dim=1)

        conv4_Convolution1_recon = self.conv12_recon(concat8_recon)
        conv4_BatchNorm1_recon = self.bn12_recon(conv4_Convolution1_recon)
        conv4_BatchNorm1_recon = self.relu12_recon(conv4_BatchNorm1_recon)

        conv4_Convolution2_recon = self.conv13_recon(conv4_BatchNorm1_recon)
        conv4_Dropout1_recon = self.dropout13_recon(conv4_Convolution2_recon)
        conv4_Concat1_recon = torch.cat([conv4_Convolution1_recon, conv4_Dropout1_recon], dim=1)
        conv4_BatchNorm2_recon = self.bn13_recon(conv4_Concat1_recon)
        conv4_BatchNorm2_recon = self.relu13_recon(conv4_BatchNorm2_recon)

        conv4_Convolution3_recon = self.conv14_recon(conv4_BatchNorm2_recon)
        conv4_Dropout2_recon = self.dropout14_recon(conv4_Convolution3_recon)
        conv4_Concat2_recon = torch.cat([conv4_Concat1_recon, conv4_Dropout2_recon], dim=1)
        conv4_BatchNorm3_recon = self.bn14_recon(conv4_Concat2_recon)
        conv4_BatchNorm3_recon = self.relu14_recon(conv4_BatchNorm3_recon)

        conv4_Convolution4_recon = self.conv15_recon(conv4_BatchNorm3_recon)  # 3 32 8 8 8
        conv4_Dropout3_recon = self.dropout15_recon(conv4_Convolution4_recon)
        conv4_recon = torch.cat([conv4_Concat2_recon, conv4_Dropout3_recon], dim=1)
        conv4_bn_recon = self.bn15_recon(conv4_recon)
        conv4_bn_recon = self.relu15_recon(conv4_bn_recon)
        deconv5_recon = self.deconv15_recon(conv4_bn_recon)  # 3 128 16 16 16
        concat16_recon = torch.cat([conv2b_recon, deconv5_recon], dim=1)

        conv5_Convolution1_recon = self.conv16_recon(concat16_recon)
        conv5_BatchNorm1_recon = self.bn16_recon(conv5_Convolution1_recon)
        conv5_BatchNorm1_recon = self.relu16_recon(conv5_BatchNorm1_recon)

        conv5_Convolution2_recon = self.conv17_recon(conv5_BatchNorm1_recon)
        conv5_Dropout1_recon = self.dropout17_recon(conv5_Convolution2_recon)
        conv5_Concat1_recon = torch.cat([conv5_Convolution1_recon, conv5_Dropout1_recon], dim=1)
        conv5_BatchNorm2_recon = self.bn17_recon(conv5_Concat1_recon)
        conv5_BatchNorm2_recon = self.relu17_recon(conv5_BatchNorm2_recon)

        conv5_Convolution3_recon = self.conv18_recon(conv5_BatchNorm2_recon)
        conv5_Dropout2_recon = self.dropout18_recon(conv5_Convolution3_recon)
        conv5_Concat2_recon = torch.cat([conv5_Concat1_recon, conv5_Dropout2_recon], dim=1)
        conv5_Concat2_recon = self.bn18_recon(conv5_Concat2_recon)
        conv5_Concat2_recon = self.relu18_recon(conv5_Concat2_recon)

        conv5_Convolution4_recon = self.conv19_recon(conv5_Concat2_recon)  # 3 32 16 16 16
        conv5_Dropout3_recon = self.dropout19_recon(conv5_Convolution4_recon)
        conv5_recon = torch.cat([conv5_Concat2_recon, conv5_Dropout3_recon], dim=1)
        conv5_bn_recon = self.bn19_recon(conv5_recon)
        conv5_bn_recon = self.relu19_recon(conv5_bn_recon)

        conv5_2_Convolution1_recon = self.conv20_recon(conv5_bn_recon)
        conv5_2_BatchNorm1_recon = self.bn20_recon(conv5_2_Convolution1_recon)
        conv5_2_BatchNorm1_recon = self.relu20_recon(conv5_2_BatchNorm1_recon)

        conv5_2_Convolution2_recon = self.conv21_recon(conv5_2_BatchNorm1_recon)
        conv5_2_Dropout1_recon = self.dropout21_recon(conv5_2_Convolution2_recon)
        conv5_2_Concat1_recon = torch.cat([conv5_2_Convolution1_recon, conv5_2_Dropout1_recon], dim=1)
        conv5_2_BatchNorm2_recon = self.bn21_recon(conv5_2_Concat1_recon)
        conv5_2_BatchNorm2_recon = self.relu21_recon(conv5_2_BatchNorm2_recon)

        conv5_2_Convolution3_recon = self.conv22_recon(conv5_2_BatchNorm2_recon)
        conv5_2_Dropout2_recon = self.dropout22_recon(conv5_2_Convolution3_recon)  # 3 32 16 16 16
        conv5_2_Concat2_recon = torch.cat([conv5_2_Concat1_recon, conv5_2_Dropout2_recon], dim=1)
        conv5_2_BatchNorm3_recon = self.bn22_recon(conv5_2_Concat2_recon)
        conv5_2_BatchNorm3_recon = self.relu22_recon(conv5_2_BatchNorm3_recon)

        conv5_2_Convolution4_recon = self.conv23_recon(conv5_2_BatchNorm3_recon)
        conv5_2_Dropout3_recon = self.dropout23_recon(conv5_2_Convolution4_recon)
        conv5_2_recon = torch.cat([conv5_2_Concat2_recon, conv5_2_Dropout3_recon], dim=1)
        conv5_2_bn_recon = self.bn23_recon(conv5_2_recon)
        conv5_2_bn_recon = self.relu23_recon(conv5_2_bn_recon)
        deconv6_recon = self.deconv23_recon(conv5_2_bn_recon)
        concat32_recon = torch.cat([conv1c_recon, deconv6_recon], dim=1)

        conv6_Convolution1_recon = self.conv24_recon(concat32_recon)
        conv6_BatchNorm1_recon = self.bn24_recon(conv6_Convolution1_recon)
        conv6_BatchNorm1_recon = self.relu24_recon(conv6_BatchNorm1_recon)

        conv6_Convolution2_recon = self.conv25_recon(conv6_BatchNorm1_recon)  # 3 16 32 32 32
        conv6_Dropout1_recon = self.dropout25_recon(conv6_Convolution2_recon)
        conv6_Concat1_recon = torch.cat([conv6_Convolution1_recon, conv6_Dropout1_recon], dim=1)
        conv6_BatchNorm2_recon = self.bn25_recon(conv6_Concat1_recon)
        conv6_BatchNorm2_recon = self.relu25_recon(conv6_BatchNorm2_recon)

        conv6_Convolution3_recon = self.conv26_recon(conv6_BatchNorm2_recon)
        conv6_Dropout2_recon = self.dropout26_recon(conv6_Convolution3_recon)
        conv6_Concat2_recon = torch.cat([conv6_Concat1_recon, conv6_Dropout2_recon], dim=1)
        conv6_BatchNorm3_recon = self.bn26_recon(conv6_Concat2_recon)
        conv6_BatchNorm3_recon = self.relu26_recon(conv6_BatchNorm3_recon)

        conv6_Convolution4_recon = self.conv27_recon(conv6_BatchNorm3_recon)
        conv6_Dropout3_recon = self.dropout27_recon(conv6_Convolution4_recon)
        conv6_recon = torch.cat([conv6_Concat2_recon, conv6_Dropout3_recon], dim=1)
        conv6_bn_recon = self.bn27_recon(conv6_recon)
        conv6_bn_recon = self.relu27_recon(conv6_bn_recon)

        conv6_2_Convolution1_recon = self.conv28_recon(conv6_bn_recon)  # 3 16 32 32 32
        conv6_2_BatchNorm1_recon = self.bn28_recon(conv6_2_Convolution1_recon)
        conv6_2_BatchNorm1_recon = self.relu28_recon(conv6_2_BatchNorm1_recon)

        conv6_2_Convolution2_recon = self.conv29_recon(conv6_2_BatchNorm1_recon)
        conv6_2_Dropout1_recon = self.dropout29_recon(conv6_2_Convolution2_recon)
        conv6_2_Concat1_recon = torch.cat([conv6_2_Convolution1_recon, conv6_2_Dropout1_recon], dim=1)  # 3 80 32 32
        conv6_2_BatchNorm2_recon = self.bn29_recon(conv6_2_Concat1_recon)
        conv6_2_BatchNorm2_recon = self.relu29_recon(conv6_2_BatchNorm2_recon)

        conv6_2_Convolution3_recon = self.conv30_recon(conv6_2_BatchNorm2_recon)  # 3 16 32 32 32
        conv6_2_Dropout2_recon = self.dropout30_recon(conv6_2_Convolution3_recon)
        conv6_2_Concat2_recon = torch.cat([conv6_2_Concat1_recon, conv6_2_Dropout2_recon], dim=1)
        conv6_2_BatchNorm3_recon = self.bn30_recon(conv6_2_Concat2_recon)
        conv6_2_BatchNorm3_recon = self.relu30_recon(conv6_2_BatchNorm3_recon)

        conv6_2_Convolution4_recon = self.conv31_recon(conv6_2_BatchNorm3_recon)
        conv6_2_Dropout3_recon = self.dropout31_recon(conv6_2_Convolution4_recon)
        conv6_2_recon = torch.cat([conv6_2_Concat2_recon, conv6_2_Dropout3_recon], dim=1)
        conv6_2_bn_recon = self.bn31_recon(conv6_2_recon)
        conv6_2_bn_recon = self.relu31_recon(conv6_2_bn_recon)

        conv6_3_Convolution1_recon = self.conv32_recon(conv6_2_bn_recon)
        conv6_3_BatchNorm1_recon = self.bn32_recon(conv6_3_Convolution1_recon)
        output_recon = self.relu32_recon(conv6_3_BatchNorm1_recon)
        return output, output_recon
