import torch
from torch import nn

from building_blocks.DenseBlock import DenseBlock
from building_blocks.TransitionBlock import TransitionBlock
from building_blocks.UpsamplingBlock import UpsamplingBlock


class DenseUNet3d_T2w(nn.Module):
    def __init__(self):
        """
        Create the layers for the model
        """
        super().__init__()
        # Initial Layers
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv3d(32, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout3 = nn.Dropout(p=0.1)
        self.bn3 = nn.BatchNorm3d(40)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv3d(40, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout4 = nn.Dropout(p=0.1)
        self.bn4 = nn.BatchNorm3d(48)
        self.relu4 = nn.ReLU()

        self.conv4_1 = nn.Conv3d(48, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout4_1 = nn.Dropout(p=0.1)
        self.bn4_1 = nn.BatchNorm3d(56)
        self.relu4_1 = nn.ReLU()

        self.conv5 = nn.Conv3d(56, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn5 = nn.BatchNorm3d(32)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))

        self.conv6 = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn6 = nn.BatchNorm3d(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv3d(32, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout7 = nn.Dropout(p=0.1)
        self.bn7 = nn.BatchNorm3d(48)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv3d(48, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout8 = nn.Dropout(p=0.1)
        self.bn8 = nn.BatchNorm3d(64)
        self.relu8 = nn.ReLU()

        self.conv9 = nn.Conv3d(64, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout9 = nn.Dropout(p=0.1)
        self.bn9 = nn.BatchNorm3d(80)
        self.relu9 = nn.ReLU()

        self.conv10 = nn.Conv3d(80, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn10 = nn.BatchNorm3d(64)
        self.relu10 = nn.ReLU()
        self.maxpool10 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))

        self.conv11 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn11 = nn.BatchNorm3d(64)
        self.relu11 = nn.ReLU()
        self.maxpool11 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))
        self.deconv11 = nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, padding=1)

        self.conv12 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn12 = nn.BatchNorm3d(128)
        self.relu12 = nn.ReLU()

        self.conv13 = nn.Conv3d(128, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout13 = nn.Dropout(p=0.1)
        self.bn13 = nn.BatchNorm3d(144)
        self.relu13 = nn.ReLU()

        self.conv14 = nn.Conv3d(144, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout14 = nn.Dropout(p=0.1)
        self.bn14 = nn.BatchNorm3d(160)
        self.relu14 = nn.ReLU()

        self.conv15 = nn.Conv3d(160, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout15 = nn.Dropout(p=0.1)
        self.bn15 = nn.BatchNorm3d(176)
        self.relu15 = nn.ReLU()
        self.deconv15 = nn.ConvTranspose3d(176, 64, kernel_size=4, stride=2, padding=1)

        self.conv16 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn16 = nn.BatchNorm3d(128)
        self.relu16 = nn.ReLU()

        self.conv17 = nn.Conv3d(128, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout17 = nn.Dropout(p=0.1)
        self.bn17 = nn.BatchNorm3d(144)
        self.relu17 = nn.ReLU()

        self.conv18 = nn.Conv3d(144, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout18 = nn.Dropout(p=0.1)
        self.bn18 = nn.BatchNorm3d(160)
        self.relu18 = nn.ReLU()

        self.conv19 = nn.Conv3d(160, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout19 = nn.Dropout(p=0.1)
        self.bn19 = nn.BatchNorm3d(176)
        self.relu19 = nn.ReLU()

        self.conv20 = nn.Conv3d(176, 128, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn20 = nn.BatchNorm3d(128)
        self.relu20 = nn.ReLU()

        self.conv21 = nn.Conv3d(128, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout21 = nn.Dropout(p=0.1)
        self.bn21 = nn.BatchNorm3d(144)
        self.relu21 = nn.ReLU()

        self.conv22 = nn.Conv3d(144, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout22 = nn.Dropout(p=0.1)
        self.bn22 = nn.BatchNorm3d(160)
        self.relu22 = nn.ReLU()

        self.conv23 = nn.Conv3d(160, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout23 = nn.Dropout(p=0.1)
        self.bn23 = nn.BatchNorm3d(176)
        self.relu23 = nn.ReLU()
        self.deconv23 = nn.ConvTranspose3d(176, 32, kernel_size=4, stride=2, padding=1)

        self.conv24 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn24 = nn.BatchNorm3d(64)
        self.relu24 = nn.ReLU()

        self.conv25 = nn.Conv3d(64, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout25 = nn.Dropout(p=0.1)
        self.bn25 = nn.BatchNorm3d(72)
        self.relu25 = nn.ReLU()

        self.conv26 = nn.Conv3d(72, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout26 = nn.Dropout(p=0.1)
        self.bn26 = nn.BatchNorm3d(80)
        self.relu26 = nn.ReLU()

        self.conv27 = nn.Conv3d(80, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout27 = nn.Dropout(p=0.1)
        self.bn27 = nn.BatchNorm3d(88)
        self.relu27 = nn.ReLU()

        self.conv28 = nn.Conv3d(88, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout28 = nn.Dropout(p=0.1)
        self.bn28 = nn.BatchNorm3d(32)
        self.relu28 = nn.ReLU()

        self.conv29 = nn.Conv3d(32, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout29 = nn.Dropout(p=0.1)
        self.bn29 = nn.BatchNorm3d(40)
        self.relu29 = nn.ReLU()

        self.conv30 = nn.Conv3d(40, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout30 = nn.Dropout(p=0.1)
        self.bn30 = nn.BatchNorm3d(48)
        self.relu30 = nn.ReLU()

        self.conv31 = nn.Conv3d(48, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.dropout31 = nn.Dropout(p=0.1)
        self.bn31 = nn.BatchNorm3d(56)
        self.relu31 = nn.ReLU()

        self.conv32 = nn.Conv3d(56, 4, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn32 = nn.BatchNorm3d(4)
        self.relu32 = nn.ReLU()

        self.softmax = nn.Softmax()

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

        output = self.softmax(output)

        return output
