from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import SimpleITK as sitk
import layers
from torch.distributions.normal import Normal

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets import ViT
from networks.blocks import UnetBasicBlock, UnetUpBlock, UnetResBlock, get_conv_layer
class proj_feat(nn.Module):
    def __init__(self, hidden_size, feat_size, out_channels, upsample_kernel_size, upsample_stride):
        super(proj_feat, self).__init__()
        self.hidden_size = hidden_size
        self.feat_size = feat_size
        self.in_channels = hidden_size
        self.out_channels = out_channels
        self.upsample_kernel_size = upsample_kernel_size
        self.upsample_stride = upsample_stride
        self.transp_conv_init = get_conv_layer(
            3,
            hidden_size,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=False,
            is_transposed=True,
        )

    def forward(self, x):
        x = x.view(x.size(0), self.feat_size[0], self.feat_size[1], self.feat_size[2], self.hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.transp_conv_init(x)

        return x


class NET(nn.Module):
    def __init__(
        self,
        device: int,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = False,
        dropout_rate: float = 0.0,
        
    ) -> None:

        super().__init__()

        self.num_layers = 2
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False

        self.pool = nn.MaxPool3d(2, 2)

        self.encoder1 = UnetBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.vit1 = ViT(
            in_channels=1,
            img_size=(128,128,128),
            patch_size=(16,16,16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=2,
            num_heads=12,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )

        self.proj_feat1 = proj_feat(768, (8,8,8), 16, (16,16,16), (16,16,16))


        self.encoder2 = UnetBasicBlock(
            spatial_dims=3,
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.vit2 = ViT(
            in_channels=32,
            img_size=(64,64,64),
            patch_size=(8,8,8),
            hidden_size=128,
            mlp_dim=384,
            num_layers=2,
            num_heads=8,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.proj_feat2 = proj_feat(128, (8,8,8), 32, (8,8,8), (8,8,8))

        self.encoder3 = UnetBasicBlock(
            spatial_dims=3,
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.vit3 = ViT(
            in_channels=64,
            img_size=(32,32,32),
            patch_size=(4,4,4),
            hidden_size=16,
            mlp_dim=48,
            num_layers=2,
            num_heads=4,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.proj_feat3 = proj_feat(16, (8,8,8), 64, (4,4,4), (4,4,4))

        self.encoder4 = UnetBasicBlock(
            spatial_dims=3,
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder3 = UnetUpBlock(
            spatial_dims=3,
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
        )
        self.decoder2 = UnetUpBlock(
            spatial_dims=3,
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
        )
        self.decoder1 = UnetUpBlock(
            spatial_dims=3,
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
        )
        self.out = UnetOutBlock(spatial_dims=3, in_channels=32, out_channels=2)  # type: ignore
        self.softmax = nn.Softmax()
        
        
        self.localization = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=7), nn.ReLU(),
            nn.MaxPool3d(2, stride=2), nn.Conv3d(64, 128, kernel_size=5),
            nn.ReLU(), nn.MaxPool3d(2, stride=2),
            nn.Conv3d(128, 256, kernel_size=3),nn.MaxPool3d(2, stride=2))

        # Regressor for the 3*4 affine matrix
        self.fc_loc = nn.Sequential(nn.Linear(256 * 13 * 13 * 13, 128),
                                    nn.ReLU(True), nn.Linear(128, 6))

        # Initial the weights/bias with indentity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float))
            
        self.enc1 = nn.Sequential(
            nn.Conv3d(2, 16, 3, 1, 1), nn.ReLU())
        self.enc2 = nn.Sequential(
            nn.Conv3d(16, 32, 3, 1, 1), nn.ReLU())
        self.enc3 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, 1), nn.ReLU())
        self.enc4 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, 1), nn.ReLU())
            
        self.dec5 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, 1), nn.ReLU())
            
        self.dec4 = nn.Sequential(
            nn.Conv3d(64, 32, 3, 1, 1), nn.ReLU())
        self.dec3 = nn.Sequential(
            nn.Conv3d(64, 32, 3, 1, 1), nn.ReLU())
        self.dec2 = nn.Sequential(
            nn.Conv3d(64, 32, 3, 1, 1), nn.ReLU())
        self.dec1 = nn.Sequential(
            nn.Conv3d(48, 32, 3, 1, 1), nn.ReLU())
            
        self.dec_r2 = nn.Sequential(
            nn.Conv3d(32, 16, 3, 1, 1), nn.ReLU())
        self.dec_r1 = nn.Sequential(
            nn.Conv3d(16, 16, 3, 1, 1), nn.ReLU())
        
        self.pool = nn.MaxPool3d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.flow = nn.Conv3d(16, 3, 3, 1, 1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        
        
        self.resize = layers.ResizeTransform(2, 3)
        self.fullsize = layers.ResizeTransform(0.5, 3)
        
        down_shape = [int(dim / 2) for dim in (128,128,128)]
        self.integrate = layers.VecInt(down_shape, 7)
        
        self.transformer = layers.SpatialTransformer((128,128,128))




    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights['state_dict']:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(weights['state_dict']['module.transformer.patch_embedding.position_embeddings_3d'])
            self.vit.patch_embedding.cls_token.copy_(weights['state_dict']['module.transformer.patch_embedding.cls_token'])
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(weights['state_dict']['module.transformer.patch_embedding.patch_embeddings.1.weight'])
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(weights['state_dict']['module.transformer.patch_embedding.patch_embeddings.1.bias'])

            # copy weights from  encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                print(block)
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights['state_dict']['module.transformer.norm.weight'])
            self.vit.norm.bias.copy_(weights['state_dict']['module.transformer.norm.bias'])


    def forward(self, x_in, fix_image, atlas_mask):
    
        enc1 = self.encoder1(x_in)
        x1, hidden_states_out1 = self.vit1(x_in)
        pro1 = self.proj_feat1(x1)
        enc2 = self.encoder2(self.pool(torch.cat((enc1, pro1), 1)))
        x2, hidden_states_out2 = self.vit2(self.pool(torch.cat((enc1, pro1), 1)))
        pro2 = self.proj_feat2(x2)
        enc3 = self.encoder3(self.pool(torch.cat((enc2, pro2), 1)))
        x3, hidden_states_out3 = self.vit3(self.pool(torch.cat((enc2, pro2), 1)))
        pro3 = self.proj_feat3(x3)
        enc4 = self.encoder4(self.pool(torch.cat((enc3, pro3), 1)))
        
        dec3 = self.decoder3(enc4, torch.cat((enc3, pro3), 1))
        dec2 = self.decoder2(dec3, torch.cat((enc2, pro2), 1))
        dec1 = self.decoder1(dec2, torch.cat((enc1, pro1), 1))
        logits = self.out(dec1)
        logits = self.softmax(logits)
        
        x_reg = torch.mul(logits[0,1,:,:,:], x_in[0,0,:,:,:]).float()
        
        del enc1, x1, pro1, enc2, x2, pro2, enc3, x3, pro3, enc4, dec3, dec2, dec1

        x_reg = x_reg.unsqueeze(dim=0)
        x_reg = x_reg.unsqueeze(dim=0)
        fix_image = fix_image.unsqueeze(dim=0)
        fix_image = fix_image.unsqueeze(dim=0)
        
        xs = torch.cat([x_reg, fix_image], dim=1)
        xs = self.localization(xs)
        xs = xs.view(-1, 256 * 13 * 13 * 13)
        y = self.fc_loc(xs)
        
        rx = torch.cos(y[0, 0]).repeat(4, 4) * torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=float, device='cuda') \
                 + torch.sin(y[0, 0]).repeat(4, 4) * torch.tensor([[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=float, device='cuda') \
                 + torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=float, device='cuda')
            
            # rotation y - y[0, 1]
        ry = torch.cos(y[0, 1]).repeat(4, 4) * torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=float, device='cuda') \
                 + torch.sin(y[0, 1]).repeat(4, 4) * torch.tensor([[0, 0, 1, 0], [0, 0, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0]], dtype=float, device='cuda') \
                 + torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=float, device='cuda')

            # rotation y - y[0, 2]
        rz = torch.cos(y[0, 2]).repeat(4, 4) * torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=float, device='cuda') \
                 + torch.sin(y[0, 2]).repeat(4, 4) * torch.tensor([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=float, device='cuda') \
                 + torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float, device='cuda')

            # translation x 
        d = y[0, 3:6].unsqueeze(1).repeat(1, 4)
        d = d * torch.tensor([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]], dtype=float, device='cuda')
            
            # transform matrix
        R = torch.mm(torch.mm(rx, ry), rz)
        theta = R[0:3, :] + d
        theta = theta.unsqueeze(0)


        grid = F.affine_grid(theta, fix_image.size()).float()
        moving_trans = F.grid_sample(fix_image, grid)
        
        atlas_mask = atlas_mask.unsqueeze(dim=0)
        atlas_mask = atlas_mask.unsqueeze(dim=0)
        moving_atlas_mask = F.grid_sample(atlas_mask, grid)
        
        del xs,y,rx,ry,rz,d,R
        
        
        x = torch.cat([moving_trans, x_reg], dim=1)
        x_enc1 = self.enc1(x)
        x_enc2 = self.enc2(self.pool(x_enc1))
        x_enc3 = self.enc3(self.pool(x_enc2))
        x_enc4 = self.enc4(self.pool(x_enc3))
        
        x_dec5 = self.dec5(self.pool(x_enc4))
        
        x_dec4 = self.dec4(torch.cat([self.upsample(x_dec5), x_enc4], dim=1))
        x_dec3 = self.dec3(torch.cat([self.upsample(x_dec4), x_enc3], dim=1))
        x_dec2 = self.dec2(torch.cat([self.upsample(x_dec3), x_enc2], dim=1))
        x_dec1 = self.dec1(torch.cat([self.upsample(x_dec2), x_enc1], dim=1))
        
        x_dec_r2 = self.dec_r2(x_dec1)
        x_dec_r1 = self.dec_r1(x_dec_r2)
        
        flow_field = self.flow(x_dec_r1)
        pos_flow = flow_field
        pos_flow = self.resize(pos_flow)
        pos_flow = self.integrate(pos_flow)
        pos_flow = self.fullsize(pos_flow)
        y_source = self.transformer(moving_trans, pos_flow)      
        atlas_mask_formable = self.transformer(moving_atlas_mask, pos_flow)
        
        del x, x_enc1, x_enc2,x_enc3,x_enc4,x_dec5,x_dec4,x_dec3,x_dec2,x_dec1,x_dec_r2,x_dec_r1, flow_field
        

        return logits, moving_trans, y_source, pos_flow, x_reg, atlas_mask_formable
