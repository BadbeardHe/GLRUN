'''
Author: Jin Zeng
Date: 2023-04-12
LastEditTime: 2023-08-07
Description: GLRUN for iToF denoising
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from torch.utils.data import Dataset


class conv3x3(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(conv3x3, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 3, 1, 0)
    def forward(self, input):
        input = F.pad(input, (1, 1, 1, 1), mode='constant').contiguous()
        return self.conv(input)
    

class feat_extract_submodule(nn.Module):
    def __init__(self, input_channels, output_channels, conv_depth = 3):
        super(feat_extract_submodule, self).__init__()
        submodule = []
        submodule.append(conv3x3(input_channels, output_channels))
        submodule.append(nn.LeakyReLU())
        for i in range(conv_depth - 1):
            submodule.append(conv3x3(output_channels, output_channels))
            submodule.append(nn.LeakyReLU())
        self.seq = nn.Sequential(*submodule)
    def forward(self, input):
        return self.seq(input)


class feat_extract_submodule_fin(nn.Module):
    def __init__(self, input_channels, output_channels, conv_depth = 3):
        super(feat_extract_submodule_fin, self).__init__()
        submodule = []
        submodule.append(conv3x3(input_channels, output_channels))
        if conv_depth>1:
            submodule.append(nn.LeakyReLU())
            for i in range(conv_depth - 2):
                submodule.append(conv3x3(output_channels, output_channels))
                submodule.append(nn.LeakyReLU())
            submodule.append(conv3x3(output_channels, output_channels))
        
        self.seq = nn.Sequential(*submodule)
    def forward(self, input):
        return self.seq(input)
    

def weights_init_normal(m):
    """
    Initialize weights of convolutional layers
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


class cnny(nn.Module):
    def __init__(self):
        super(cnny, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):     
        out = self.layer(x)
        out = x + out

        return out
    

class UGLR(nn.Module):
    def __init__(self, iter_time):
        super(UGLR, self).__init__()
        self.times = iter_time

    def svconv(self, input, kernel, kernel_size, stride=1, padding=0, dilation=1, native_impl=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        (bs, ch), in_sz = input.shape[:2], input.shape[2:]

        cols = F.unfold(input, kernel_size, dilation, padding, stride)
        output = cols.view(bs, ch, *kernel.shape[2:]) * kernel
        output = torch.einsum('ijklmn->ijmn', (output,))
        return output

    def forward(self, x_init, affinity_bias, mu, K=3):
        # Inputs
        # x: pre_filtered img; affinity: guidance; mu: weight for GLR
      
        B, C, H, W = x_init.size()
        w_pre = 0.1
        x_result = x_init
       
        kernel = F.softmax(affinity_bias[:,:K*K,:,:], dim=1)
        kernel = kernel.reshape(B,3,3,H,W).unsqueeze(dim=1)
        bias = affinity_bias[:,K*K:,:,:]

        for i in range(self.times):
            x_result = (mu * x_result +  w_pre*x_init)/(mu + w_pre)   # for ablation 2
            x_result = self.svconv(x_result, kernel, kernel_size=3, stride=1, padding=1, dilation=1)           
            
        return x_result+bias
    

class GLRUN(nn.Module):
    def __init__(self, input_channel=2, ch0 = 16, K = 3, conv_depth = 2, unet_depth = 3):
        super(GLRUN, self).__init__()
        self.K = K
        self.channels = [ch0]
        self.unet_depth = unet_depth
        for i in range(unet_depth - 1):
            self.channels.append(ch0 * 2)
            ch0 *= 2
        self.channels.append(ch0)
        if unet_depth>2:
            self.channels.append(ch0)
            for i in range(unet_depth - 3):
                self.channels.append(ch0 // 2)
                ch0 //= 2
        self.channels.append(input_channel*K*K+2)
        # self.channels.append(1*K*K+1)
        curr_chn = self.channels
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.up = nn.ModuleList()
        self.bilinear = nn.ModuleList()
        self.down.append(feat_extract_submodule(input_channel, curr_chn[0], conv_depth = conv_depth))
        for i in range(unet_depth):
            self.pool.append(nn.AvgPool2d(2))
            self.down.append(feat_extract_submodule(curr_chn[i], curr_chn[i + 1], conv_depth = conv_depth))
        for i in range(unet_depth - 2):
            self.bilinear.append(nn.UpsamplingBilinear2d(scale_factor=2))
            self.up.append(feat_extract_submodule(curr_chn[unet_depth + i] + curr_chn[unet_depth - i - 1], curr_chn[unet_depth + i + 1], conv_depth = conv_depth))
        
        i = unet_depth - 2
        self.bilinear.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.up.append(feat_extract_submodule_fin(curr_chn[unet_depth + i] + curr_chn[unet_depth - i - 1], curr_chn[unet_depth + i + 1], conv_depth = conv_depth))

        self.bilinear.append(nn.UpsamplingBilinear2d(scale_factor=2))

        self.mu_dec0_1 = feat_extract_submodule_fin(curr_chn[unet_depth + i] + curr_chn[unet_depth - i - 1], input_channel, conv_depth = conv_depth)

        # Depth refinement
        self.uglr = UGLR(5)

    def forward(self, concat_IQ):
        unet_depth = self.unet_depth
        # concat_IQ = torch.cat((input_I, input_Q), dim = 1)
        feature_maps = []
        feature_maps.append(self.down[0](concat_IQ))
        for i in range(unet_depth):
            pooled = self.pool[i](feature_maps[i])
            feature_maps.append(self.down[i+1](pooled))
        for i in range(unet_depth - 1):
            bilineared = self.bilinear[i](feature_maps[unet_depth + i])
            concated = torch.cat((bilineared, feature_maps[unet_depth - i - 1]), 1)
            feature_maps.append(self.up[i](concated))

        guide = self.bilinear[unet_depth - 1](feature_maps[-1])
        
        # Mu Decoding
        mu_feat = self.mu_dec0_1(concated) #up
        mu = self.bilinear[unet_depth - 1](mu_feat) #up
        mu = torch.sigmoid(mu)

        xout_0 = self.uglr(concat_IQ[:,0:1,:,:], guide[:,:(self.K*self.K+1),:,:], mu[:,0:1,:,:])
        xout_1 = self.uglr(concat_IQ[:,1:2,:,:], guide[:,(self.K*self.K+1):,:,:], mu[:,1:2,:,:])

        return torch.concat((xout_0, xout_1),axis=1), mu


if __name__ == '__main__':
    device = torch.device("cuda")
    glrun = GLRUN()
    glrun = glrun.to(device)
    inputs = torch.rand((12, 2, 424, 512)).to(device)
    out = glrun(inputs)
    print("Done")
