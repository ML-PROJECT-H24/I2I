import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

# Code taken from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
# We removed the code that was not necessary for our project and refactored the code

def nonlinearity(x):
    return x*torch.sigmoid(x) # swish


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels,in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (0,1,0,1), mode="constant", value=0)
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = in_channels != out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.use_conv_shortcut:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.use_conv_shortcut:
            x = self.shortcut(x)

        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)    # b,hw,c
        k = k.reshape(b,c,h*w)  # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_
    

class Encoder(nn.Module):
    def __init__(self, ch=128, ch_mult=(1,2,4), num_res_blocks=2, z_channels=3, dropout=0.0):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # (batch_size, 3, width, height) -> (batch_size, ch, width, height)
        self.conv_in = torch.nn.Conv2d(3, self.ch, kernel_size=3, stride=1, padding=1)

        #    (batch_size, ch,             width,                  height) 
        # -> (batch_size, ch*ch_mult[-1], width / 2^len(ch_mult), height / 2^len(ch_mult))
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i]
            block_out = ch * ch_mult[i]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)
        
        #    (batch_size, ch*ch_mult[-1], width / 2^len(ch_mult), height / 2^len(ch_mult))
        # -> (batch_size, ch*ch_mult[-1], width / 2^len(ch_mult), height / 2^len(ch_mult))
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout=dropout)
        self.mid.attn_1 = AttentionBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout=dropout)

        #    (batch_size, ch*ch_mult[-1], width / 2^len(ch_mult), height / 2^len(ch_mult))
        # -> (batch_size, 2*z_channels,   width / 2^len(ch_mult), height / 2^len(ch_mult))
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hs = [self.conv_in(x)]
        for i in range(self.num_resolutions):
            for j in range(self.num_res_blocks):
                hs.append(self.down[i].block[j](hs[-1]))
            if i != self.num_resolutions - 1:
                hs.append(self.down[i].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        mean, log_variance = torch.chunk(h, 2, dim=1)

        return mean, log_variance


class Decoder(nn.Module):
    def __init__(self, ch, ch_mult=(1,2,4), num_res_blocks=2, z_channels=3, dropout=0.0):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[-1]

        #    (batch_size, z_channels,     width / 2^len(ch_mult), height / 2^len(ch_mult)) 
        # -> (batch_size, ch*ch_mult[-1], width / 2^len(ch_mult), height / 2^len(ch_mult))
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        #    (batch_size, ch*ch_mult[-1], width / 2^len(ch_mult), height / 2^len(ch_mult))
        # -> (batch_size, ch*ch_mult[-1], width / 2^len(ch_mult), height / 2^len(ch_mult))
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout=dropout)
        self.mid.attn_1 = AttentionBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout=dropout)

        #    (batch_size, ch*ch_mult[-1], width / 2^len(ch_mult), height / 2^len(ch_mult))
        # -> (batch_size, ch*ch_mult[0],  width,                  height)
        self.up = nn.ModuleList()
        for i in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch*ch_mult[i]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up) # prepend to get consistent order

        #    (batch_size, ch*ch_mult[0], width, height)
        # -> (batch_size, 3,             width, height)
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.conv_in(z)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for i in reversed(range(self.num_resolutions)):
            for j in range(self.num_res_blocks+1):
                h = self.up[i].block[j](h)
            if i != 0:
                h = self.up[i].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h
