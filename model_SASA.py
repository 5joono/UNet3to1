import os
import numpy as np

import torch
import torch.nn as nn

from torch import einsum, nn, optim
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms

from torchsummary import summary

import matplotlib.pyplot as plt
import numpy as np

from einops import rearrange

import json
import os
import os
import numpy as np

import torch
import torch.nn as nn

from torch import einsum, nn, optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from torchsummary import summary

import matplotlib.pyplot as plt
import numpy as np

from einops import rearrange

import json
import os
      
class SASA(nn.Module):
    def __init__(self, in_channels, kernel_size, heads=1, dim_head=128, rel_pos_emb=False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        out_channels = heads * dim_head
        self.kernel_size = kernel_size

        self.to_q = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.to_kv = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, kvmap1, qmap, kvmap2):
        heads = self.heads
        b, c, h, w = qmap.shape
        
        padded_kvmap1 = F.pad(kvmap1, [self.kernel_size // 2, (self.kernel_size-1) // 2,self.kernel_size // 2, (self.kernel_size-1) // 2])
        padded_kvmap2 = F.pad(kvmap2, [self.kernel_size // 2, (self.kernel_size-1) // 2,self.kernel_size // 2, (self.kernel_size-1) // 2])
        padded_kvmap = torch.cat((padded_kvmap1.unsqueeze(-1), padded_kvmap2.unsqueeze(-1)),dim=-1)
        q = self.to_q(qmap) # b nd h w
        k = self.to_kv(padded_kvmap) # b nd h w 2
        v = self.to_kv(padded_kvmap) # b nd h w 2
        k = k.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1) # b nd h w 2 k k
        v = v.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1) # b nd h w 2 k k

        q = rearrange(q, 'b (n d) h w -> b n (h w) d', n=heads)
        k = rearrange(k, 'b (n d) h w l k1 k2 -> b n (h w) (l k1 k2) d', n=heads)
        v = rearrange(v, 'b (n d) h w l k1 k2 -> b n (h w) (l k1 k2) d', n=heads)

        q *= self.scale
        logits = einsum('b n x d, b n x y d -> b n x y', q, k)

        weights = self.softmax(logits)
        attn_out = einsum('b n x y, b n x y d -> b n x d', weights, v)
        attn_out = rearrange(attn_out, 'b n (h w) d -> b (n d) h w', h=h)

        return attn_out
     
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        k = 64

        self.enc1_1 = CBR2d(in_channels=1, out_channels=k)
        self.enc1_2 = CBR2d(in_channels=k, out_channels=k)
        self.att1 = SASA(in_channels=k, kernel_size=3, heads=4, dim_head=k//4)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=k, out_channels=2*k)
        self.enc2_2 = CBR2d(in_channels=2*k, out_channels=2*k)
        self.att2 = SASA(in_channels=2*k, kernel_size=3, heads=4, dim_head=2*k//4)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=2*k, out_channels=4*k)
        self.enc3_2 = CBR2d(in_channels=4*k, out_channels=4*k)
        self.att3 = SASA(in_channels=4*k, kernel_size=3, heads=4, dim_head=4*k//4)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=4*k, out_channels=8*k)
        self.enc4_2 = CBR2d(in_channels=8*k, out_channels=8*k)
        self.att4 = SASA(in_channels=8*k, kernel_size=3, heads=4, dim_head=8*k//4)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=8*k, out_channels=16*k)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=16*k, out_channels=8*k)

        self.unpool4 = nn.ConvTranspose2d(in_channels=8*k, out_channels=8*k,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 8*k, out_channels=8*k)
        self.dec4_1 = CBR2d(in_channels=8*k, out_channels=4*k)

        self.unpool3 = nn.ConvTranspose2d(in_channels=4*k, out_channels=4*k,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 4*k, out_channels=4*k)
        self.dec3_1 = CBR2d(in_channels=4*k, out_channels=2*k)

        self.unpool2 = nn.ConvTranspose2d(in_channels=2*k, out_channels=2*k,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 2*k, out_channels=2*k)
        self.dec2_1 = CBR2d(in_channels=2*k, out_channels=k)

        self.unpool1 = nn.ConvTranspose2d(in_channels=k, out_channels=k,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * k, out_channels=k)
        self.dec1_1 = CBR2d(in_channels=k, out_channels=k)

        self.fc = nn.Conv2d(in_channels=k, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        a, b, c = torch.chunk(x, 3, -1)
        a = self.enc1_1(a)
        a = self.enc1_2(a)
        b = self.enc1_1(b)
        b = self.enc1_2(b)
        c = self.enc1_1(c)
        c = self.enc1_2(c)
        enc1_2b = self.att1(a, b, c)
        a = self.pool1(a)
        b = self.pool1(enc1_2b)
        c = self.pool1(c)

        a = self.enc2_1(a)
        a = self.enc2_2(a)
        b = self.enc2_1(b)
        b = self.enc2_2(b)
        c = self.enc2_1(c)
        c = self.enc2_2(c)
        enc2_2b = self.att2(a, b, c)
        a = self.pool2(a)
        b = self.pool2(enc2_2b)
        c = self.pool2(c)

        a = self.enc3_1(a)
        a = self.enc3_2(a)
        b = self.enc3_1(b)
        b = self.enc3_2(b)
        c = self.enc3_1(c)
        c = self.enc3_2(c)
        enc3_2b = self.att3(a, b, c)
        a = self.pool3(a)
        b = self.pool3(enc3_2b)
        c = self.pool3(c)

        a = self.enc4_1(a)
        a = self.enc4_2(a)
        b = self.enc4_1(b)
        b = self.enc4_2(b)
        c = self.enc4_1(c)
        c = self.enc4_2(c)
        enc4_2b = self.att4(a, b, c)
        a = self.pool4(a)
        b = self.pool4(enc4_2b)
        c = self.pool4(c)

        b = self.enc5_1(b)
        b = self.dec5_1(b)

        b = self.unpool4(b)
        b = torch.cat((b, enc4_2b), dim=1)
        b = self.dec4_2(b)
        b = self.dec4_1(b)

        b = self.unpool3(b)
        b = torch.cat((b, enc3_2b), dim=1)
        b = self.dec3_2(b)
        b = self.dec3_1(b)

        b = self.unpool2(b)
        b = torch.cat((b, enc2_2b), dim=1)
        b = self.dec2_2(b)
        b = self.dec2_1(b)

        b = self.unpool1(b)
        b = torch.cat((b, enc1_2b), dim=1)
        b = self.dec1_2(b)
        b = self.dec1_1(b)

        b = self.fc(b)

        return b
