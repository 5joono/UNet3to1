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

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, nh, l, _ = x.shape
    col_pad = torch.zeros((b, nh, l, 1)).cuda()
    x = torch.cat((x, col_pad), dim=3)
    flat_x = rearrange(x, 'b nh l c -> b nh (l c)')
    flat_pad = torch.zeros((b, nh, l-1)).cuda()
    flat_x = torch.cat((flat_x, flat_pad), dim=2)
    final_x = torch.reshape(flat_x, (b, nh, l+1, 2*l-1))
    return final_x[:,:,:l,(l-1):]

def relative_logits_1d(q, rel_k):
    b, n, hq, wq, _ = q.shape
    logits = einsum('b n hq wq d, r d -> b n hq wq r', q, rel_k)
    logits = rearrange(logits, 'b n hq wq r -> b (n hq) wq r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, n, hq, wq, None)
    logits = expand_dim(logits, dim = 3, k = hq)
    return logits

class AbsPosEmb(nn.Module):
    def __init__(self, hq, wq, hk, wk, dim_head):
        super().__init__()
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(hk, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(wk, dim_head) * scale)
        
    def forward(self, q):
        emb = rearrange(self.height, 'hk d -> hk () d') + rearrange(self.width, 'wk d -> () wk d')
        emb = rearrange(emb, 'hk wk d -> (hk wk) d')
        logits = einsum('b n x d, y d -> b n x y', q, emb)
        return logits

class RelPosEmb(nn.Module):
    def __init__(self, hq, wq, hk, wk, dim_head):
        super().__init__()
        scale = dim_head ** -0.5
        self.hq = hq
        self.wq = wq
        self.hk = hk
        self.wk = wk
        self.rel_height = nn.Parameter(torch.randn(hk * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(wk * 2 - 1, dim_head) * scale)

    def forward(self, q):
        q = rearrange(q, 'b n (hq wq) d -> b n hq wq d', hq=self.hq)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b n hq hk wq wk-> b n (hq wq) (hk wk)')

        q = rearrange(q, 'b n hq wq d -> b n wq hq d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b n wq wk hq hk -> b n (hq wq) (hk wk)')
        return rel_logits_w + rel_logits_h        
        
class MHSA(nn.Module):
    def __init__(self, in_channels, qmap_size, kvmap_size, heads=1, dim_head=128, rel_pos_emb=False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        out_channels = heads * dim_head
        hq, wq = qmap_size
        hk, wk = kvmap_size

        self.to_qkv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        if rel_pos_emb:
            self.pos_emb = RelPosEmb(hq, wq, hk, wk, dim_head)
        else:
            self.pos_emb = AbsPosEmb(hq, wq, hk, wk, dim_head)

    def forward(self, qmap, kvmap):
        heads = self.heads
        b, c, hq, wq = qmap.shape
        b, c, hk, wk = kvmap.shape
        q = self.to_qkv(qmap)
        k = self.to_qkv(kvmap)
        v = self.to_qkv(kvmap)
        q = rearrange(q, 'b (n d) hq wq -> b n (hq wq) d', n=heads)
        k = rearrange(k, 'b (n d) hk wk -> b n (hk wk) d', n=heads)
        v = rearrange(v, 'b (n d) hk wk -> b n (hk wk) d', n=heads)

        q *= self.scale
        #print(q.shape)
        #print(k.shape)
        logits = einsum('b n x d, b n y d -> b n x y', q, k)
        logits += self.pos_emb(q)

        weights = self.softmax(logits)
        attn_out = einsum('b n x y, b n y d -> b n x d', weights, v)
        attn_out = rearrange(attn_out, 'b n (hq wq) d -> b (n d) hq wq', hq=hq)

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
        xx = 96
        yy = 96
        # Contracting path

        self.enc1_1 = CBR2d(in_channels=1, out_channels=k)
        self.enc1_2 = CBR2d(in_channels=k, out_channels=k)
        self.att1 = MHSA(in_channels=k, qmap_size = (xx,yy), kvmap_size = (xx,2*yy), dim_head=k)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=k, out_channels=2*k)
        self.enc2_2 = CBR2d(in_channels=2*k, out_channels=2*k)
        self.att2 = MHSA(in_channels=2*k, qmap_size = (xx//2,yy//2), kvmap_size = (xx//2,yy), dim_head=2*k)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=2*k, out_channels=4*k)
        self.enc3_2 = CBR2d(in_channels=4*k, out_channels=4*k)
        self.att3 = MHSA(in_channels=4*k, qmap_size = (xx//4,yy//4), kvmap_size = (xx//4,yy//2), dim_head=4*k)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=4*k, out_channels=8*k)
        self.enc4_2 = CBR2d(in_channels=8*k, out_channels=8*k)
        self.att4 = MHSA(in_channels=8*k, qmap_size = (xx//8,yy//8), kvmap_size = (xx//8,yy//4), dim_head=8*k)
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
        a, b, c = torch.chunk(x,3,-1)
        a = self.enc1_1(a)
        a = self.enc1_2(a)
        b = self.enc1_1(b)
        b = self.enc1_2(b)
        c = self.enc1_1(c)
        c = self.enc1_2(c)
        enc1_2b = self.att1(b, torch.cat((a, c), dim=-1))
        a = self.pool1(a)
        b = self.pool1(enc1_2b)
        c = self.pool1(c)

        a = self.enc2_1(a)
        a = self.enc2_2(a)
        b = self.enc2_1(b)
        b = self.enc2_2(b)
        c = self.enc2_1(c)
        c = self.enc2_2(c)
        enc2_2b = self.att2(b, torch.cat((a, c), dim=-1))
        a = self.pool2(a)
        b = self.pool2(enc2_2b)
        c = self.pool2(c)

        a = self.enc3_1(a)
        a = self.enc3_2(a)
        b = self.enc3_1(b)
        b = self.enc3_2(b)
        c = self.enc3_1(c)
        c = self.enc3_2(c)
        enc3_2b = self.att3(b, torch.cat((a, c), dim=-1))
        a = self.pool3(a)
        b = self.pool3(enc3_2b)
        c = self.pool3(c)

        a = self.enc4_1(a)
        a = self.enc4_2(a)
        b = self.enc4_1(b)
        b = self.enc4_2(b)
        c = self.enc4_1(c)
        c = self.enc4_2(c)
        enc4_2b = self.att4(b, torch.cat((a, c), dim=-1))
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
