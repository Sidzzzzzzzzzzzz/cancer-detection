# backend/model_prostate.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True)
    )

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, init_features=8):
        super().__init__()
        f = init_features
        # encoder
        self.enc1 = conv_block(in_channels, f)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = conv_block(f, f*2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = conv_block(f*2, f*4)
        self.pool3 = nn.MaxPool3d(2)
        # bottleneck
        self.bottleneck = conv_block(f*4, f*8)
        # decoder
        self.up3 = nn.ConvTranspose3d(f*8, f*4, kernel_size=2, stride=2)
        self.dec3 = conv_block(f*8, f*4)
        self.up2 = nn.ConvTranspose3d(f*4, f*2, kernel_size=2, stride=2)
        self.dec2 = conv_block(f*4, f*2)
        self.up1 = nn.ConvTranspose3d(f*2, f, kernel_size=2, stride=2)
        self.dec1 = conv_block(f*2, f)
        # out
        self.conv_out = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        out = self.conv_out(d1)
        return out

