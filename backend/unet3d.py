import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):  # ⚡ keep 4 instead of 8
        super(UNet3D, self).__init__()
        features = init_features

        self.encoder1 = self._block(in_channels, features)
        self.encoder2 = self._block(features, features * 2)

        self.bottleneck = self._block(features * 2, features * 4)

        self.up2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features * 4, features * 2)

        self.up1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)

        self.conv_out = nn.Conv3d(features, out_channels, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool3d(2)(enc1))

        bottleneck = self.bottleneck(nn.MaxPool3d(2)(enc2))

        dec2 = self.up2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv_out(dec1)


def get_unet3d(in_channels=1, out_channels=1):
    return UNet3D(in_channels=in_channels, out_channels=out_channels, init_features=4)  # ✅ match training












