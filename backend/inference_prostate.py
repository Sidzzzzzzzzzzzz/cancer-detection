import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Minimal 3D UNet (adjust as per your training code)
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(UNet3D, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(2)
        self.bottleneck = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.conv_out = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder1(x)
        x = self.pool1(x1)
        x = self.bottleneck(x)
        x = self.up1(x)
        x = self.conv_out(x)
        return x

# Load model
model = UNet3D(in_channels=1, out_channels=2).to(device)
checkpoint = torch.load("models/unet3d_prostate.pth", map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.eval()

def predict_prostate(npy_path):
    try:
        volume = np.load(npy_path)  # shape (D, H, W)
        if volume.ndim == 3:
            volume = np.expand_dims(volume, axis=0)  # (1, D, H, W)
        volume = np.expand_dims(volume, axis=0)  # (1, 1, D, H, W)

        tensor = torch.tensor(volume, dtype=torch.float32).to(device)

        with torch.no_grad():
            output = model(tensor)  # (1, 2, D, H, W)
            probs = F.softmax(output, dim=1)

            # mean probability of class=1 across the full volume
            positive_prob = probs[:, 1].mean().item() * 100

        return {
            "prediction": "Positive" if positive_prob > 50 else "Negative",
            "confidence": round(positive_prob, 2),
        }
    except Exception as e:
        return {"error": str(e)}











































