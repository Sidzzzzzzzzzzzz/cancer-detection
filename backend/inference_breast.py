import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_breast_model():
    # ResNet18 with modified input (1-channel) and output (binary classification)
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load("models/resnet_inbreast.pth", map_location=device)
    model.load_state_dict(checkpoint, strict=False)  # allow partial load
    model.to(device)
    model.eval()
    return model

model = load_breast_model()

def predict_breast(npy_path):
    try:
        img = np.load(npy_path)  # could be (H, W), (1, H, W), or (2, H, W)

        if img.ndim == 2:  # grayscale
            img = np.expand_dims(img, axis=0)  # (1, H, W)
        elif img.shape[0] == 2:  # 2-channel input → average channels
            img = img.mean(axis=0, keepdims=True)  # (1, H, W)

        # normalize to [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 1, H, W)

        with torch.no_grad():
            output = model(tensor)  # (1, 2)
            probs = F.softmax(output, dim=1)
            pred = probs.argmax(dim=1).item()
            confidence = probs.max().item() * 100

        return {
            "prediction": "Positive" if pred == 1 else "Negative",
            "confidence": round(confidence, 2),
        }
    except Exception as e:
        return {"error": str(e)}





































