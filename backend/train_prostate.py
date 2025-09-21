# backend/train_prostate.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model_prostate import UNet3D   # your UNet3D file

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../data/PROMISE12/preprocessed"
MODEL_PATH = "models/prostate_model.pth"


# ---------------- Dataset ----------------
class ProstateDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.cases = []

        # Expect files like CaseXX_img.npy + CaseXX_mask.npy
        for fname in os.listdir(data_dir):
            if fname.endswith("_img.npy"):
                case_id = fname.replace("_img.npy", "")
                mask_file = os.path.join(data_dir, case_id + "_mask.npy")
                if os.path.exists(mask_file):
                    self.cases.append(case_id)

        if len(self.cases) == 0:
            raise RuntimeError(f"No PROMISE cases with segmentation found in {data_dir}")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_id = self.cases[idx]
        img_path = os.path.join(self.data_dir, case_id + "_img.npy")
        mask_path = os.path.join(self.data_dir, case_id + "_mask.npy")

        img = np.load(img_path).astype(np.float32)  # (D,H,W)
        mask = np.load(mask_path).astype(np.int64)  # (D,H,W)

        # --------- Speed-up: Downsample 2x ----------
        img = img[::2, ::2, ::2]
        mask = mask[::2, ::2, ::2]

        # Add channel
        img = np.expand_dims(img, axis=0)  # (1,D,H,W)

        return torch.tensor(img), torch.tensor(mask)


# ---------------- Main ----------------
def main():
    print(f"✅ Found PROMISE12 preproc: {DATA_DIR}")
    dataset = ProstateDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Lightweight UNet3D (fewer channels for speed)
    model = UNet3D(in_channels=1, out_channels=2, init_features=8).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 2  # keep small for testing
    for ep in range(epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)  # (B,C,D,H,W)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"[Prostate] Epoch {ep+1}/{epochs}, Loss={loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Prostate model saved at {MODEL_PATH}")


if __name__ == "__main__":
    main()





























