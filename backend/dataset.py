import os
import numpy as np
import torch
from torch.utils.data import Dataset  # ✅ missing import


class INbreastDataset(Dataset):
    def __init__(self, images_dir, labels_dict=None, transform=None):
        self.images_dir = images_dir
        self.files = sorted([f for f in os.listdir(images_dir) if f.endswith(".npy")])
        self.labels = labels_dict or {}
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = np.load(os.path.join(self.images_dir, fname)).astype(np.float32)

        # If 2D, make it (1,H,W)
        if img.ndim == 2:
            img = img[np.newaxis, :, :]  # -> (1,H,W)
        elif img.ndim == 3:
            # e.g. if already has channel or multiple slices
            img = img[0:1, :, :]  # take first channel

        # Normalize
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        x = torch.from_numpy(img).float()

        y = int(self.labels.get(fname, 0))  # default label = 0 if not found
        return x, y





