# train_breast.py
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from utils import find_preprocessed_dir

# config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = os.path.join("..", "data")
INBREAST_ROOT = os.path.join(DATA_ROOT, "INbreast")
# possible locations: "./data/INbreast/INbreast Release 1.0/preprocessed"
PREPROC_CANDIDATES = [
    os.path.join(INBREAST_ROOT, "preprocessed"),
    os.path.join(INBREAST_ROOT, "INbreast Release 1.0", "preprocessed"),
    os.path.join(INBREAST_ROOT, "INbreast Release 1.0"),
]
CSV_CANDIDATES = [
    os.path.join(INBREAST_ROOT, "INbreast.csv"),
    os.path.join(INBREAST_ROOT, "INbreast Release 1.0", "INbreast.csv"),
    os.path.join(INBREAST_ROOT, "INbreast Release 1.0", "INbreast.csv"),
]
MODEL_PATH = os.path.join("models", "breast_model.pth")
os.makedirs("models", exist_ok=True)

# ----------
def read_csv_auto(path):
    # auto-detect delimiter (handles semicolon CSVs)
    return pd.read_csv(path, sep=None, engine="python")

class BreastDataset(Dataset):
    def __init__(self, csv_path, preproc_dir):
        df = read_csv_auto(csv_path)
        # normalize candidate column names
        # possible variant: "Bi-Rads" or "Bi-Rads " etc
        columns = [c.strip() for c in df.columns]
        df.columns = columns

        # find label column
        label_col = None
        for cand in ["Bi-Rads", "Bi-RADS", "BIRADS", "BiRads", "Bi-Rads "]:
            if cand in df.columns:
                label_col = cand
                break
        if label_col is None:
            # try heuristic: last column
            label_col = df.columns[-1]
            print(f"[warning] Using last CSV column as label: {label_col}")

        # file id column
        file_col = None
        for cand in ["File Name", "FileName", "File_Name", "File name"]:
            if cand in df.columns:
                file_col = cand
                break
        if file_col is None:
            # try to find a numeric-looking column
            for c in df.columns:
                if df[c].astype(str).str.isnumeric().all():
                    file_col = c
                    break
        if file_col is None:
            raise ValueError("Could not determine which column contains file identifiers in CSV.")

        # filter rows
        df = df.dropna(subset=[file_col])
        self.samples = []
        preproc_files = os.listdir(preproc_dir)
        for _, row in df.iterrows():
            fid = str(row[file_col]).strip()
            # find first preproc file that contains fid substring (robust)
            candidate = None
            for f in preproc_files:
                if fid in f:
                    if f.lower().endswith(".npy") or f.lower().endswith(".png") or f.lower().endswith(".jpg"):
                        candidate = os.path.join(preproc_dir, f)
                        break
            if candidate and os.path.exists(candidate):
                # binary label: suspicious if Bi-Rads >= 4 (common rule)
                try:
                    label_raw = row[label_col]
                    label = 1 if float(str(label_raw).strip()) >= 4 else 0
                except Exception:
                    label = 0
                self.samples.append((candidate, label))
            else:
                # skip missing files rather than crash
                print(f"[skip] No preprocessed file found for ID {fid}")

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found. looked in: {preproc_dir}. Check preprocessed files and CSV.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = np.load(path).astype(np.float32)
        # ensure HxW shape
        if arr.ndim == 3:
            # some files may be (D,H,W) — take middle slice if so
            z = arr.shape[0] // 2
            arr = arr[z]
        arr = np.expand_dims(arr, 0)  # 1xHxW
        tensor = torch.from_numpy(arr)
        return tensor, torch.tensor(label, dtype=torch.long)

def get_model():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

# -------------- main training ---------------
def main():
    preproc_dir = find_preprocessed_dir(PREPROC_CANDIDATES)
    if preproc_dir is None:
        raise FileNotFoundError(f"Could not locate INbreast preprocessed directory. Candidates: {PREPROC_CANDIDATES}")

    csv_path = None
    for c in CSV_CANDIDATES:
        if os.path.exists(c):
            csv_path = c
            break
    if csv_path is None:
        raise FileNotFoundError("Could not find INbreast CSV metadata (INbreast.csv) under data folder.")

    print("Found preprocessed dir:", preproc_dir)
    dataset = BreastDataset(csv_path, preproc_dir)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = get_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 2
    for ep in range(epochs):
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"[breast] epoch {ep+1} loss {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Saved breast model to", MODEL_PATH)

if __name__ == "__main__":
    main()







