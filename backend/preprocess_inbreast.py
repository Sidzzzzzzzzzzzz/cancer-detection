# Example stub – implement as needed
import os
from PIL import Image

def preprocess_inbreast(raw_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(raw_dir):
        if f.lower().endswith(".jpg") or f.lower().endswith(".png"):
            img = Image.open(os.path.join(raw_dir,f)).convert("L")
            img.save(os.path.join(out_dir,f))



