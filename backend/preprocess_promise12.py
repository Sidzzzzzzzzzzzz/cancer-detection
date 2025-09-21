# Example stub – implement as needed
import os
import nibabel as nib
import numpy as np

def preprocess_promise12(raw_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(raw_dir):
        if f.endswith(".nii.gz") and "img" in f:
            img = nib.load(os.path.join(raw_dir,f)).get_fdata()
            np.save(os.path.join(out_dir,f.replace(".nii.gz",".npy")), img)






