# utils.py
import os
import torch

def find_preprocessed_dir(base_paths):
    """Return first existing directory from list of base_paths"""
    for p in base_paths:
        if os.path.isdir(p):
            return p
    return None

def load_state_dict_flexible(model, checkpoint_path, map_location="cpu"):
    """
    Loads checkpoint but only assigns parameters whose shapes match the model.
    This avoids runtime errors from mismatched fc/out channel sizes.
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    # if checkpoint is a dict with extra keys like 'model', attempt to extract
    if isinstance(ckpt, dict):
        # find likely state_dict key
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt

    # remove common prefixes
    new_state = {}
    for k, v in state.items():
        key = k
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("model."):
            key = key[len("model.") :]
        new_state[key] = v

    model_state = model.state_dict()
    filtered = {}
    for k, v in new_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        # else: skip mismatched shapes

    model_state.update(filtered)
    model.load_state_dict(model_state, strict=False)
    return list(filtered.keys())


