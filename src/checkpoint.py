import torch
import logging

def save_ckpt(model, path):
    state_dict = model.state_dict()
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace("_orig_mod.", "")
        clean_state_dict[clean_key] = v
    torch.save(clean_state_dict, path)
    logging.info(f"Model saved to {path}")

def load_ckpt(model, path, device):
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
        # Attempt to load cleanly. PyTorch handles _orig_mod. internally for compiled models when loading in PyTorch > 2.0,
        # but to be safe, we can use strict=False
        model.load_state_dict(state_dict, strict=False)
        logging.info(f"Successfully loaded {path}")
        return True
    except Exception as e:
        logging.warning(f"Failed to load {path}: {e}")
        return False
