import torch
import logging

def get_device():
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")

def empty_cache():
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_autocast_kwargs():
    device = get_device()
    if device.type == "xpu":
        return {"device_type": "xpu", "dtype": torch.bfloat16}
    elif device.type == "cuda":
        return {"device_type": "cuda", "dtype": torch.bfloat16}
    else:
        return {"device_type": "cpu", "dtype": torch.float32}
