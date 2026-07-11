import torch
import gc
from contextlib import nullcontext

def get_device():
    # Since we are forced to target Intel Arc Pro B70 natively on Windows using XPU
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    else:
        return torch.device("cpu")

def empty_cache():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    gc.collect()

def get_autocast_kwargs():
    device = get_device()
    if device.type == "xpu":
        return {"device_type": "xpu", "dtype": torch.bfloat16}
    else:
        return {"device_type": "cpu", "enabled": False}

def autocast_ctx(device):
    if device.type == "xpu":
        return torch.autocast(device_type="xpu", dtype=torch.bfloat16)
    else:
        return nullcontext()
