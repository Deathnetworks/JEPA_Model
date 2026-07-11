import torch
import gc
from contextlib import nullcontext
from transformers import AutoTokenizer

def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")

def empty_cache():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    gc.collect()

def autocast_ctx(device=None):
    device = device or get_device()
    if device.type in ("xpu", "cuda"):
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return nullcontext()

def get_vocab_size(tokenizer_name="Qwen/Qwen2.5-7B-Instruct"):
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    # Pad to nearest multiple of 64 for native Intel XMX GEMM alignment
    return ((max(len(tok), tok.vocab_size) + 63) // 64) * 64
