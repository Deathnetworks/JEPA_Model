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

def get_tokenizer_and_vocab(tokenizer_name="Qwen/Qwen2.5-7B-Instruct"):
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({'pad_token': '<|pad|>'})
    vocab_size = ((max(len(tok), tok.vocab_size) + 63) // 64) * 64
    return tok, vocab_size, tok.pad_token_id
