import sys
import os
import torch
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.backbone.mamba_engine import MambaJEPAEngine

def test_engine_tensor_shapes():
    """
    Validates that the modular Mamba2-JEPA execution loop yields the exact tensor
    dimensions required.
    """
    device = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")

    # Initialize a miniature version for testing
    d_model = 64
    num_blocks = 2
    vocab_size = 151643
    model = MambaJEPAEngine(vocab_size=vocab_size, d_model=d_model, num_blocks=num_blocks, use_sycl_kernel=False).to(device)

    batch_size = 2
    seq_len = 16
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    # Forward Pass
    logits, h = model(input_ids, max_budget=2)

    # Assertions
    assert h.shape == (batch_size, seq_len, 2048), f"Expected latent state shape (batch_size, seq_len, 2048), got {h.shape}"
    assert logits.shape == (batch_size, seq_len, vocab_size), f"Expected logits shape (batch_size, seq_len, vocab_size), got {logits.shape}"
