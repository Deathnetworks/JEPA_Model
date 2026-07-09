import sys
import os
import torch
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.routing.mamba_router import MambaGraphRouter

def test_router_halting_mechanism():
    """
    Test that the MambaGraphRouter correctly forces termination when global_steps >= max_budget.
    """
    batch_size = 2
    seq_len = 16
    d_model = 64
    d_latent = 2048
    num_blocks = 4

    router = MambaGraphRouter(d_model=d_model, num_blocks=num_blocks, d_latent=d_latent)

    h = torch.randn(batch_size, seq_len, d_latent)
    h_delta = torch.randn(batch_size, seq_len, 1)

    # Test step within budget
    global_steps_valid = torch.zeros(batch_size, seq_len, 1)
    probs_valid = router(h, h_delta, global_steps_valid, max_budget=4)

    # Assert halting probability is not heavily biased (unless learned)
    # The last index is the explicit halt action.
    assert torch.all(probs_valid[..., -2] < 0.99)

    # Test step exceeding budget
    global_steps_exceeded = torch.ones(batch_size, seq_len, 1) * 5
    probs_halted = router(h, h_delta, global_steps_exceeded, max_budget=4)

    # Assert halting probability is forced to 1.0 (or very close due to softmax scaling)
    assert torch.all(probs_halted[..., -2] > 0.99)
