import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_architecture import Mamba2LatentLoop8B

def test_engine_tensor_shapes():
    """
    Validates that the Mamba2-JEPA execution loop yields the exact tensor 
    dimensions specified in the master architecture document.
    """
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"[TEST RUNNER] Executing on natively targeted compute engine: {device}")

    # Initialize a miniature version of the 8B engine for local memory safety
    d_model = 64
    num_blocks = 2
    model = Mamba2LatentLoop8B(d_model=d_model, num_blocks=num_blocks, max_budget=4).to(device)
    model.eval()

    batch_size = 2
    seq_len = 512
    mock_tokens = torch.randint(0, 151643, (batch_size, seq_len)).to(device)

    # Initial state for Chunk 1
    mamba_state = None

    with torch.no_grad():
        # Execute the ALGR routed forward pass
        logits, jepa_concept, final_state = model(mock_tokens, mamba_state=mamba_state)

    print("\n--- Tensor Dimensionality Report ---")
    
    # 1. Check Logits (Cross-Entropy Target)
    print(f"Logits Shape: {list(logits.shape)}")
    assert logits.shape == (batch_size, seq_len, 151643), "❌ Logit dimension mismatch."
    
    # 2. Check JEPA Projection Vector
    print(f"JEPA Concept Shape: {list(jepa_concept.shape)}")
    assert jepa_concept.shape == (batch_size, 1024), "❌ JEPA Concept dimension mismatch."

    # 3. Check Mamba2 Recurrent State Matrix (For TBPTT state-passing)
    # Expected: [Batch, num_blocks, nheads, d_state, d_state]
    print(f"Mamba State Shape: {list(final_state.shape)}")
    assert final_state.dim() == 5, "❌ Mamba state matrix lacks expected recurrence depth."

    print("\n[TEST SUCCESS] All tensor shapes structurally align with ARCHITECTURE_SPEC.md")

if __name__ == "__main__":
    test_engine_tensor_shapes()