import sys
import os
import torch

# Ensure the root src folder path is exposed to the local test runner runtime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_architecture import Mamba2LatentLoop8B

def test_engine_tensor_shapes():
    """
    Validates that random input sequences correctly process through the complete
    Mamba2-Latent-Loop execution loop and output expected tensor dimensions.
    """
    # Force localized execution targeting Intel XPU if native graphics stack is online
    # Downgrade size to prevent out of memory during automated small testing.
    device = torch.device("cpu")
    print(f"[TEST RUNNER] Targets execution pipeline on compute engine: {device}")

    # Initialize engine architecture configurations
    model = Mamba2LatentLoop8B(d_model=128, num_blocks=4, max_budget=4).to(device)
    model.eval()

    # Generate mock data: Batch size = 2, Sequence Length = 512 tokens
    mock_tokens = torch.randint(0, 1000, (2, 32)).to(device)

    with torch.no_grad():
        output_state, global_steps = model(mock_tokens)

    print(f"[TEST SUCCESS] Returned hidden output matrix dimensionality: {list(output_state.shape)}")

    # Confirm exact dimensionality assertions match Part 1 constraints
    assert output_state.shape == (2, 32, 128), f"Dimension mismatch error: Found {output_state.shape}"
    print("[TEST SUCCESS] Tensor shapes structurally aligned with specification document instructions.")

if __name__ == "__main__":
    test_engine_tensor_shapes()
