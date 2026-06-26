import time
import sys
import torch
import logging
from transformers import AutoTokenizer

from src.model_architecture import Mamba2LatentLoop8B

# Suppress verbose warnings
logging.basicConfig(level=logging.WARNING, format='%(message)s')

def chunked_inference_ingestion(model, input_ids, mamba_state, chunk_size=4096):
    """
    Executes Truncated BPTT state-passing for inference.
    """
    seq_len = input_ids.shape[1]
    final_concept = None
    final_logits = None

    for start_idx in range(0, seq_len, chunk_size):
        end_idx = min(start_idx + chunk_size, seq_len)
        chunk_ids = input_ids[:, start_idx:end_idx]

        # Execute block and capture state for the next chunk
        final_logits, final_concept, mamba_state = model(chunk_ids, mamba_state=mamba_state)

        # Detach state to prevent computational graph memory bloat
        if mamba_state is not None:
            mamba_state = mamba_state.detach()

        if hasattr(torch.xpu, 'empty_cache'):
            torch.xpu.empty_cache()

    return final_logits, final_concept, mamba_state

def print_header():
    print("="*60)
    print(" 🧠 Mamba2-JEPA-8B Interactive Harness (XPU) ".center(60))
    print("="*60)
    print("Type 'exit' or 'quit' to terminate the session.\n")

def run_interactive_cli():
    # STRICT Native XPU Targeting (No IPEX)
    device = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    print(f"[SYSTEM] Initializing natively on: {device}...")

    # 1. Initialize Qwen 2.5 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    # 2. Instantiate Engine (Using bfloat16 for native Arc performance)
    model = Mamba2LatentLoop8B(d_model=6144, num_blocks=32).to(device).to(torch.bfloat16).eval()

    try:
        model.load_state_dict(torch.load("jepa_engine.pth", map_location=device, weights_only=True), strict=False)
        print("[SYSTEM] Successfully loaded trained weights into VRAM.")
    except FileNotFoundError:
        print("[WARNING] Checkpoints not found. Running with initialized random weights.")

    # 3. Hardware SYCL Graph Warmup
    print("[SYSTEM] Compiling SYCL execution graphs via torch.compile...")
    # Wrap with inductor for native hardware fusion
    model = torch.compile(model, backend="inductor", mode="max-autotune")
    
    dummy_input = tokenizer("Warmup sequence.", return_tensors="pt").to(device)["input_ids"]
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _ = model(dummy_input)
    if hasattr(torch.xpu, 'synchronize'):
        torch.xpu.synchronize()

    print_header()

    # 4. Interactive Loop
    mamba_state = None
    while True:
        try:
            prompt = input("\n[User] >>> ")
            if prompt.lower() in ['exit', 'quit']:
                print("[SYSTEM] Shutting down latent engine. Goodbye.")
                sys.exit(0)
            if not prompt.strip():
                continue

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_tokens = inputs["input_ids"]

            start_time = time.perf_counter()

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    # Phase A: Latent State Routing & Chunking
                    logits, student_concept, mamba_state = chunked_inference_ingestion(model, input_tokens, mamba_state)
                    
                    if hasattr(torch.xpu, 'synchronize'): 
                        torch.xpu.synchronize()
                    
                    # Phase B: Causal Extraction (Simplistic argmax for the test harness)
                    token_ids = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                    output_text = tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]

            total_time = time.perf_counter() - start_time

            print("\n[Mamba-8B] --------------------------------------------------")
            print(output_text.strip())
            print("-------------------------------------------------------------")
            print(f"⏱️ Execution Time: {total_time:.3f}s | Latent Vector Generated: True")

        except KeyboardInterrupt:
            print("\n[SYSTEM] Session interrupted. Goodbye.")
            sys.exit(0)
        except Exception as e:
            print(f"\n[ERROR] Engine fault: {e}")

if __name__ == "__main__":
    run_interactive_cli()