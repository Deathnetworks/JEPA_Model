import time
import sys
import torch
import logging
from transformers import AutoTokenizer

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

from src.model_architecture import MambaJEPAEngine, DualStageLatentDecoder

# Reduce logging spam for the interactive CLI
logging.basicConfig(level=logging.WARNING, format='%(message)s')

def chunked_latent_ingestion(engine, input_ids, mamba_state, chunk_size=8192):
    seq_len = input_ids.shape[1]
    final_concept = None
    final_global_steps = None

    for start_idx in range(0, seq_len, chunk_size):
        end_idx = min(start_idx + chunk_size, seq_len)
        chunk_ids = input_ids[:, start_idx:end_idx]

        final_concept, final_global_steps, mamba_state = engine(chunk_ids, mamba_state=mamba_state)

        if hasattr(torch.xpu, 'empty_cache'):
            torch.xpu.empty_cache()

    return final_concept, final_global_steps, mamba_state

def print_header():
    print("="*60)
    print(" 🧠 Mamba2-Latent-Loop-8B Interactive Harness ".center(60))
    print("="*60)
    print("Type 'exit' or 'quit' to terminate the session.\n")

def run_interactive_cli():
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"[SYSTEM] Initializing inference pipeline on: {device}...")

    # 1. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Jackrong/Qwopus3.6-27B-v2", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    # 2. Instantiate and Load Models
    engine = MambaJEPAEngine(d_model=6144, num_blocks=32).to(device).eval()
    decoder = DualStageLatentDecoder(d_model=6144).to(device).eval()

    try:
        engine.load_state_dict(torch.load("jepa_engine.pth", map_location=device, weights_only=True), strict=False)
        decoder.load_state_dict(torch.load("latent_decoder.pth", map_location=device, weights_only=True), strict=False)
        print("[SYSTEM] Successfully loaded trained weights into VRAM.")
    except FileNotFoundError:
        print("[WARNING] Checkpoints not found. Running with initialized random weights for testing.")

    # 3. Hardware Warmup Pass (Forces IPEX/PyTorch to compile the graph kernels)
    print("[SYSTEM] Warming up XMX matrix lanes...")
    dummy_input = tokenizer("Warmup sequence.", return_tensors="pt").to(device)["input_ids"]
    with torch.no_grad():
        for _ in range(2):
            _concept, _, _ = engine(dummy_input)
            _logits = decoder(_concept)
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

            # Measured Execution Block
            start_time = time.perf_counter()

            with torch.no_grad():
                # Phase A: Latent Reasoning (ALGR)
                reasoning_start = time.perf_counter()
                student_concept, global_steps, mamba_state = chunked_latent_ingestion(engine, input_tokens, mamba_state)
                if hasattr(torch.xpu, 'synchronize'): torch.xpu.synchronize()
                reasoning_time = time.perf_counter() - reasoning_start

                # Phase B: Syntax Decoding
                decoding_start = time.perf_counter()
                logits = decoder(student_concept)
                if hasattr(torch.xpu, 'synchronize'): torch.xpu.synchronize()
                decoding_time = time.perf_counter() - decoding_start

                # Token Extraction
                token_ids = torch.argmax(logits, dim=-1)
                output_text = tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]

            total_time = time.perf_counter() - start_time
            avg_routing_hops = global_steps.float().mean().item()

            print("\n[Mamba-8B] --------------------------------------------------")
            print(output_text.strip())
            print("-------------------------------------------------------------")
            print(f"⏱️ Metrics: Reasoning: {reasoning_time:.3f}s | Decoding: {decoding_time:.3f}s | ALGR Hops: {avg_routing_hops:.1f}")

        except KeyboardInterrupt:
            print("\n[SYSTEM] Session interrupted. Goodbye.")
            sys.exit(0)
        except Exception as e:
            print(f"\n[ERROR] Engine fault: {e}")

if __name__ == "__main__":
    run_interactive_cli()