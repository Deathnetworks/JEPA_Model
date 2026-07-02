import torch
import re
import subprocess
import os
from transformers import AutoTokenizer
import logging

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_architecture import MambaJEPAEngine, ClosedLoopLatentDecoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_rust_code(text: str) -> str:
    """Extracts Rust code from a markdown block or returns the text itself."""
    # Uses hex escape sequences (\x60) for backticks to completely prevent markdown renderer truncation
    match = re.search(r"\x60\x60\x60(?:rust)?\n(.*?)\x60\x60\x60", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

def setup_device():
    """
    Ensure the target device is the Intel Arc Pro B70 GPU (xpu) if available.
    """
    if torch.xpu.is_available():
        device = torch.device("xpu")
        logging.info(f"Targeting native Intel GPU compute via device: {device}")
    else:
        device = torch.device("cpu")
        logging.warning("XPU not available, falling back to CPU. Performance will be degraded.")
    return device

class InferencePipeline:
    def __init__(self, engine_path="jepa_engine.pth", decoder_path="latent_decoder.pth", tokenizer_name="Qwen/Qwen2.5-7B-Instruct"):
        self.device = setup_device()

        logging.info(f"Loading tokenizer {tokenizer_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        logging.info("Instantiating MambaJEPAEngine and ClosedLoopLatentDecoder...")
        if self.device.type == 'cpu':
            logging.warning("Running on CPU, using heavily downgraded hyperparameters to avoid OOM.")
            self.engine = MambaJEPAEngine(d_model=64, num_blocks=2, max_budget=2, d_latent=1024).to(self.device)
            self.decoder = ClosedLoopLatentDecoder(d_model=64, d_latent=1024).to(self.device)
        else:
            self.engine = MambaJEPAEngine().to(self.device)
            self.decoder = ClosedLoopLatentDecoder().to(self.device)

        if self.device.type == "xpu":
            torch._inductor.config.freezing = True
            torch._inductor.config.max_autotune = True
            torch._inductor.config.coordinate_descent_tuning = True
            self.engine = torch.compile(self.engine, backend="inductor")
            self.decoder = torch.compile(self.decoder, backend="inductor")

        logging.info(f"Loading weights from {engine_path} and {decoder_path}...")
        try:
            self.engine.load_state_dict(torch.load(engine_path, map_location=self.device, weights_only=True), strict=False)
            logging.info(f"Successfully loaded {engine_path}")
        except FileNotFoundError:
            logging.warning(f"Could not find {engine_path}, initializing with random weights.")

        try:
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device, weights_only=True), strict=False)
            logging.info(f"Successfully loaded {decoder_path}")
        except FileNotFoundError:
            logging.warning(f"Could not find {decoder_path}, initializing with random weights.")

        self.engine.eval()
        self.decoder.eval()

    def generate_code(self, prompt: str, max_retries: int = 3):
        current_prompt = prompt
        extracted_code = ""
        chunk_size = 4096

        for attempt in range(max_retries):
            logging.info(f"Attempt {attempt + 1}/{max_retries} for prompt: '{current_prompt[:50]}...'")

            inputs = self.tokenizer(current_prompt, return_tensors="pt").to(self.device)
            input_tokens = inputs["input_ids"]
            seq_len = input_tokens.size(1)

            mamba_state = None
            student_concept = None

            with torch.no_grad():
                for t in range(0, seq_len, chunk_size):
                    chunk_input = input_tokens[:, t:t+chunk_size]

                    # --- NEW: Dynamic Inference Budgeting (arXiv:2604.07822) ---
                    # Cap the compute loops based on sequence complexity to prevent "Overthinking" degradation.
                    # Base budget of 8, scaling up by 1 loop per 64 tokens, capped at absolute max (64)
                    dynamic_budget = min(64, max(8, chunk_input.size(1) // 64))

                    with torch.autocast(device_type="xpu" if self.device.type == "xpu" else "cpu", dtype=torch.bfloat16 if self.device.type == "xpu" else torch.float32):
                        student_concept, _, mamba_state = self.engine(
                            chunk_input, 
                            mamba_state=mamba_state,
                            active_budget=dynamic_budget # Enforce the mathematical bound
                        )

                    if mamba_state is not None:
                        mamba_state = mamba_state.detach()

                    if hasattr(torch, 'xpu') and hasattr(torch.xpu, 'empty_cache'):
                        torch.xpu.empty_cache()

                # Enforce complete token-by-token autoregressive cross-attention loop decoding
                max_gen_len = self.decoder.max_seq_len
                generated_ids = torch.full((1, 1), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
                
                for step in range(max_gen_len):
                    with torch.autocast(device_type="xpu" if self.device.type == "xpu" else "cpu", dtype=torch.bfloat16 if self.device.type == "xpu" else torch.float32):
                        logits = self.decoder(generated_ids, student_concept)
                    
                    next_token_logits = logits[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
                    
                    if next_token_id.item() == self.tokenizer.eos_token_id:
                        break

                decoded_text = self.tokenizer.batch_decode(generated_ids[:, 1:], skip_special_tokens=True)[0]

            logging.info(f"\n--- Raw Generated Output (Attempt {attempt + 1}) ---")
            print(decoded_text)

            extracted_code = extract_rust_code(decoded_text)

            logging.info(f"\n--- Extracted Rust Code ---")
            print(extracted_code)
            logging.info("------------------------\n")

            temp_file = "temp_agent_output.rs"
            with open(temp_file, "w") as f:
                f.write(extracted_code)

            try:
                logging.info(f"Running rustc on {temp_file}...")
                result = subprocess.run(
                    ["rustc", temp_file, "--color", "never"],
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    logging.info("Rust compilation SUCCESSFUL.")
                    break
                else:
                    logging.warning(f"Rust compilation FAILED. Exit code {result.returncode}")
                    error_msg = result.stderr
                    current_prompt = f"{prompt}\n\nThe previous attempt failed with:\n{error_msg}\nPlease fix the architectural logic."

            except Exception as e:
                logging.error(f"Compilation execution failed: {e}")
                break

        try:
            if os.path.exists("temp_agent_output.rs"):
                os.remove("temp_agent_output.rs")
            if os.path.exists("temp_agent_output"):
                os.remove("temp_agent_output")
            if os.path.exists("temp_agent_output.exe"):
                os.remove("temp_agent_output.exe")
            if os.path.exists("temp_agent_output.pdb"):
                os.remove("temp_agent_output.pdb")
        except Exception as e:
            logging.warning(f"Cleanup error: {e}")

        return extracted_code

if __name__ == "__main__":
    pipeline = InferencePipeline()
    prompt = "Write a Rust struct for a Tauri SQLite database connection."
    pipeline.generate_code(prompt)