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

from src.model_architecture import MambaJEPAEngine, DualStageLatentDecoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_rust_code(text: str) -> str:
    """Extracts Rust code from a markdown block or returns the text itself."""
    match = re.search(r"```(?:rust)?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

def setup_device():
    """
    Ensure the target device is the Intel Arc Pro B70 GPU (xpu).
    """
    try:
        import intel_extension_for_pytorch as ipex
        logging.info("Successfully imported intel_extension_for_pytorch.")
    except ImportError:
        logging.warning("intel_extension_for_pytorch not found.")

    device = torch.device("xpu")
    logging.info(f"Targeting native Intel GPU compute via device: {device}")
    return device

class InferencePipeline:
    def __init__(self, engine_path="jepa_engine.pth", decoder_path="latent_decoder.pth", tokenizer_name="Qwen/Qwen3.6-27B"):
        self.device = setup_device()

        logging.info(f"Loading tokenizer {tokenizer_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        logging.info("Instantiating MambaJEPAEngine and DualStageLatentDecoder...")
        self.engine = MambaJEPAEngine().to(self.device)
        self.decoder = DualStageLatentDecoder().to(self.device)

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

        for attempt in range(max_retries):
            logging.info(f"Attempt {attempt + 1}/{max_retries} for prompt: '{current_prompt[:50]}...'")

            # Tokenize prompt
            inputs = self.tokenizer(current_prompt, return_tensors="pt").to(self.device)
            input_tokens = inputs["input_ids"]

            with torch.no_grad():
                # Pass through MambaJEPAEngine
                # Returns student_concept [Batch, 1024], global_steps [Batch, Seq_Len, 1]
                student_concept, _ = self.engine(input_tokens)

                # Pass through DualStageLatentDecoder
                # Returns logits [Batch, 256, Vocab_Size]
                logits = self.decoder(student_concept)

                # Argmax to get token IDs
                token_ids = torch.argmax(logits, dim=-1) # [Batch, 256]

                # Decode tokens
                decoded_text = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]

            logging.info(f"\n--- Raw Generated Output (Attempt {attempt + 1}) ---")
            print(decoded_text)

            extracted_code = extract_rust_code(decoded_text)

            logging.info(f"\n--- Extracted Rust Code ---")
            print(extracted_code)
            logging.info("------------------------\n")

            # Subprocess Compilation
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

            finally:
                # Cleanup iteration logic (just in case we need it here, but we do full cleanup later)
                pass

        # Cleanup
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
