import torch
from transformers import AutoTokenizer
import logging

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

from src.model_architecture import MambaJEPAEngine, DualStageLatentDecoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    def generate_code(self, prompt: str):
        logging.info(f"Generating code for prompt: '{prompt}'")

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
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

        logging.info("\n--- Generated Output ---")
        print(decoded_text)
        logging.info("------------------------\n")
        return decoded_text

if __name__ == "__main__":
    pipeline = InferencePipeline()
    prompt = "Write a Rust struct for a Tauri SQLite database connection."
    pipeline.generate_code(prompt)
