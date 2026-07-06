import torch
import torch.nn as nn
import os
import logging
from src.model_architecture import MambaJEPAEngine, ClosedLoopLatentDecoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def apply_dynamic_quantization(model):
    """
    Applies PyTorch native INT8 Dynamic Quantization strictly to Linear layers.
    This reduces the weight footprint by 50% for inference while maintaining accuracy.
    """
    logging.info("Applying dynamic INT8 quantization to Linear layers...")
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    return quantized_model

def quantize_pipeline(engine_path="jepa_engine.pth", decoder_path="latent_decoder.pth"):
    device = torch.device("cpu") # Quantization tracing is safest on CPU
    
    # 1. Initialize empty architectures (matched to the new 18GB safe dimensions)
    logging.info("Initializing model architectures...")
    engine = MambaJEPAEngine(d_model=6144, num_blocks=32, d_latent=5120)
    decoder = ClosedLoopLatentDecoder(d_latent=5120, d_model=6144)
    
    # 2. Load the trained BF16/FP32 weights
    logging.info("Loading trained bfloat16 weights...")
    engine.load_state_dict(torch.load(engine_path, map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
    
    # 3. Quantize
    q_engine = apply_dynamic_quantization(engine)
    q_decoder = apply_dynamic_quantization(decoder)
    
    # 4. Save the compressed Q8 models
    q_engine_path = "q8_jepa_engine.pth"
    q_decoder_path = "q8_latent_decoder.pth"
    
    torch.save(q_engine.state_dict(), q_engine_path)
    torch.save(q_decoder.state_dict(), q_decoder_path)
    
    logging.info(f"Quantization Complete!")
    logging.info(f"Saved to: {q_engine_path} and {q_decoder_path}")

if __name__ == "__main__":
    quantize_pipeline()