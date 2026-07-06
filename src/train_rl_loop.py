import argparse
import logging
import os
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
import bitsandbytes as bnb
from accelerate import Accelerator

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

from src.model_architecture import MambaJEPAEngine, ClosedLoopLatentDecoder
from src.GRPOLearningEngine import GRPOLearningEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def stream_rl_prompts(dataset_name="ise-uiuc/Magicoder-OSS-Instruct-75K", start_idx=0):
    """
    Streams multi-language RL prompts.
    """
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    dataset = dataset.skip(start_idx)
    for row in dataset:
        yield row['instruction']

def train_rl_loop(
    epochs: int = 1,
    dataset_name: str = "ise-uiuc/Magicoder-OSS-Instruct-75K",
    group_size: int = 4,
    learning_rate: float = 1e-5,
    engine_path: str = "jepa_engine.pth",
    decoder_path: str = "latent_decoder.pth",
    tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct",
):
    accelerator = Accelerator()
    device = setup_device()

    logging.info(f"Loading tokenizer {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    logging.info("Instantiating MambaJEPAEngine and ClosedLoopLatentDecoder...")
    if device.type == 'cpu':
        logging.warning("Running on CPU, using heavily downgraded hyperparameters to avoid OOM.")
        model = MambaJEPAEngine(d_model=64, num_blocks=2, max_budget=2, d_latent=1024)
        decoder = ClosedLoopLatentDecoder(d_model=64, d_latent=1024)
    else:
        model = MambaJEPAEngine()
        decoder = ClosedLoopLatentDecoder()

    if device.type == "xpu":
        torch._inductor.config.freezing = True
        torch._inductor.config.max_autotune = True
        torch._inductor.config.coordinate_descent_tuning = True
        model = torch.compile(model, backend="inductor")
        decoder = torch.compile(decoder, backend="inductor")

    model = model.to(device)
    decoder = decoder.to(device)

    logging.info(f"Loading weights from {engine_path} and {decoder_path}...")
    try:
        model.load_state_dict(torch.load(engine_path, map_location=device, weights_only=True), strict=False)
        logging.info(f"Successfully loaded {engine_path}")
    except FileNotFoundError:
        logging.warning(f"Could not find {engine_path}, initializing with random weights.")

    try:
        decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True), strict=False)
        logging.info(f"Successfully loaded {decoder_path}")
    except FileNotFoundError:
        logging.warning(f"Could not find {decoder_path}, initializing with random weights.")

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
        {"params": [p for n, p in decoder.named_parameters() if p.requires_grad]}
    ]

    optimizer = bnb.optim.AdamW8bit(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    model, decoder, optimizer = accelerator.prepare(model, decoder, optimizer)
    grpo_engine = GRPOLearningEngine(model, decoder, tokenizer)

    checkpoint_dir = f"checkpoint_rl"
    starting_epoch = 0
    starting_step = 0

    metadata_path = os.path.join(checkpoint_dir, "rl_training_state.txt")
    if os.path.exists(checkpoint_dir):
        logging.info(f"Resuming from checkpoint {checkpoint_dir}")
        accelerator.load_state(checkpoint_dir)
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                parts = f.read().split(',')
                starting_epoch = int(parts[0])
                starting_step = int(parts[1])
            logging.info(f"Resuming at Epoch {starting_epoch}, Step {starting_step}")

    for epoch in range(starting_epoch, epochs):
        prompt_generator = stream_rl_prompts(dataset_name=dataset_name, start_idx=starting_step if epoch == starting_epoch else 0)

        step_idx = starting_step if epoch == starting_epoch else 0

        for prompt in prompt_generator:
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                optimizer.zero_grad()

                with torch.autocast(device_type="xpu" if device.type == "xpu" else "cpu", dtype=torch.bfloat16 if device.type == "xpu" else torch.float32):
                    loss = grpo_engine.train_grpo_step(inputs["input_ids"], optimizer, group_size=group_size)

                accelerator.backward(loss)
                optimizer.step()

                if hasattr(torch.xpu, 'empty_cache'):
                    torch.xpu.empty_cache()

                logging.info(f"Epoch {epoch+1}/{epochs} | Step {step_idx+1} | Loss: {loss.item():.4f}")

                step_idx += 1

                if step_idx % 100 == 0:
                     accelerator.save_state(checkpoint_dir)
                     if accelerator.is_main_process:
                         with open(metadata_path, 'w') as f:
                             f.write(f"{epoch},{step_idx}")
                         logging.info(f"Checkpoint saved to {checkpoint_dir}")
            except Exception as e:
                logging.error(f"Error during RL step {step_idx}: {e}")

    if accelerator.is_main_process:
        torch.save(model.state_dict(), "jepa_engine_rl.pth")
        torch.save(decoder.state_dict(), "latent_decoder_rl.pth")
        logging.info("Models saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="ise-uiuc/Magicoder-OSS-Instruct-75K")
    parser.add_argument("--group_size", type=int, default=4)
    args = parser.parse_args()
    train_rl_loop(epochs=args.epochs, dataset_name=args.dataset, group_size=args.group_size)
