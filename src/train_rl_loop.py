import argparse
import logging
import os
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
import bitsandbytes as bnb
import torch.optim as optim

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

from src.model_architecture import MambaJEPAEngine, ClosedLoopLatentDecoder
from src.GRPOLearningEngine import GRPOLearningEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def stream_rl_prompts(dataset_name="bigcode/humanevalpack", start_idx=0):
    """
    Streams multi-language RL prompts across all 6 languages.
    """
    from datasets import interleave_datasets

    languages = ['python', 'cpp', 'js', 'java', 'go', 'rust']
    datasets = []

    for lang in languages:
        try:
            ds = load_dataset(dataset_name, lang, split="train", streaming=True)
            # Tag each language so the engine knows how to compile/run the test harness
            ds = ds.map(lambda x: {**x, '_lang': lang})
            datasets.append(ds)
        except Exception as e:
            logging.warning(f"Could not load language {lang}: {e}")

    if not datasets:
        return

    dataset = interleave_datasets(datasets)
    dataset = dataset.skip(start_idx)

    for row in dataset:
        yield row['instruction'], row['test'], row['_lang']

def train_rl_loop(
    epochs: int = 1,
    dataset_name: str = "bigcode/humanevalpack",
    group_size: int = 4,
    learning_rate: float = 1e-5,
    engine_path: str = "jepa_engine.pth",
    decoder_path: str = "latent_decoder.pth",
    tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct",
):
    from src.runtime import get_device, empty_cache, get_autocast_kwargs
    device = get_device()

    logging.info(f"Loading tokenizer {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    vocab_size = ((len(tokenizer) + 63) // 64) * 64

    logging.info("Instantiating MambaJEPAEngine and ClosedLoopLatentDecoder...")
    if device.type == 'cpu':
        logging.warning("Running on CPU, using heavily downgraded hyperparameters to avoid OOM.")
        model = MambaJEPAEngine(vocab_size=vocab_size, d_model=64, num_blocks=2, max_budget=2, d_latent=1024)
        decoder = ClosedLoopLatentDecoder(vocab_size=vocab_size, d_model=64, d_latent=1024)
    else:
        model = MambaJEPAEngine(vocab_size=vocab_size)
        decoder = ClosedLoopLatentDecoder(vocab_size=vocab_size)



    logging.info(f"Loading weights from {engine_path} and {decoder_path}...")
    try:
        from src.checkpoint import load_ckpt; load_ckpt(model, engine_path, device)
        logging.info(f"Successfully loaded {engine_path}")
    except FileNotFoundError:
        logging.warning(f"Could not find {engine_path}, initializing with random weights.")

    try:
        load_ckpt(decoder, decoder_path, device)
        logging.info(f"Successfully loaded {decoder_path}")
    except FileNotFoundError:
        logging.warning(f"Could not find {decoder_path}, initializing with random weights.")
    if device.type == "xpu":
        torch._inductor.config.freezing = True
        torch._inductor.config.max_autotune = True
        torch._inductor.config.coordinate_descent_tuning = True
        model = torch.compile(model, backend="inductor")
        decoder = torch.compile(decoder, backend="inductor")

    model = model.to(device)
    decoder = decoder.to(device)

    import bitsandbytes as bnb
    from galore_torch import GaLoreAdamW8bit

    galore_params = []
    non_galore_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if isinstance(model.get_submodule(n.rsplit('.', 1)[0] if '.' in n else ''), nn.Linear):
            galore_params.append(p)
        else:
            non_galore_params.append(p)

    for n, p in decoder.named_parameters():
        if not p.requires_grad:
            continue
        if isinstance(decoder.get_submodule(n.rsplit('.', 1)[0] if '.' in n else ''), nn.Linear):
            galore_params.append(p)
        else:
            non_galore_params.append(p)

    param_groups = [
        {'params': non_galore_params},
        {'params': galore_params, 'rank': 128, 'update_proj_gap': 200, 'scale': 0.25, 'proj_type': 'std'}
    ]

    optimizer = bnb.optim.PagedAdamW8bit(
        param_groups,

        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    grpo_engine = GRPOLearningEngine(model, decoder, tokenizer)

    checkpoint_dir = f"checkpoint_rl"
    starting_epoch = 0
    starting_step = 0

    metadata_path = os.path.join(checkpoint_dir, "rl_training_state.txt")
    if os.path.exists(checkpoint_dir):
        logging.info(f"Resuming from checkpoint {checkpoint_dir}")

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                parts = f.read().split(',')
                starting_epoch = int(parts[0])
                starting_step = int(parts[1])
            logging.info(f"Resuming at Epoch {starting_epoch}, Step {starting_step}")

    for epoch in range(starting_epoch, epochs):
        prompt_generator = stream_rl_prompts(dataset_name=dataset_name, start_idx=starting_step if epoch == starting_epoch else 0)

        step_idx = starting_step if epoch == starting_epoch else 0

        optimizer.zero_grad(set_to_none=True)
        for prompt, test_harness, lang in prompt_generator:
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                with torch.autocast(**get_autocast_kwargs()):
                    loss = grpo_engine.train_grpo_step(inputs["input_ids"], test_harness, lang, optimizer, group_size=group_size)

                loss_scaled = loss / 16
                loss_scaled.backward()

                if step_idx % 16 == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                empty_cache()

                logging.info(f"Epoch {epoch+1}/{epochs} | Step {step_idx+1} | Loss: {loss.item():.4f}")

                step_idx += 1

                if step_idx % 100 == 0:

                     if True:
                         with open(metadata_path, 'w') as f:
                             f.write(f"{epoch},{step_idx}")
                         logging.info(f"Checkpoint saved to {checkpoint_dir}")
            except Exception as e:
                logging.error(f"Error during RL step {step_idx}: {e}")

    if True:
        raw_model = getattr(model, "_orig_mod", model)
        from src.checkpoint import save_ckpt; save_ckpt(raw_model, "jepa_engine_rl.pth")
        raw_decoder = getattr(decoder, "_orig_mod", decoder)
        save_ckpt(raw_decoder, "latent_decoder_rl.pth")
        logging.info("Models saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="bigcode/humanevalpack")
    parser.add_argument("--group_size", type=int, default=4)
    args = parser.parse_args()
    train_rl_loop(epochs=args.epochs, dataset_name=args.dataset, group_size=args.group_size)
