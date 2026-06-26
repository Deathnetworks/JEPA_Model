import os
import csv
import time
import math
import logging
import argparse
from accelerate import Accelerator
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import bitsandbytes as bnb

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_architecture import Mamba2LatentLoop8B, MambaJEPAEngine, DualStageLatentDecoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TripartiteLoss(nn.Module):
    def __init__(self, max_loops=4):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.max_loops = max_loops
        self.lambda_route = 0.01

    def forward(self, logits, qwen_tokens, student_concept, target_concept, global_steps, lambda_jepa):
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(batch_size * seq_len, vocab_size)
        qwen_tokens_flat = qwen_tokens.reshape(-1)

        l_ce = self.ce_loss(logits_flat, qwen_tokens_flat)

        l_jepa = 1 - F.cosine_similarity(student_concept, target_concept, dim=-1).mean()

        avg_loops = global_steps.float().mean() if global_steps.dtype != torch.float32 else global_steps.mean()
        if avg_loops > self.max_loops:
            l_route = (avg_loops - self.max_loops) ** 2
        else:
            l_route = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        total_loss = l_ce + (lambda_jepa * l_jepa) + (self.lambda_route * l_route)
        return total_loss, l_ce, l_jepa, l_route

def get_lambda_jepa(step, warmup_steps=1000):
    if step >= warmup_steps:
        return 1.0
    start_val = 0.01
    end_val = 1.0
    progress = step / warmup_steps
    return start_val + progress * (end_val - start_val)

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class JEPADataset(Dataset):
    def __init__(self, data_dir=r"F:\JEPA_Model\data\shards", curriculum_phase="frontier_traces"):
        super().__init__()
        self.data_dir = Path(data_dir.replace("\\\\", "/"))
        self.file_paths = [p for p in self.data_dir.glob("*.pt") if curriculum_phase in p.name]
        if len(self.file_paths) == 0:
            logging.warning(f"No .pt files found in {self.data_dir} for phase {curriculum_phase}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            data_chunk = torch.load(file_path, map_location="cpu", weights_only=False)
            return data_chunk
        except Exception as e:
            logging.warning(f"Failed to load chunk {file_path}: {e}")
            return []

def collate_jepa_chunk(batch):
    flattened = [item for sublist in batch for item in sublist]
    if not flattened:
        return None
    return flattened

def get_dataloader(data_dir=r"F:\JEPA_Model\data\shards", batch_size=1, num_workers=0, curriculum_phase="frontier_traces"):
    dataset = JEPADataset(data_dir, curriculum_phase=curriculum_phase)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_jepa_chunk)


def train_loop(
    epochs=10,
    mini_batch_size=1,
    learning_rate=3e-4,
    data_dir=r"F:\JEPA_Model\data\shards",
    curriculum_phase="frontier_traces"
):
    accelerator = Accelerator(gradient_accumulation_steps=16)
    device = accelerator.device

    if device.type == 'cpu':
        logging.warning("Running on CPU, using heavily downgraded hyperparameters to avoid OOM.")
        model = MambaJEPAEngine(d_model=64, num_blocks=2, max_budget=2, d_latent=1024)
        decoder = DualStageLatentDecoder(d_model=64, d_latent=1024)
    else:
        model = MambaJEPAEngine()
        decoder = DualStageLatentDecoder()

    if device.type == "xpu":
        torch._inductor.config.freezing = True
        torch._inductor.config.max_autotune = True
        torch._inductor.config.coordinate_descent_tuning = True
        model = torch.compile(model, backend="inductor")
        decoder = torch.compile(decoder, backend="inductor")

    model = model.to(device)
    decoder = decoder.to(device)


    if os.path.exists("jepa_engine.pth"):
        try:
            model.load_state_dict(torch.load("jepa_engine.pth", map_location=device, weights_only=True), strict=False)
            logging.info("Successfully loaded pre-existing weights for jepa_engine.pth.")
        except Exception as e:
            logging.warning(f"Failed to load jepa_engine.pth: {e}")

    if os.path.exists("latent_decoder.pth"):
        try:
            decoder.load_state_dict(torch.load("latent_decoder.pth", map_location=device, weights_only=True), strict=False)
            logging.info("Successfully loaded pre-existing weights for latent_decoder.pth.")
        except Exception as e:
            logging.warning(f"Failed to load latent_decoder.pth: {e}")

    model.train()
    decoder.train()


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

    criterion = TripartiteLoss(max_loops=4)

    dataloader = get_dataloader(data_dir=data_dir, batch_size=1, curriculum_phase=curriculum_phase)

    total_steps = len(dataloader) * epochs
    warmup_steps = int(0.05 * total_steps)
    lr_scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    model, decoder, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, decoder, optimizer, dataloader, lr_scheduler
    )

    checkpoint_dir = f"checkpoint_{curriculum_phase}"
    starting_epoch = 0
    starting_batch = 0

    metadata_path = os.path.join(checkpoint_dir, "training_state.txt")
    if os.path.exists(checkpoint_dir):
        logging.info(f"Resuming from checkpoint {checkpoint_dir}")
        accelerator.load_state(checkpoint_dir)
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                parts = f.read().split(',')
                starting_epoch = int(parts[0])
                starting_batch = int(parts[1])
            logging.info(f"Resuming at Epoch {starting_epoch}, Batch {starting_batch}")

    csv_filename = "training_trace.csv"
    file_exists = os.path.isfile(csv_filename)
    if accelerator.is_main_process:
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Epoch", "ChunkIdx", "MB", "CE_Loss", "JEPA_Loss", "Route_Loss", "Total_Loss"])

    chunk_size = 4096
    accumulation_steps = 16
    optimizer.zero_grad()

    global_mb_step = 0
    start_time = time.time()

    for epoch in range(starting_epoch, epochs):
        if epoch == starting_epoch and starting_batch > 0:
            active_dataloader = accelerator.skip_first_batches(dataloader, starting_batch)
        else:
            active_dataloader = dataloader

        for chunk_idx, flattened_chunk in enumerate(active_dataloader):
            if not flattened_chunk:
                continue

            actual_chunk_idx = chunk_idx + (starting_batch if epoch == starting_epoch else 0)
            num_items = len(flattened_chunk)

            for i in range(0, num_items, mini_batch_size):
                mini_batch = flattened_chunk[i:i+mini_batch_size]

                input_tokens_list = [item["input_tokens"] for item in mini_batch]
                qwen_tokens_list = [item.get("qwen_tokens", item["input_tokens"]) for item in mini_batch]
                target_concepts_list = [item["target_concept"] for item in mini_batch]

                padded_input = torch.nn.utils.rnn.pad_sequence(input_tokens_list, batch_first=True, padding_value=0).to(device)
                padded_qwen = torch.nn.utils.rnn.pad_sequence(qwen_tokens_list, batch_first=True, padding_value=0).to(device)
                target_concepts = torch.stack(target_concepts_list).to(device)

                seq_len = padded_input.size(1)
                mamba_state = None

                num_chunks = (seq_len + chunk_size - 1) // chunk_size
                track_loss = 0.0
                track_ce = 0.0
                track_jepa = 0.0
                track_route = 0.0

                for t in range(0, seq_len, chunk_size):
                    c_input = padded_input[:, t:t+chunk_size]
                    c_qwen = padded_qwen[:, t:t+chunk_size] if padded_qwen.size(1) > 1 else padded_qwen

                    with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
                        student_concept, global_steps, mamba_state = model(c_input, mamba_state=mamba_state)
                        logits = decoder(student_concept)

                        min_len = min(logits.size(1), c_qwen.size(1))
                        logits_aligned = logits[:, :min_len, :]
                        c_qwen_aligned = c_qwen[:, :min_len]

                        completed_opt_steps = lr_scheduler.last_epoch
                        lambda_jepa = get_lambda_jepa(completed_opt_steps, warmup_steps=1000)

                        loss, l_ce, l_jepa, l_route = criterion(
                            logits_aligned, c_qwen_aligned, student_concept, target_concepts, global_steps, lambda_jepa
                        )

                        loss_scaled = loss / (accumulation_steps * num_chunks)

                    accelerator.backward(loss_scaled)

                    track_loss += loss.detach().item()
                    track_ce += l_ce.detach().item()
                    track_jepa += l_jepa.detach().item()
                    track_route += l_route.detach().item()

                    if mamba_state is not None:
                        mamba_state = mamba_state.detach()

                global_mb_step += 1

                avg_loss = track_loss / num_chunks
                avg_ce = track_ce / num_chunks
                avg_jepa = track_jepa / num_chunks
                avg_route = track_route / num_chunks

                if global_mb_step % accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    if hasattr(torch.xpu, 'empty_cache'):
                        torch.xpu.empty_cache()

                if accelerator.is_main_process and global_mb_step % 10 == 0:
                    elapsed = time.time() - start_time
                    it_per_sec = 10 / elapsed if elapsed > 0 else 0
                    logging.info(
                        f"Epoch {epoch+1}/{epochs} | Chunk {actual_chunk_idx+1} | MB {global_mb_step} | "
                        f"Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} | JEPA: {avg_jepa:.4f} | Route: {avg_route:.4f} | "
                        f"Speed: {it_per_sec:.2f} it/s"
                    )
                    start_time = time.time()
                    with open(csv_filename, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch+1, actual_chunk_idx+1, global_mb_step, avg_ce, avg_jepa, avg_route, avg_loss])
        if global_mb_step % accumulation_steps != 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(torch.xpu, 'empty_cache'):
                torch.xpu.empty_cache()



        accelerator.save_state(checkpoint_dir)
        if accelerator.is_main_process:
            with open(metadata_path, 'w') as f:
                f.write(f"{epoch+1},0")
            logging.info(f"Checkpoint saved to {checkpoint_dir}")

    if accelerator.is_main_process:
        torch.save(model.state_dict(), "jepa_engine.pth")
        torch.save(decoder.state_dict(), "latent_decoder.pth")
        logging.info("Models saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum_phase", type=str, default="frontier_traces")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    train_loop(epochs=args.epochs, curriculum_phase=args.curriculum_phase)
