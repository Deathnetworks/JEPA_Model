from src.runtime import get_device, empty_cache, autocast_ctx, get_vocab_size
import os
import csv
import time
import math
import logging
import argparse
import torch.optim as optim
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
from src.model_architecture import Mamba2LatentLoop8B, MambaJEPAEngine, ClosedLoopLatentDecoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TripartiteLoss(nn.Module):
    def __init__(self, max_loops=4, lambda_spectral=0.1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.max_loops = max_loops
        self.lambda_route = 0.01
        self.lambda_spectral = lambda_spectral # NEW: Coefficient for Spectral Uniformity

    def forward(self, logits, qwen_tokens, student_concept, target_concept, global_steps, lambda_jepa):
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(batch_size * seq_len, vocab_size)
        qwen_tokens_flat = qwen_tokens.reshape(-1)

        # 1. Cross-Entropy Loss (Text Generation)
        l_ce = self.ce_loss(logits_flat, qwen_tokens_flat)

        # 2. JEPA Alignment Loss (Positive Pairs)
        # --- NEW: Hierarchical Sliced Alignment ---
        # Anchor only the foundational 1024D syntax space to the static BGE-M3 vectors
        student_micro = student_concept[:, :1024]
        l_jepa_align = 1 - F.cosine_similarity(student_micro, target_concept, dim=-1).mean()
        l_jepa = l_jepa_align

        # 3. Routing Penalty Loss (Compute Budget)
        avg_loops = global_steps.float().mean() if global_steps.dtype != torch.float32 else global_steps.mean()
        if avg_loops > self.max_loops:
            l_route = (avg_loops - self.max_loops) ** 2
        else:
            l_route = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        # --- EXISTING ELEGANCE: Native Sparsity Gating ---
        # Because we concatenated the gated Macro space onto the student_concept tensor, 
        # this L2 Energy Contraction inherently acts as a Sparsity Penalty. 
        # The model is forced to keep the Macro gate at 0.0 to save energy UNLESS 
        # the Cross-Entropy loss strictly demands macro-architectural resolution!
        # Penalizing the micro-space destroys the BGE-M3 baseline knowledge.
        macro_concept = student_concept[:, 1024:]
        l_energy_contraction = torch.norm(macro_concept, p=2, dim=-1).mean() * 0.001

        total_loss = l_ce + (lambda_jepa * l_jepa) + (self.lambda_route * l_route) + l_energy_contraction
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

def get_dataloader(data_dir=r"F:\JEPA_Model\data\shards", batch_size=1, num_workers=2, curriculum_phase="frontier_traces"):
    dataset = JEPADataset(data_dir, curriculum_phase=curriculum_phase)
    # --- NEW: Asynchronous Data Prefetching (Thanks Emre!) ---
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=collate_jepa_chunk,
        pin_memory=True,       # Stages tensors in page-locked RAM for instant PCIe transfer
        prefetch_factor=2      # CPU prepares the next 2 batches in the background
    )


def train_loop(
    epochs=10,
    mini_batch_size=1,
    learning_rate=3e-4,
    data_dir=r"F:\JEPA_Model\data\shards",
    curriculum_phase="frontier_traces"
):
    # --- NEW: DeepSpeed ZeRO-2 CPU Offload ---
    



    from src.checkpoint import save_ckpt, load_ckpt
    device = get_device()
    vocab_size = get_vocab_size()

    if device.type == 'cpu':
        logging.warning("Running on CPU, using heavily downgraded hyperparameters to avoid OOM.")
        model = MambaJEPAEngine(vocab_size=vocab_size, d_model=64, num_blocks=2, max_budget=2, d_latent=5120)
        decoder = ClosedLoopLatentDecoder(vocab_size=vocab_size, d_latent=5120, d_model=64)
    else:
        model = MambaJEPAEngine(vocab_size=vocab_size)
        decoder = ClosedLoopLatentDecoder(vocab_size=vocab_size)




    if os.path.exists("jepa_engine.pth"):
        try:
            load_ckpt(model, "jepa_engine.pth", device)
            logging.info("Successfully loaded pre-existing weights for jepa_engine.pth.")
        except Exception as e:
            logging.warning(f"Failed to load jepa_engine.pth: {e}")

    if os.path.exists("latent_decoder.pth"):
        try:
            load_ckpt(decoder, "latent_decoder.pth", device)
            logging.info("Successfully loaded pre-existing weights for latent_decoder.pth.")
        except Exception as e:
            logging.warning(f"Failed to load latent_decoder.pth: {e}")
    if device.type == "xpu":
        torch._inductor.config.freezing = True
        torch._inductor.config.max_autotune = True
        torch._inductor.config.coordinate_descent_tuning = True
        model = torch.compile(model, backend="inductor")
        decoder = torch.compile(decoder, backend="inductor")

    model = model.to(device)
    decoder = decoder.to(device)

    model.train()
    decoder.train()


    import bitsandbytes as bnb
    from galore_torch import GaLoreAdamW8bit

    galore_params = []
    plain_params = []
    exclude_tokens = ['bias', 'norm', 'A_log_spectral', 'discrete_embeddings', 'expansion_gate', 'foresight_head', 'embedding']
    for n, p in list(model.named_parameters()) + list(decoder.named_parameters()):
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and not any(t in n for t in exclude_tokens) and p.shape[0] >= 256 and p.shape[1] >= 256:
            galore_params.append(p)
        else:
            plain_params.append(p)

    param_groups = [
        {'params': plain_params},
        {'params': galore_params, 'rank': 128, 'update_proj_gap': 200, 'scale': 0.25, 'proj_type': 'std'}
    ]
    if device.type == 'cpu':
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
    else:
        optimizer = bnb.optim.PagedAdamW8bit(
            param_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )

    criterion = TripartiteLoss(max_loops=4)

    dataloader = get_dataloader(data_dir=data_dir, batch_size=1, curriculum_phase=curriculum_phase)

    total_steps = len(dataloader) * epochs
    warmup_steps = int(0.05 * total_steps)
    lr_scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)



    checkpoint_dir = f"checkpoint_{curriculum_phase}"
    starting_epoch = 0
    starting_batch = 0

    metadata_path = os.path.join(checkpoint_dir, "training_state.txt")
    if os.path.exists(checkpoint_dir):
        logging.info(f"Resuming from checkpoint {checkpoint_dir}")

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                parts = f.read().split(',')
                starting_epoch = int(parts[0])
                starting_batch = int(parts[1])
            logging.info(f"Resuming at Epoch {starting_epoch}, Batch {starting_batch}")

    csv_filename = "training_trace.csv"
    file_exists = os.path.isfile(csv_filename)
    if True:
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Epoch", "ChunkIdx", "MB", "CE_Loss", "JEPA_Loss", "Route_Loss", "Total_Loss"])

    chunk_size = 4096
    accumulation_steps = 16
    global_mb_step = 1
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    for epoch in range(starting_epoch, epochs):
        if epoch == starting_epoch and starting_batch > 0:
            active_dataloader = dataloader
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
                prev_student_concept = None # --- NEW: Delta-JEPA state tracking ---

                num_chunks = (seq_len + chunk_size - 1) // chunk_size
                track_loss, track_ce, track_jepa, track_route = 0.0, 0.0, 0.0, 0.0

                for t in range(0, seq_len, chunk_size):
                    c_input = padded_input[:, t:t+chunk_size]
                    c_qwen = padded_qwen[:, t:t+chunk_size] if padded_qwen.size(1) > 1 else padded_qwen

                    with autocast_ctx(device):
                        student_concept, global_steps, mamba_state = model(c_input, mamba_state=mamba_state)
                        
                        # Apply teacher forcing shift logic to train the cross-attended decoder
                        if c_qwen.size(1) > 1:
                            decoder_input = c_qwen[:, :-1]
                            decoder_target = c_qwen[:, 1:]
                            
                            # --- NEW: Pass previous concept for Latent Difference Decoding ---
                            logits = decoder(decoder_input, student_concept, prev_student_concept)
                            
                            min_len = min(logits.size(1), decoder_target.size(1))
                            logits_aligned = logits[:, :min_len, :]
                            c_qwen_aligned = decoder_target[:, :min_len]
                        else:
                            logits = decoder(c_qwen, student_concept)
                            logits_aligned = logits
                            c_qwen_aligned = c_qwen

                        # Cache the current concept for the next sequence chunk
                        prev_student_concept = student_concept.detach()
                        
                        completed_opt_steps = lr_scheduler.last_epoch
                        lambda_jepa = get_lambda_jepa(completed_opt_steps, warmup_steps=1000)

                        loss, l_ce, l_jepa, l_route = criterion(
                            logits_aligned, c_qwen_aligned, student_concept, target_concepts, global_steps, lambda_jepa
                        )

                        loss_scaled = loss / (accumulation_steps * num_chunks)

                    loss_scaled.backward()

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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    empty_cache()

                if global_mb_step % 10 == 0:
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            empty_cache()


        if True:
            os.makedirs(checkpoint_dir, exist_ok=True)
            with open(metadata_path, 'w') as f:
                f.write(f"{epoch+1},0")
            logging.info(f"Checkpoint saved to {checkpoint_dir}")

    if True:
        raw_model = getattr(model, "_orig_mod", model)
        save_ckpt(raw_model, "jepa_engine.pth")
        raw_decoder = getattr(decoder, "_orig_mod", decoder)
        save_ckpt(raw_decoder, "latent_decoder.pth")
        logging.info("Models saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum_phase", type=str, default="frontier_traces")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    train_loop(epochs=args.epochs, curriculum_phase=args.curriculum_phase)