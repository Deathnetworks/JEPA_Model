import os
import csv
import time
import logging
import argparse
from accelerate import Accelerator
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

from .model_architecture import Mamba2LatentLoop8B, MambaJEPAEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            # We load the chunk, which is a list of dicts.
            # However, since this returns the whole chunk (up to 1000 items),
            # we want the DataLoader to iterate over individual items.
            # Standard PyTorch Dataset isn't perfectly mapped for this if we return the whole list.
            # But we can use `collate_fn` to flatten the list of lists.
            data_chunk = torch.load(file_path, map_location="cpu", weights_only=False)
            return data_chunk
        except Exception as e:
            logging.warning(f"Failed to load chunk {file_path}: {e}")
            return []

def collate_jepa_chunk(batch):
    # `batch` is a list of `data_chunk` lists from the Dataset.
    # Usually batch_size=1 from DataLoader if we want to yield chunks.
    # We will flatten the chunks and take a random subset or process sequentially.
    # To keep memory bounded, we just return the flattened items up to a batch size,
    # but the simplest way is to yield the flattened list and let the training loop handle batching
    # OR create a custom IterableDataset.

    # Let's flatten everything in the current Dataset batch
    flattened = [item for sublist in batch for item in sublist]
    if not flattened:
        return None, None

    # We will just pad the flattened list together into one giant batch,
    # or the user specifies batch_size=1 to load 1 chunk (1000 items) and then
    # in the train loop we process the chunk in mini-batches.

    return flattened

def get_dataloader(data_dir=r"F:\JEPA_Model\data\shards", batch_size=1, num_workers=0, curriculum_phase="frontier_traces"):
    # batch_size=1 means we load 1 chunk (e.g. 1000 items) per iteration
    dataset = JEPADataset(data_dir, curriculum_phase=curriculum_phase)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_jepa_chunk)


def train_loop(
    epochs=10,
    mini_batch_size=4,
    accumulation_steps=4,
    gamma=0.001,
    learning_rate=1e-4,
    data_dir=r"F:\JEPA_Model\data\shards",
    curriculum_phase="frontier_traces"
):
    accelerator = Accelerator()
    device = accelerator.device
    logging.info(f"Targeting compute via accelerate device: {device}")

    # Initialize Engine
    model = MambaJEPAEngine()
    model.train()

    # Optimized optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Loss functions
    mse_criterion = nn.MSELoss()

    # DataLoader (loads full chunks)
    # We use shuffle=False because we process chunk by chunk
    dataloader = get_dataloader(data_dir=data_dir, batch_size=1, curriculum_phase=curriculum_phase)

    # Accelerate prepare
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    checkpoint_dir = "checkpoint_latest"
    starting_epoch = 0
    starting_batch = 0

    # Custom metadata file for epoch/batch tracking
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


    # Setup logging CSV
    csv_filename = "training_trace.csv"
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Epoch", "ChunkIdx", "MiniBatch", "Latent_Loss", "Efficiency_Loss", "Total_Loss", "Avg_Routing_Steps"])

    optimizer.zero_grad()

    for epoch in range(starting_epoch, epochs):

        # If resuming in the middle of an epoch, skip the first `starting_batch` batches
        if epoch == starting_epoch and starting_batch > 0:
            active_dataloader = accelerator.skip_first_batches(dataloader, starting_batch)
        else:
            active_dataloader = dataloader

        for chunk_idx, flattened_chunk in enumerate(active_dataloader):
            if not flattened_chunk:
                continue

            actual_chunk_idx = chunk_idx + (starting_batch if epoch == starting_epoch else 0)

            # Process the chunk in mini-batches
            num_items = len(flattened_chunk)

            for i in range(0, num_items, mini_batch_size):
                mini_batch = flattened_chunk[i:i+mini_batch_size]

                # Pad input_tokens
                input_tokens_list = [item["input_tokens"] for item in mini_batch]
                target_concepts_list = [item["target_concept"] for item in mini_batch]

                # Dynamic padding using pad_sequence
                # Pad token id was 0 or eos from qwen, let's use 0
                padded_input_tokens = torch.nn.utils.rnn.pad_sequence(input_tokens_list, batch_first=True, padding_value=0).to(device)
                target_concepts = torch.stack(target_concepts_list).to(device)

                mamba_state = None

                # Forward pass
                student_concept, global_steps, mamba_state = model(padded_input_tokens, mamba_state=mamba_state)

                # Latent Alignment Loss: MSE(H_final, Y_target) + (1.0 - CosineSimilarity(H_final, Y_target))
                mse_loss = mse_criterion(student_concept, target_concepts)
                cos_sim = F.cosine_similarity(student_concept, target_concepts, dim=-1).mean()
                alignment_loss = mse_loss + (1.0 - cos_sim)

                # Efficiency Regularization: gamma * average routing steps
                # global_steps shape: [Batch, Seq_Len, 1]
                avg_routing_steps = global_steps.float().mean() if global_steps.dtype != torch.float32 else global_steps.mean()
                efficiency_loss = gamma * avg_routing_steps

                # Total loss
                loss = alignment_loss + efficiency_loss
                loss_scaled = loss / accumulation_steps

                # Backward pass
                accelerator.backward(loss_scaled)

                if mamba_state is not None:
                    mamba_state = [s.detach() if s is not None else None for s in mamba_state]

                # Gradient Accumulation
                if ((i // mini_batch_size) + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    # Clear VRAM cache hook
                    if hasattr(torch.xpu, 'empty_cache'):
                        torch.xpu.empty_cache()

                # Print real-time metrics
                if (i // mini_batch_size) % 10 == 0:
                    print(
                        f"Epoch: {epoch+1}/{epochs} | "
                        f"Chunk: {actual_chunk_idx+1} | MB: {(i // mini_batch_size) + 1}/{(num_items // mini_batch_size) + 1} | "
                        f"Align Loss: {alignment_loss.item():.4f} | "
                        f"Eff Loss: {efficiency_loss.item():.4f} | "
                        f"Total Loss: {loss.item():.4f} | "
                        f"Routing Steps: {avg_routing_steps.item():.2f}"
                    )

                    # Log to CSV
                    with open(csv_filename, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            epoch + 1,
                            actual_chunk_idx + 1,
                            (i // mini_batch_size) + 1,
                            alignment_loss.item(),
                            efficiency_loss.item(),
                            loss.item(),
                            avg_routing_steps.item()
                        ])

        # Save checkpoint at end of epoch
        accelerator.save_state(checkpoint_dir)
        with open(metadata_path, 'w') as f:
            f.write(f"{epoch+1},0")
        logging.info(f"Checkpoint saved to {checkpoint_dir}")

    # Save model checkpoint
    torch.save(model.state_dict(), "jepa_engine.pth")
    logging.info("Model saved to jepa_engine.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum_phase", type=str, default="frontier_traces", help="Curriculum phase to filter dataset")
    args = parser.parse_args()
    train_loop(curriculum_phase=args.curriculum_phase)
