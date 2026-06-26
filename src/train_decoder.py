import os
import csv
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

from .model_architecture import DualStageLatentDecoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DecoderDataset(Dataset):
    def __init__(self, data_dir=r"F:\JEPA_Model\data\shards"):
        super().__init__()
        self.data_dir = Path(data_dir.replace("\\\\", "/"))
        self.file_paths = list(self.data_dir.glob("*.pt"))
        if len(self.file_paths) == 0:
            logging.warning(f"No .pt files found in {self.data_dir}")

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

def collate_decoder_chunk(batch):
    flattened = [item for sublist in batch for item in sublist]
    if not flattened:
        return None
    return flattened

def get_decoder_dataloader(data_dir=r"F:\JEPA_Model\data\shards", batch_size=1, num_workers=0):
    dataset = DecoderDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_decoder_chunk)

def train_decoder_loop(
    epochs=10,
    mini_batch_size=4,
    learning_rate=1e-4,
    data_dir=r"F:\JEPA_Model\data\shards"
):
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    logging.info(f"Targeting native compute via device: {device}")

    # Initialize Engine
    # Max seq len 256 might not be enough for full traces, but let's stick to the architecture default unless we modify model_architecture.py
    # We will dynamically pad up to 256 or slice
    model = DualStageLatentDecoder().to(device)
    model.train()

    # Optimized optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Loss functions
    criterion = nn.CrossEntropyLoss(ignore_index=0) # 0 is assumed to be padding

    # DataLoader
    dataloader = get_decoder_dataloader(data_dir=data_dir, batch_size=1)

    # Setup logging CSV
    csv_filename = "decoder_training_trace.csv"
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Epoch", "ChunkIdx", "MiniBatch", "CrossEntropyLoss"])

    for epoch in range(epochs):
        for chunk_idx, flattened_chunk in enumerate(dataloader):
            if not flattened_chunk:
                continue

            num_items = len(flattened_chunk)

            for i in range(0, num_items, mini_batch_size):
                mini_batch = flattened_chunk[i:i+mini_batch_size]

                target_concepts_list = [item["target_concept"] for item in mini_batch]
                qwen_tokens_list = [item["qwen_tokens"] for item in mini_batch]

                target_concept = torch.stack(target_concepts_list).to(device)

                # Pad sequences, but also slice to max_seq_len supported by decoder (e.g. 256 from model_architecture)
                max_len = model.max_seq_len
                qwen_tokens_list_sliced = [tokens[:max_len] for tokens in qwen_tokens_list]

                # Pad to uniform shape
                qwen_tokens = torch.nn.utils.rnn.pad_sequence(qwen_tokens_list_sliced, batch_first=True, padding_value=0).to(device).long()

                # Further pad to exact max_len if needed by architecture (DualStageLatentDecoder expects it or handles dynamically?)
                # Wait, DualStageLatentDecoder stage1 outputs `[Batch, max_seq_len, d_model]` unconditionally.
                # So we must pad/slice exactly to `max_seq_len`.
                seq_len_current = qwen_tokens.shape[1]
                if seq_len_current < max_len:
                    padding = torch.zeros((qwen_tokens.shape[0], max_len - seq_len_current), dtype=qwen_tokens.dtype, device=device)
                    qwen_tokens = torch.cat([qwen_tokens, padding], dim=1)

                optimizer.zero_grad()

                # Forward pass
                # logits: [Batch, 256, 151643]
                logits = model(target_concept)

                batch_size, seq_len, vocab_size = logits.shape
                logits_flat = logits.view(batch_size * seq_len, vocab_size)
                qwen_tokens_flat = qwen_tokens.view(-1)

                loss = criterion(logits_flat, qwen_tokens_flat)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Clear VRAM cache hook
                if hasattr(torch.xpu, 'empty_cache'):
                    torch.xpu.empty_cache()

                # Print real-time metrics
                if (i // mini_batch_size) % 10 == 0:
                    print(
                        f"Epoch: {epoch+1}/{epochs} | "
                        f"Chunk: {chunk_idx+1} | MB: {(i // mini_batch_size) + 1}/{(num_items // mini_batch_size) + 1} | "
                        f"CrossEntropyLoss: {loss.item():.4f}"
                    )

                    # Log to CSV
                    with open(csv_filename, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            epoch + 1,
                            chunk_idx + 1,
                            (i // mini_batch_size) + 1,
                            loss.item()
                        ])

    # Save model checkpoint
    torch.save(model.state_dict(), "latent_decoder.pth")
    logging.info("Model saved to latent_decoder.pth")

if __name__ == "__main__":
    train_decoder_loop()
