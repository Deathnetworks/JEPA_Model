import os
import csv
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import bitsandbytes as bnb

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Switch import from the old feed-forward stage to the cross-attention loop architecture
from src.model_architecture import ClosedLoopLatentDecoder

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

    if device.type == 'cpu':
        logging.warning("Running on CPU, using heavily downgraded hyperparameters to avoid OOM.")
        model = ClosedLoopLatentDecoder(d_model=64, d_latent=1024).to(device)
    else:
        model = ClosedLoopLatentDecoder().to(device)

    model.train()

    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    dataloader = get_decoder_dataloader(data_dir=data_dir, batch_size=1)

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

                max_len = model.max_seq_len
                # Slice target sequence to max_len + 1 to safely capture autoregressive shifted offsets
                qwen_tokens_list_sliced = [tokens[:max_len + 1] for tokens in qwen_tokens_list]

                qwen_tokens = torch.nn.utils.rnn.pad_sequence(qwen_tokens_list_sliced, batch_first=True, padding_value=0).to(device).long()

                seq_len_current = qwen_tokens.shape[1]
                if seq_len_current < max_len + 1:
                    padding = torch.zeros((qwen_tokens.shape[0], (max_len + 1) - seq_len_current), dtype=qwen_tokens.dtype, device=device)
                    qwen_tokens = torch.cat([qwen_tokens, padding], dim=1)

                # Teacher Forcing: Shift targets to split tokens into inputs and expected labels
                decoder_input = qwen_tokens[:, :-1]   # Tokens [0, 1, ..., max_len - 1]
                decoder_target = qwen_tokens[:, 1:]    # Tokens [1, 2, ..., max_len]

                optimizer.zero_grad()

                with torch.autocast(device_type="xpu" if device.type == "xpu" else "cpu", dtype=torch.bfloat16 if device.type == "xpu" else torch.float32):
                    # Forward pass conditions token generation on both history and the concept vector
                    logits = model(decoder_input, target_concept)

                    batch_size, seq_len, vocab_size = logits.shape
                    logits_flat = logits.view(batch_size * seq_len, vocab_size)
                    qwen_tokens_flat = decoder_target.view(-1)

                    loss = criterion(logits_flat, qwen_tokens_flat)

                loss.backward()
                optimizer.step()

                if hasattr(torch, 'xpu') and hasattr(torch.xpu, 'empty_cache'):
                    torch.xpu.empty_cache()

                if (i // mini_batch_size) % 10 == 0:
                    print(
                        f"Epoch: {epoch+1}/{epochs} | "
                        f"Chunk: {chunk_idx+1} | MB: {(i // mini_batch_size) + 1}/{(num_items // mini_batch_size) + 1} | "
                        f"CrossEntropyLoss: {loss.item():.4f}"
                    )

                    with open(csv_filename, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            epoch + 1,
                            chunk_idx + 1,
                            (i // mini_batch_size) + 1,
                            loss.item()
                        ])

    torch.save(model.state_dict(), "latent_decoder.pth")
    logging.info("Model saved to latent_decoder.pth")

if __name__ == "__main__":
    train_decoder_loop()