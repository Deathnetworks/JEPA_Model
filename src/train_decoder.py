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
    def __init__(self, data_dir="F:\\JEPA_Model\\distilled_data"):
        super().__init__()
        self.data_dir = Path(data_dir.replace("\\", "/"))
        self.file_paths = list(self.data_dir.glob("*.pt"))
        if len(self.file_paths) == 0:
            logging.warning(f"No .pt files found in {self.data_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = torch.load(file_path, map_location="cpu", weights_only=False)

        target_concept = data['target_concept']
        qwen_tokens = data['qwen_tokens']

        if target_concept.dim() > 1 and target_concept.shape[0] == 1:
            target_concept = target_concept.squeeze(0)
        if qwen_tokens.dim() > 1 and qwen_tokens.shape[0] == 1:
            qwen_tokens = qwen_tokens.squeeze(0)

        return target_concept, qwen_tokens

def get_decoder_dataloader(data_dir="F:\\JEPA_Model\\distilled_data", batch_size=4, num_workers=0):
    dataset = DecoderDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

def train_decoder_loop(
    epochs=10,
    batch_size=4,
    learning_rate=1e-4,
    data_dir="F:\\JEPA_Model\\distilled_data"
):
    device = torch.device("xpu")
    logging.info(f"Targeting native Intel GPU compute via device: {device}")

    # Initialize Engine
    model = DualStageLatentDecoder().to(device)
    model.train()

    # Optimized optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Loss functions
    # Using cross entropy for logits [Batch, Seq, Vocab] and targets [Batch, Seq]
    criterion = nn.CrossEntropyLoss()

    # DataLoader
    dataloader = get_decoder_dataloader(data_dir=data_dir, batch_size=batch_size)

    # Setup logging CSV
    csv_filename = "decoder_training_trace.csv"
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Epoch", "Batch", "CrossEntropyLoss"])

    for epoch in range(epochs):
        for batch_idx, (target_concept, qwen_tokens) in enumerate(dataloader):
            target_concept = target_concept.to(device)
            # qwen_tokens: [Batch, 256] long
            qwen_tokens = qwen_tokens.to(device).long()

            optimizer.zero_grad()

            # Forward pass
            # target_concept: [Batch, 1024]
            # logits: [Batch, 256, 151643]
            logits = model(target_concept)

            # Reshape for CrossEntropyLoss which expects (Batch * Seq, Vocab)
            # and target (Batch * Seq)
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
            if batch_idx % 1 == 0:
                print(
                    f"Epoch: {epoch+1}/{epochs} | "
                    f"Batch: {batch_idx+1} | "
                    f"CrossEntropyLoss: {loss.item():.4f}"
                )

                # Log to CSV
                with open(csv_filename, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch + 1,
                        batch_idx + 1,
                        loss.item()
                    ])

    # Save model checkpoint
    torch.save(model.state_dict(), "latent_decoder.pth")
    logging.info("Model saved to latent_decoder.pth")

if __name__ == "__main__":
    train_decoder_loop()
