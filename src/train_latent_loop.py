import os
import csv
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

from .model_architecture import Mamba2LatentLoop4B, MambaJEPAEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class JEPADataset(Dataset):
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

        input_tokens = data['input_tokens']
        target_concept = data['target_concept']

        if input_tokens.dim() > 1 and input_tokens.shape[0] == 1:
            input_tokens = input_tokens.squeeze(0)
        if target_concept.dim() > 1 and target_concept.shape[0] == 1:
            target_concept = target_concept.squeeze(0)

        return input_tokens, target_concept

def get_dataloader(data_dir="F:\\JEPA_Model\\distilled_data", batch_size=4, num_workers=0):
    dataset = JEPADataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)


def train_loop(
    epochs=10,
    batch_size=4,
    accumulation_steps=4,
    gamma=0.001,
    learning_rate=1e-4,
    data_dir="F:\\JEPA_Model\\distilled_data"
):
    device = torch.device("xpu")
    logging.info(f"Targeting native Intel GPU compute via device: {device}")

    # Initialize Engine
    model = MambaJEPAEngine().to(device)
    model.train()

    # Optimized optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Loss functions
    mse_criterion = nn.MSELoss()

    # DataLoader
    dataloader = get_dataloader(data_dir=data_dir, batch_size=batch_size)

    # Setup logging CSV
    csv_filename = "training_trace.csv"
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Epoch", "Batch", "Latent_Loss", "Efficiency_Loss", "Total_Loss", "Avg_Routing_Steps"])

    optimizer.zero_grad()

    for epoch in range(epochs):
        for batch_idx, (input_tokens, target_concept) in enumerate(dataloader):
            input_tokens = input_tokens.to(device)
            target_concept = target_concept.to(device)

            # Forward pass
            student_concept, global_steps = model(input_tokens)

            # Latent Alignment Loss: MSE(H_final, Y_target) + (1.0 - CosineSimilarity(H_final, Y_target))
            mse_loss = mse_criterion(student_concept, target_concept)
            cos_sim = F.cosine_similarity(student_concept, target_concept, dim=-1).mean()
            alignment_loss = mse_loss + (1.0 - cos_sim)

            # Efficiency Regularization: gamma * average routing steps
            # global_steps shape: [Batch, Seq_Len, 1]
            avg_routing_steps = global_steps.float().mean() if global_steps.dtype != torch.float32 else global_steps.mean()
            efficiency_loss = gamma * avg_routing_steps

            # Total loss
            loss = alignment_loss + efficiency_loss
            loss_scaled = loss / accumulation_steps

            # Backward pass
            loss_scaled.backward()

            # Gradient Accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                # Clear VRAM cache hook
                if hasattr(torch.xpu, 'empty_cache'):
                    torch.xpu.empty_cache()

            # Print real-time metrics
            if batch_idx % 1 == 0:
                print(
                    f"Epoch: {epoch+1}/{epochs} | "
                    f"Batch: {batch_idx+1} | "
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
                        batch_idx + 1,
                        alignment_loss.item(),
                        efficiency_loss.item(),
                        loss.item(),
                        avg_routing_steps.item()
                    ])

if __name__ == "__main__":
    train_loop()
