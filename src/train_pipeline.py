import os
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import load_dataset
import wandb

from src.model.backbone.mamba_engine import MambaJEPAEngine
from src.config import parse_args

def train_stage(config, accelerator, model, dataloader, optimizer, stage_name="pretrain"):
    model.train()
    total_loss = 0

    # Initialize wandb
    if accelerator.is_main_process:
        wandb.init(project="JEPA_Model_v2", name=f"{stage_name}_run", config=config)

    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(model):
            input_ids = batch['input_ids'].to(accelerator.device)
            labels = batch['labels'].to(accelerator.device)

            # Forward pass
            logits, h = model(input_ids)

            # Simulated loss calculation
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))

            accelerator.backward(loss)

            # Gradient clipping per-router/model
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if accelerator.is_main_process and step % config['training']['logging_steps'] == 0:
                wandb.log({"loss": loss.item(), "stage": stage_name, "step": step})
                print(f"[{stage_name}] Step {step} | Loss: {loss.item():.4f}")

        if step >= config['training'].get('max_steps_per_stage', 100):
            break

    if accelerator.is_main_process:
        wandb.finish()

def main():
    config = parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=config['training']['gradient_accumulation_steps'])

    model = MambaJEPAEngine(
        d_model=config['model']['d_model'],
        num_blocks=config['model']['num_blocks'],
        use_sycl_kernel=True
    )

    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])

    # Simulated dataloader for demonstration
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return 1000
        def __getitem__(self, idx):
            return {"input_ids": torch.randint(0, 1000, (64,)), "labels": torch.randint(0, 1000, (64,))}

    dataloader = DataLoader(DummyDataset(), batch_size=config['training']['batch_size'])

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Training Stages
    stages = ["pretrain", "alignment", "rl"]

    for stage in stages:
        if accelerator.is_main_process:
            print(f"=== Starting Stage: {stage.upper()} ===")
        train_stage(config, accelerator, model, dataloader, optimizer, stage_name=stage)

if __name__ == "__main__":
    main()
