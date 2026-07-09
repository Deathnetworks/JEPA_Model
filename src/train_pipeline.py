import os
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
import wandb
import bitsandbytes as bnb

from src.model.backbone.mamba_engine import MambaJEPAEngine
from src.config import parse_args
from src.data_pipeline import StreamingFrontierDataset

class TripartiteLoss(nn.Module):
    def __init__(self, lambda_jepa=0.5, lambda_route=0.01):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_jepa = lambda_jepa
        self.lambda_route = lambda_route

    def forward(self, logits, labels, h, routing_penalties=None):
        # Flatten for Cross Entropy
        l_ce = self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))

        # JEPA Cosine alignment proxy (in a real scenario we'd use h vs target embeddings)
        # Using variance maximization as a stand-in for preventing representation collapse
        h_flat = h.view(-1, h.size(-1))
        l_jepa = -torch.var(h_flat, dim=0).mean() * self.lambda_jepa

        # ALGR Routing penalty (encourages early halting)
        l_route = 0
        if routing_penalties is not None:
            l_route = routing_penalties.mean() * self.lambda_route

        return l_ce + l_jepa + l_route, l_ce, l_jepa

def train_stage(config, accelerator, model, dataset_iterator, optimizer, criterion, stage_name="pretrain"):
    model.train()

    # Initialize wandb
    if accelerator.is_main_process:
        wandb.init(project="JEPA_Model_v2", name=f"{stage_name}_run", config=config)

    max_steps = config['training'].get('max_steps_per_stage', 100)

    for step in range(max_steps):
        with accelerator.accumulate(model):
            try:
                batch = next(dataset_iterator)
            except StopIteration:
                break

            input_ids = batch['input_ids'].unsqueeze(0).to(accelerator.device)
            labels = batch['labels'].unsqueeze(0).to(accelerator.device)

            # Forward pass
            logits, h = model(input_ids)

            # Loss calculation
            loss, l_ce, l_jepa = criterion(logits, labels, h)

            accelerator.backward(loss)

            # Gradient clipping per-router/model
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            if accelerator.is_main_process and step % config['training']['logging_steps'] == 0:
                wandb.log({
                    "loss": loss.item(),
                    "ce_loss": l_ce.item(),
                    "jepa_loss": l_jepa.item(),
                    "stage": stage_name,
                    "step": step
                })
                print(f"[{stage_name}] Step {step}/{max_steps} | Loss: {loss.item():.4f}")

        # Clean up XPU cache after chunking to flatten VRAM curve
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()

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

    # Use 8-bit optimizer state quantization for memory constraints
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config['training']['learning_rate'])

    dataset = StreamingFrontierDataset(config)
    dataset_iterator = dataset.get_iterator(split='train')

    model, optimizer = accelerator.prepare(model, optimizer)

    criterion = TripartiteLoss()

    # Training Stages
    stages = ["pretrain", "alignment", "rl"]

    for stage in stages:
        if accelerator.is_main_process:
            print(f"=== Starting Stage: {stage.upper()} ===")
        train_stage(config, accelerator, model, dataset_iterator, optimizer, criterion, stage_name=stage)

        # Save checkpoint between stages
        if accelerator.is_main_process:
            save_path = os.path.join(config['training']['checkpoint_dir'], f"model_{stage}.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            accelerator.save_state(save_path)

if __name__ == "__main__":
    main()
