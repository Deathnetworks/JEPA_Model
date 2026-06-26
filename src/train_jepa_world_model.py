import os
import glob
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import bitsandbytes as bnb
from accelerate import Accelerator
from src.model_architecture import MambaJEPAEngine, DualStageLatentDecoder

class JEPAShardDataset(Dataset):
    def __init__(self, data_dir, curriculum_phase="logic"):
        self.data_dir = data_dir
        self.curriculum_phase = curriculum_phase
        # Use memory detail about curriculum phase filtering
        self.files = sorted(glob.glob(os.path.join(data_dir, f"{curriculum_phase}_set_*.pt")))
        if not self.files:
            # Fallback if no specific phase files are found
            self.files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Expected to return dict with: input_tokens, qwen_tokens, target_concept
        data = torch.load(self.files[idx], map_location="cpu")
        return data

def get_dataloader(data_dir, curriculum_phase="logic", batch_size=1):
    dataset = JEPAShardDataset(data_dir, curriculum_phase=curriculum_phase)
    # Important: shuffle=False for TBPTT streaming
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

def plan_step_complete():
    pass

class TripartiteLoss(nn.Module):
    def __init__(self, max_loops=4):
        super().__init__()
        self.max_loops = max_loops
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, qwen_tokens, jepa_out, target_concept, router_probs, lambda_jepa, lambda_route=0.01):
        # L_CE: Cross-Entropy loss between logits and qwen_tokens
        # Flatten logits and targets for CE loss
        l_ce = self.ce_loss(logits.view(-1, logits.size(-1)), qwen_tokens.view(-1))

        # L_JEPA: Cosine Embedding Loss between JEPA projection head output and target_concept
        # Target for Cosine Embedding Loss is 1 (meaning they should be similar)
        batch_size = jepa_out.size(0)
        target = torch.ones(batch_size, device=jepa_out.device)
        l_jepa = F.cosine_embedding_loss(jepa_out, target_concept, target)

        # L_Route: Router Z-loss penalty
        # Penalizes if routing distribution becomes excessively sparse or loops beyond max_loops
        # router_probs shape is usually [batch, seq_len, num_layers] or similar
        # For simplicity in this placeholder, assuming router_probs contains loop counts or probs
        # Compute mean squared difference from uniform or penalty for high loop counts
        if router_probs is not None:
            # Placeholder for router loss calculation based on specifics of MambaJEPAEngine
            # A common Z-loss formulation:
            l_route = torch.mean(torch.square(torch.logsumexp(router_probs, dim=-1)))
            # Additionally, penalize exceeding max_loops (assuming router_probs encodes loops somehow)
        else:
            l_route = torch.tensor(0.0, device=logits.device)

        total_loss = l_ce + (lambda_jepa * l_jepa) + (lambda_route * l_route)
        return total_loss, l_ce, l_jepa, l_route

def get_lambda_jepa(step, warmup_steps=1000):
    """
    Scheduler for lambda_jepa that starts at 0.01 at step 0 and
    linearly scales to 1.0 by the end of the warmup phase.
    """
    if step >= warmup_steps:
        return 1.0

    start_val = 0.01
    end_val = 1.0
    # Linear interpolation
    progress = step / warmup_steps
    return start_val + progress * (end_val - start_val)

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """
    Cosine Decay learning rate scheduler with a warmup phase.
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, decoder, dataloader, optimizer, lr_scheduler, accelerator, epoch, args):
    model.train()
    decoder.train()

    criterion = TripartiteLoss(max_loops=4)
    chunk_size = 4096
    accumulation_steps = 16

    total_loss = 0
    total_ce_loss = 0
    total_jepa_loss = 0
    total_route_loss = 0

    # We will accumulate gradients over 16 chunks
    chunk_counter = 0

    for batch_idx, batch in enumerate(dataloader):
        # We need to run TBPTT loop on each sequence
        # batch represents a dictionary with 'input_tokens', 'qwen_tokens', 'target_concept'
        input_tokens = batch['input_tokens']
        qwen_tokens = batch['qwen_tokens']
        target_concept = batch['target_concept']

        # Sequences might be long, so chunk them up.
        seq_len = input_tokens.size(1)
        mamba_state = None

        for i in range(0, seq_len, chunk_size):
            chunk_input = input_tokens[:, i:i+chunk_size]
            chunk_qwen = qwen_tokens[:, i:i+chunk_size] if qwen_tokens.size(1) > 1 else qwen_tokens

            # Forward pass wrapped in autocast for XPU bfloat16
            with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
                # 1. Forward Pass
                student_concept, global_steps, mamba_state = model(chunk_input, mamba_state=mamba_state)

                # The decoder maps the student concept to vocabulary logits
                # Using the chunk size or the decoder's expected sequence length
                logits = decoder(student_concept)

                # Make sure logits and chunk_qwen match in seq length if necessary.
                # DualStageLatentDecoder usually outputs [Batch, max_seq_len, vocab_size].
                # For CE loss we must align logits and qwen_tokens.
                # Assuming the decoder outputs a sequence that should match chunk_qwen.
                min_len = min(logits.size(1), chunk_qwen.size(1))
                logits_aligned = logits[:, :min_len, :]
                chunk_qwen_aligned = chunk_qwen[:, :min_len]

                # Calculate lambda_jepa for current step
                current_step = optimizer.step_count if hasattr(optimizer, 'step_count') else 0
                lambda_jepa = get_lambda_jepa(current_step, warmup_steps=1000)

                # Calculate loss
                # Note: global_steps represents loops, we pass it as router_probs for now to calculate Z-loss penalty
                loss, l_ce, l_jepa, l_route = criterion(
                    logits_aligned,
                    chunk_qwen_aligned,
                    student_concept,
                    target_concept,
                    global_steps,
                    lambda_jepa
                )

                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps

            # Backward pass using accelerator
            accelerator.backward(loss)

            # 2. State Save: Detach the mamba state
            if mamba_state is not None:
                # Based on the memory: explicit pass, return, and detach()
                if isinstance(mamba_state, tuple) or isinstance(mamba_state, list):
                    mamba_state = tuple(s.detach() for s in mamba_state if s is not None)
                elif isinstance(mamba_state, dict):
                    mamba_state = {k: v.detach() for k, v in mamba_state.items() if v is not None}
                else:
                    mamba_state = mamba_state.detach()

            chunk_counter += 1

            # Optimization step
            if chunk_counter % accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Manual cache clearing to flatten VRAM curve
                torch.xpu.empty_cache()

            # Metrics
            total_loss += loss.item() * accumulation_steps
            total_ce_loss += l_ce.item()
            total_jepa_loss += l_jepa.item()
            total_route_loss += l_route.item()

        # Empty cache after each sequence batch processing
        torch.xpu.empty_cache()

    return total_loss / max(1, chunk_counter), total_ce_loss / max(1, chunk_counter)


def main():
    parser = argparse.ArgumentParser(description="Train MambaJEPAEngine on Arc Pro B70 (XPU)")
    parser.add_argument("--data_dir", type=str, default="data/shards", help="Path to .pt shards")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (simulated via GA=16)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    # 1. Initialize Accelerator for XPU
    # As per memory, HF accelerate is used for device management.
    # We must explicitly target the XPU. (Actually, Accelerate detects it automatically,
    # but we can ensure device='xpu' during manual moves if needed.)
    accelerator = Accelerator(
        gradient_accumulation_steps=16,
        mixed_precision="bf16"
    )

    device = torch.device("xpu") if torch.xpu.is_available() else accelerator.device

    print(f"Using device: {device}")

    # 2. Initialize Models
    # Apply heavily downgraded parameters if running on CPU (memory constrained sandbox)
    if device.type == 'cpu':
        print("WARNING: Running on CPU, using heavily downgraded hyperparameters to avoid OOM.")
        model = MambaJEPAEngine(d_model=64, num_blocks=2, max_budget=2, d_latent=1024)
        decoder = DualStageLatentDecoder(d_model=64, d_latent=1024)
    else:
        # Full 8B parameters
        model = MambaJEPAEngine()
        decoder = DualStageLatentDecoder()

    # Apply torch.compile with aggressive Inductor config as specified in memory
    # "torch._inductor.config tuning (e.g., freezing=True, max_autotune=True, coordinate_descent_tuning=True)"
    if device.type == "xpu":
        torch._inductor.config.freezing = True
        torch._inductor.config.max_autotune = True
        torch._inductor.config.coordinate_descent_tuning = True

        # We need native SDPA - this is usually done at model architecture level,
        # but compiling with Inductor uses it automatically where applicable.
        print("Compiling models with Inductor for XPU...")
        model = torch.compile(model, backend="inductor")
        decoder = torch.compile(decoder, backend="inductor")

    model = model.to(device)
    decoder = decoder.to(device)

    # 3. Initialize 8-bit Optimizer
    # 8-bit AdamW to reduce optimizer state memory
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
        {"params": [p for n, p in decoder.named_parameters() if p.requires_grad]}
    ]

    optimizer = bnb.optim.AdamW8bit(
        optimizer_grouped_parameters,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # 4. Prepare Dataloader
    dataloader = get_dataloader(args.data_dir, batch_size=args.batch_size)

    # Calculate steps
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(0.05 * total_steps) # 5% warmup

    # 5. Initialize Schedulers
    lr_scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    # 6. Accelerate Prepare
    model, decoder, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, decoder, optimizer, dataloader, lr_scheduler
    )

    # 7. Training Loop
    # Keep track of training state for continuous checkpoint resumption
    checkpoint_file = "training_state.txt"
    start_epoch = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            start_epoch = int(f.read().strip())
        print(f"Resuming from epoch {start_epoch}")

    # We must explicitly track steps for lambda_jepa scheduler if step count isn't in optimizer
    optimizer.step_count = 0

    # Hook into step to track step count
    original_step = optimizer.step
    def step_with_count(*args, **kwargs):
        original_step(*args, **kwargs)
        optimizer.step_count += 1
    optimizer.step = step_with_count

    for epoch in range(start_epoch, args.epochs):
        print(f"Starting Epoch {epoch+1}/{args.epochs}")

        train_epoch(
            model=model,
            decoder=decoder,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            epoch=epoch,
            args=args
        )

        # Save state
        if accelerator.is_main_process:
            with open(checkpoint_file, "w") as f:
                f.write(str(epoch + 1))

            print(f"Epoch {epoch+1} completed.")

if __name__ == "__main__":
    main()
