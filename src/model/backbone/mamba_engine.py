import torch
import torch.nn as nn
from src.model.backbone.mamba_block import OptimizedMambaBlock
from src.model.routing.mamba_router import MambaGraphRouter
from src.model.heads.projection_head import HierarchicalLatentProjectionHead
from src.model.decoder.latent_decoder import ClosedLoopLatentDecoder

class MambaJEPAEngine(nn.Module):
    def __init__(self, d_model=6144, num_blocks=32, vocab_size=151936, use_sycl_kernel=True):
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks

        self.embedding = nn.Embedding(vocab_size, d_model)

        # Modular blocks
        self.blocks = nn.ModuleList([
            OptimizedMambaBlock(d_model=d_model, use_sycl_kernel=use_sycl_kernel)
            for _ in range(num_blocks)
        ])

        self.router = MambaGraphRouter(d_model=d_model, num_blocks=num_blocks, d_latent=2048)
        self.jepa_proj = HierarchicalLatentProjectionHead(d_model=d_model, d_latent=2048)
        self.decoder = ClosedLoopLatentDecoder(d_latent=2048, vocab_size=vocab_size)

    def forward(self, input_ids, max_budget=64):
        x = self.embedding(input_ids)
        batch_size, seq_len, _ = x.shape

        # Initialize loop state
        h = self.jepa_proj(x)
        h_delta = torch.zeros(batch_size, seq_len, 1, device=x.device)

        global_steps = torch.zeros(batch_size, seq_len, 1, device=x.device)

        # We will iterate for max_budget as a simplified loop representation
        for step in range(max_budget):
            routing_weights = self.router(h, h_delta, global_steps, max_budget)

            # Apply dynamic routing (Simplified for skeleton)
            for i, block in enumerate(self.blocks):
                x, _ = block(x)

            h_new = self.jepa_proj(x)
            h_delta = torch.norm(h_new - h, dim=-1, keepdim=True)
            h = h_new
            global_steps += 1

        logits = self.decoder(h)
        return logits, h
