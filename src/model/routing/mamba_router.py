import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaGraphRouter(nn.Module):
    """
    Upgraded ALGR Head incorporating Einstein World Model sparse tool-activation gates
    and Fixed-Point Halting Signals (arXiv:2604.11791).
    """
    def __init__(self, d_model=6144, num_blocks=32, d_latent=2048):
        super().__init__()
        self.num_blocks = num_blocks
        # The router takes the latent state (h) which has dimension d_latent
        self.routing_head = nn.Linear(d_latent + 1, num_blocks + 2)

    def forward(self, h, h_delta, global_steps, max_budget=64):
        router_input = torch.cat([h, h_delta], dim=-1)
        logits = self.routing_head(router_input)

        mask = (global_steps >= max_budget).float()
        mask_sq = mask.squeeze(-1)

        logits[:, :, :-2] = logits[:, :, :-2] * (1.0 - mask) - (mask * 1e9)
        logits[:, :, -2] = logits[:, :, -2] * (1.0 - mask_sq) + (mask_sq * 1e9)

        return F.softmax(logits, dim=-1)
