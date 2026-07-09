import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalLatentProjectionHead(nn.Module):
    """
    Projects the sequence from Mamba's inner dimension space into the
    JEPA World Model's contrastive state space (h).
    """
    def __init__(self, d_model=6144, d_latent=2048):
        super().__init__()
        self.proj_1 = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.proj_2 = nn.Linear(d_model, d_latent)

        self.norm = nn.LayerNorm(d_latent)

    def forward(self, x):
        h = self.act(self.proj_1(x))
        h = self.proj_2(h)
        return self.norm(h)
