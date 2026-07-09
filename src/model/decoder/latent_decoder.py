import torch
import torch.nn as nn

class ClosedLoopLatentDecoder(nn.Module):
    """
    Translates the JEPA world model's internal latent concept (h)
    back into vocabulary space logits.
    """
    def __init__(self, d_latent=2048, vocab_size=151936):
        super().__init__()
        self.d_latent = d_latent
        self.vocab_size = vocab_size

        self.decoder_proj = nn.Linear(d_latent, d_latent * 2)
        self.act = nn.GELU()
        self.output_head = nn.Linear(d_latent * 2, vocab_size, bias=False)

    def forward(self, h):
        x = self.act(self.decoder_proj(h))
        logits = self.output_head(x)
        return logits
