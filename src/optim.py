import torch
import logging

def build_optimizer(model, decoder, learning_rate, device):
    galore_params, plain_params = [], []
    exclude = ["bias", "norm", "embedding", "A_log_spectral", "discrete_embeddings", "expansion_gate", "foresight_head"]

    modules = [m for m in [model, decoder] if m is not None]

    for module in modules:
        for n, p in module.named_parameters():
            if not p.requires_grad: continue
            if p.ndim >= 2 and not any(t in n for t in exclude) and p.shape[0] >= 256 and p.shape[1] >= 256:
                galore_params.append(p)
            else:
                plain_params.append(p)

    groups = [
        {'params': plain_params},
        {'params': galore_params, 'rank': 128, 'update_proj_gap': 200, 'scale': 0.25, 'proj_type': 'std'}
    ]

    if device.type == "cpu":
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

    try:
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(groups, lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    except Exception as e:
        logging.warning(f"8-bit optimizer unavailable: {e}. Falling back to AdamW.")
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
