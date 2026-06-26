import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaGraphRouter(nn.Module):
    """
    Arbitrary Layer Graph Routing (ALGR) Head.
    Evaluates hidden states and routes tokens dynamically.
    """
    def __init__(self, d_model=6144, num_blocks=32):
        super().__init__()
        self.num_blocks = num_blocks
        self.routing_head = nn.Linear(d_model, num_blocks + 1)

    def forward(self, h, global_steps, max_budget=64):
        # h: [Batch, Seq_Len, d_model]
        # global_steps: [Batch, Seq_Len, 1]
        logits = self.routing_head(h)

        # Force route to exit state (index -1) if computational budget is exhausted
        mask = (global_steps >= max_budget).float()
        mask_sq = mask.squeeze(-1)
        logits[:, :, :-1] = logits[:, :, :-1] * (1.0 - mask) - (mask * 1e9)
        logits[:, :, -1] = logits[:, :, -1] * (1.0 - mask_sq) + (mask_sq * 1e9)

        return F.softmax(logits, dim=-1)

class Mamba2SSDBlock(nn.Module):
    """
    Simulated State Space Duality (SSD) Layer.
    Optimized to map down to Intel XMX hardware matrix lanes.
    """
    def __init__(self, d_model=6144, d_state=128, nheads=96):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.nheads = nheads
        self.d_head = 128
        self.d_inner = nheads * self.d_head # 12288

        # Single fused projection layout to eliminate kernel dispatch bottlenecks
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + self.nheads)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=4, padding=3, groups=self.d_inner
        )
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x, mamba_state=None):
        # x: [Batch, Seq_Len, d_model]
        batch, seq_len, _ = x.shape
        fused_states = self.in_proj(x)

        # Splitting based on sizes: d_inner, d_inner, d_state, d_state, nheads
        x_split, z, B, C, dt = torch.split(
            fused_states,
            [self.d_inner, self.d_inner, self.d_state, self.d_state, self.nheads],
            dim=-1
        )

        conv_out = self.conv1d(x_split.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        activated = F.silu(conv_out)

        activated = activated.view(batch, seq_len, self.nheads, self.d_head)
        dt = F.softplus(dt)

        if mamba_state is None:
            mamba_state = torch.zeros(batch, self.nheads, self.d_state, self.d_state, device=x.device, dtype=x.dtype)

        B = B.unsqueeze(2).expand(-1, -1, self.nheads, -1)
        C = C.unsqueeze(2).expand(-1, -1, self.nheads, -1)

        outputs = []
        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1).unsqueeze(-1)
            decay = torch.exp(-dt_t)

            B_t = B[:, t, :, :].unsqueeze(-1)
            act_t = activated[:, t, :, :].unsqueeze(-2)

            mamba_state = decay * mamba_state + dt_t * torch.matmul(B_t, act_t)

            C_t = C[:, t, :, :].unsqueeze(-2)
            out_t = torch.matmul(C_t, mamba_state).squeeze(-2)

            outputs.append(out_t)

        out_tensor = torch.stack(outputs, dim=1)
        out_tensor = out_tensor.view(batch, seq_len, self.d_inner)

        out_tensor = out_tensor * F.silu(z)

        return self.out_proj(out_tensor), mamba_state

class Mamba2LatentLoop8B(nn.Module):
    def __init__(self, d_model=6144, num_blocks=32, max_budget=64):
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.max_budget = max_budget

        self.embedding_global = nn.Embedding(max_budget + 1, d_model)
        self.embedding_block = nn.Embedding(num_blocks + 1, d_model)

        self.blocks = nn.ModuleList([Mamba2SSDBlock(d_model=d_model) for _ in range(num_blocks)])
        self.routers = nn.ModuleList([MambaGraphRouter(d_model=d_model, num_blocks=num_blocks) for _ in range(num_blocks)])

    def forward(self, tokens, hidden_state=None, mamba_state=None):
        if hidden_state is None:
            hidden_state = torch.zeros(tokens.shape[0], tokens.shape[1], self.d_model, device=tokens.device, dtype=tokens.dtype)

        batch, seq_len, _ = hidden_state.shape
        global_steps = torch.zeros(batch, seq_len, 1, device=hidden_state.device, dtype=hidden_state.dtype)

        if mamba_state is None:
            mamba_state_list = [None] * self.num_blocks
        elif isinstance(mamba_state, torch.Tensor):
            mamba_state_list = list(torch.unbind(mamba_state, dim=1))
        else:
            mamba_state_list = mamba_state

        current_block_idx = 0
        new_mamba_state = [None] * self.num_blocks

        while (global_steps < self.max_budget).any() and current_block_idx < self.num_blocks:
            step_env = self.embedding_global(global_steps.squeeze(-1).long())
            block_env = self.embedding_block(torch.full_like(global_steps, current_block_idx).squeeze(-1).long())

            hidden_state = hidden_state + step_env + block_env

            block_out, new_mamba_state[current_block_idx] = self.blocks[current_block_idx](
                hidden_state, mamba_state_list[current_block_idx]
            )
            hidden_state = block_out + hidden_state

            route_probs = self.routers[current_block_idx](hidden_state, global_steps, self.max_budget)

            global_steps += 1
            current_block_idx += 1

        for i in range(self.num_blocks):
            if new_mamba_state[i] is None:
                if mamba_state_list[i] is not None:
                    new_mamba_state[i] = mamba_state_list[i]
                else:
                    new_mamba_state[i] = torch.zeros(
                        batch, self.blocks[i].nheads, self.blocks[i].d_state, self.blocks[i].d_state,
                        device=hidden_state.device, dtype=hidden_state.dtype
                    )

        mamba_state_out = torch.stack(new_mamba_state, dim=1)

        return hidden_state, global_steps, mamba_state_out

class LatentProjectionHead(nn.Module):
    def __init__(self, d_model=6144, d_latent=1024):
        super().__init__()
        self.proj = nn.Linear(d_model, d_latent)

    def forward(self, x):
        pooled = x.mean(dim=1)
        return self.proj(pooled)

class MambaJEPAEngine(nn.Module):
    def __init__(self, vocab_size=151643, d_model=6144, num_blocks=32, max_budget=64, d_latent=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.mamba_loop = Mamba2LatentLoop8B(d_model=d_model, num_blocks=num_blocks, max_budget=max_budget)
        self.projection_head = LatentProjectionHead(d_model=d_model, d_latent=d_latent)

    def forward(self, input_tokens, mamba_state=None):
        hidden_state = self.embedding(input_tokens)
        hidden_state, global_steps, mamba_state = self.mamba_loop(input_tokens, hidden_state=hidden_state, mamba_state=mamba_state)

        student_concept = self.projection_head(hidden_state)

        return student_concept, global_steps, mamba_state

class DualStageLatentDecoder(nn.Module):
    def __init__(self, d_latent=1024, max_seq_len=256, d_model=6144, vocab_size=151643):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.stage1_proj = nn.Linear(d_latent, max_seq_len * d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=16,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.stage2_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, concept_vector):
        batch_size = concept_vector.size(0)

        draft_seq = self.stage1_proj(concept_vector)
        draft_seq = draft_seq.view(batch_size, self.max_seq_len, self.d_model)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(self.max_seq_len).to(concept_vector.device)

        encoded_seq = self.stage2_transformer(draft_seq, mask=causal_mask, is_causal=True)

        logits = self.output_proj(encoded_seq)

        return logits
