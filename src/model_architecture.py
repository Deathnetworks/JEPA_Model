import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaGraphRouter(nn.Module):
    """
    Upgraded ALGR Head incorporating Einstein World Model sparse tool-activation gates
    and Fixed-Point Halting Signals (arXiv:2604.11791).
    """
    def __init__(self, d_model=6144, num_blocks=32):
        super().__init__()
        self.num_blocks = num_blocks
        # NEW: Input dimension is d_model + 1 (to accept the h_delta L2 distance)
        self.routing_head = nn.Linear(d_model + 1, num_blocks + 2)

    def forward(self, h, h_delta, global_steps, max_budget=64):
        # Concatenate the latent state with the continuous convergence signal
        router_input = torch.cat([h, h_delta], dim=-1)
        logits = self.routing_head(router_input)
        
        # Force terminate if computation budget is completely exhausted
        mask = (global_steps >= max_budget).float()
        mask_sq = mask.squeeze(-1)
        
        # Suppress active computation paths if budget is blown
        logits[:, :, :-2] = logits[:, :, :-2] * (1.0 - mask) - (mask * 1e9)
        logits[:, :, -2] = logits[:, :, -2] * (1.0 - mask_sq) + (mask_sq * 1e9)
        
        return F.softmax(logits, dim=-1)

class Mamba2SSDBlock(nn.Module):
    """
    Simulated State Space Duality (SSD) Layer.
    Tuned for ~19.5GB BF16 Training / 9.7GB INT8 Inference.
    """
    def __init__(self, d_model=6144, d_state=256, nheads=128):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.nheads = nheads
        
        self.d_head = 64 
        self.d_inner = nheads * self.d_head # 8192

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
            mamba_state = torch.zeros(batch, self.nheads, self.d_state, self.d_head, device=x.device, dtype=x.dtype)

        B = B.unsqueeze(2).expand(-1, -1, self.nheads, -1)
        C = C.unsqueeze(2).expand(-1, -1, self.nheads, -1)

        # --- NEW: Chunked SSD Architecture ---
        chunk_size = min(64, seq_len)
        outputs = []

        # We need to process sequentially over chunks to maintain recurrent state
        # but can parallelize within each chunk.
        for t_start in range(0, seq_len, chunk_size):
            t_end = min(t_start + chunk_size, seq_len)
            c_len = t_end - t_start

            dt_chunk = dt[:, t_start:t_end, :] # (B, L, H)
            B_chunk = B[:, t_start:t_end, :, :].unsqueeze(-1) # (B, L, H, N, 1)
            act_chunk = activated[:, t_start:t_end, :, :].unsqueeze(-2) # (B, L, H, 1, D)
            C_chunk = C[:, t_start:t_end, :, :].unsqueeze(-2) # (B, L, H, 1, N)

            # V = dt * (B @ act) -> (B, L, H, N, D)
            V = dt_chunk.unsqueeze(-1).unsqueeze(-1) * (B_chunk @ act_chunk)

            neg_dt = -dt_chunk
            cumsum_dt = torch.cumsum(neg_dt, dim=1)

            C_t = cumsum_dt.unsqueeze(2) # (B, L, 1, H)
            C_i = cumsum_dt.unsqueeze(1) # (B, 1, L, H)
            decay_matrix = C_t - C_i

            mask = torch.tril(torch.ones(c_len, c_len, device=x.device)).view(1, c_len, c_len, 1)
            decay_matrix = decay_matrix.masked_fill(mask == 0, float('-inf'))
            decay_weights = torch.exp(decay_matrix)

            H_intra = torch.einsum('btih,bihnd->bthnd', decay_weights, V)

            decay_state = torch.exp(cumsum_dt).view(batch, c_len, self.nheads, 1, 1)
            H_inter = decay_state * mamba_state.unsqueeze(1)

            H = H_intra + H_inter

            out = (C_chunk @ H).squeeze(-2)
            outputs.append(out)

            mamba_state = H[:, -1]

        out_tensor = torch.cat(outputs, dim=1)
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

        # --- EXISTING: Spectral Injection Constraints ---
        self.A_log_spectral = nn.Parameter(torch.randn(d_model))
        
        # --- NEW: DiscoLoop Discrete Channel (arXiv:2607.00341) ---
        # Soft Vector Quantization bottleneck to prevent continuous state drift during multi-hop reasoning
        self.num_discrete_concepts = 4096
        self.discrete_bottleneck = nn.Linear(d_model, self.num_discrete_concepts)
        self.discrete_embeddings = nn.Embedding(self.num_discrete_concepts, d_model)
        # Initialize embeddings to small random values to prevent early disruption
        nn.init.normal_(self.discrete_embeddings.weight, std=0.02)

        self.blocks = nn.ModuleList([Mamba2SSDBlock(d_model=d_model) for _ in range(num_blocks)])
        self.routers = nn.ModuleList([MambaGraphRouter(d_model=d_model, num_blocks=num_blocks) for _ in range(num_blocks)])

    def forward(self, tokens, hidden_state=None, mamba_state=None, active_budget=64):
        if hidden_state is None:
            hidden_state = torch.zeros(tokens.shape[0], tokens.shape[1], self.d_model, device=tokens.device, dtype=tokens.dtype)

        batch, seq_len, _ = hidden_state.shape
        global_steps = torch.zeros(batch, seq_len, 1, device=hidden_state.device, dtype=hidden_state.dtype)
        
        expected_steps = torch.zeros(batch, seq_len, 1, device=hidden_state.device, dtype=hidden_state.dtype)
        unhalted_prob = torch.ones(batch, seq_len, 1, device=hidden_state.device, dtype=hidden_state.dtype)

        if mamba_state is None:
            mamba_state_list = [None] * self.num_blocks
        elif isinstance(mamba_state, torch.Tensor):
            mamba_state_list = list(torch.unbind(mamba_state, dim=1))
        else:
            mamba_state_list = mamba_state

        current_block_idx = 0
        new_mamba_state = [None] * self.num_blocks
        
        prev_hidden_state = hidden_state.clone()

        while (global_steps < active_budget).any() and current_block_idx < self.num_blocks:
            active_mask = (unhalted_prob > 0.05).float()
            
            step_env = self.embedding_global(global_steps.squeeze(-1).long())
            block_env = self.embedding_block(torch.full_like(global_steps, current_block_idx).squeeze(-1).long())

            A_discrete = torch.exp(-torch.exp(self.A_log_spectral) * 1.0)
            hidden_state = (hidden_state * A_discrete) + step_env + block_env

            block_out, new_mamba_state[current_block_idx] = self.blocks[current_block_idx](
                hidden_state, mamba_state_list[current_block_idx]
            )
            
            # --- NEW: DiscoLoop Discrete Realignment (arXiv:2607.00341) ---
            # 1. Project the continuous block output to the discrete concept space
            discrete_logits = self.discrete_bottleneck(block_out)
            # 2. Extract soft probabilities (maintains backprop differentiability)
            soft_discrete_idx = F.softmax(discrete_logits, dim=-1)
            # 3. Retrieve the discrete embeddings via matrix multiplication
            discrete_channel = torch.matmul(soft_discrete_idx, self.discrete_embeddings.weight)
            
            # Combine the continuous state with the explicitly grounded discrete channel
            block_out_aligned = block_out + discrete_channel

            # Apply Token-Level Active Mask using the explicitly aligned output
            hidden_state = (block_out_aligned * active_mask) + hidden_state

            h_delta = torch.norm(hidden_state - prev_hidden_state, dim=-1, keepdim=True)
            prev_hidden_state = hidden_state.clone()

            route_probs = self.routers[current_block_idx](hidden_state, h_delta, global_steps, self.max_budget)
            
            halt_prob = route_probs[:, :, -2].unsqueeze(-1)
            step_halt_prob = halt_prob * unhalted_prob
            expected_steps = expected_steps + (global_steps + 1) * step_halt_prob
            unhalted_prob = unhalted_prob * (1.0 - halt_prob)

            global_steps += active_mask
            current_block_idx += 1
            
        expected_steps = expected_steps + global_steps * unhalted_prob

        for i in range(self.num_blocks):
            if new_mamba_state[i] is None:
                if mamba_state_list[i] is not None:
                    new_mamba_state[i] = mamba_state_list[i]
                else:
                    new_mamba_state[i] = torch.zeros(
                        batch, self.blocks[i].nheads, self.blocks[i].d_state, self.blocks[i].d_head,
                        device=hidden_state.device, dtype=hidden_state.dtype
                    )

        mamba_state_out = torch.stack(new_mamba_state, dim=1)

        return hidden_state, expected_steps, mamba_state_out

class HierarchicalLatentProjectionHead(nn.Module):
    """
    Dynamic Hierarchical Concept Expansion (Mitigates Latent Saturation).
    """
    def __init__(self, d_model=6144, d_micro=1024, d_macro=4096):
        super().__init__()
        self.d_micro = d_micro
        self.d_macro = d_macro
        
        self.micro_proj = nn.Linear(d_model, d_micro)
        self.macro_proj = nn.Linear(d_model, d_macro)
        
        # UPGRADE: Removed Sigmoid from Sequential to apply Temperature Scaling manually
        self.expansion_gate = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        pooled = x.mean(dim=1)
        micro_concept = self.micro_proj(pooled)
        
        # UPGRADE: Temperature Scaled Sigmoid (T=0.1) for a strict On/Off boundary
        gate_logits = self.expansion_gate(pooled)
        gate_scores = torch.sigmoid(gate_logits / 0.1) 
        
        macro_concept = self.macro_proj(pooled) * gate_scores
        
        return torch.cat([micro_concept, macro_concept], dim=-1)

class MambaJEPAEngine(nn.Module):
    # Notice d_latent defaults to 1024 (Micro) + 4096 (Macro) = 5120
    def __init__(self, vocab_size=151680, d_model=6144, num_blocks=32, max_budget=64, d_latent=5120):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.mamba_loop = Mamba2LatentLoop8B(d_model=d_model, num_blocks=num_blocks, max_budget=max_budget)
        
        # --- NEW: Hierarchical Expansion ---
        self.projection_head = HierarchicalLatentProjectionHead(d_model=d_model, d_micro=1024, d_macro=4096)
        
        self.foresight_head = nn.Sequential(
            nn.Linear(d_latent, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.max_budget = max_budget

    def forward(self, input_tokens, mamba_state=None, active_budget=None):
        hidden_state = self.embedding(input_tokens)
        
        if active_budget is None:
            active_budget = self.max_budget
        
        if self.training and self.max_budget > 2 and torch.rand(1).item() < 0.3:
            active_budget = torch.randint(low=2, high=self.max_budget, size=(1,)).item()

        hidden_state, global_steps, mamba_state = self.mamba_loop(
            input_tokens, 
            hidden_state=hidden_state, 
            mamba_state=mamba_state, 
            active_budget=active_budget
        )

        student_concept = self.projection_head(hidden_state)

        # Note: We do NOT alter the return signature to prevent breaking train_latent_loop.py unpacking.
        # The foresight head will be invoked sequentially during the GRPO phase.
        return student_concept, global_steps, mamba_state

class ClosedLoopLatentDecoder(nn.Module):
    """
    Upgraded Decoder mapping to EWM principles.
    Treats the latent concept vector as an examineable canvas conditioned via cross-attention.
    """
    # CRITICAL: Default d_latent updated to 5120 to ingest the hierarchical space
    def __init__(self, d_latent=5120, max_seq_len=256, d_model=6144, vocab_size=151680):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        self.concept_to_memory = nn.Linear(d_latent, 16 * d_model) 
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=16, dim_feedforward=d_model * 4, dropout=0.1, activation="gelu", batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
        # CRITICAL UPGRADE: Remove bias, and physically tie the weights
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight # Eliminates the tying gap

    def forward(self, target_tokens, concept_vector, prev_concept_vector=None):
        batch_size = concept_vector.size(0)
        seq_len = target_tokens.size(1)
        
        # --- NEW: Delta-JEPA Latent Difference (arXiv:2606.31232) ---
        # Reconstruct the code structure from the geometric displacement of the thought
        if prev_concept_vector is not None:
            delta_concept = concept_vector - prev_concept_vector
        else:
            delta_concept = concept_vector
        
        # 1. Project the displacement vector into explicit "inspectable frames" memory
        concept_memory = self.concept_to_memory(delta_concept).view(batch_size, 16, self.d_model)
        
        # 2. Embed the text reasoning tokens
        token_embeddings = self.token_embedding(target_tokens)
        
        # 3. Create causal mask for token sequence text autoregression
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(concept_vector.device)
        
        # 4. Decode text conditioned directly on the externalized latent displacement via cross-attention
        decoded_seq = self.transformer_decoder(
            tgt=token_embeddings,
            memory=concept_memory,
            tgt_mask=causal_mask,
            tgt_is_causal=True
        )
        
        return self.output_proj(decoded_seq)

class DualStageLatentDecoder(nn.Module):
    def __init__(self, d_latent=1024, max_seq_len=256, d_model=6144, vocab_size=151680):
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
