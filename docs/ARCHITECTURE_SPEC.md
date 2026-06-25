# PART 1: Architectural Specification (The Mamba2-Latent-Loop-4B Engine)

## I. System Overview & Hyperparameters

The **Mamba2-Latent-Loop-4B Engine** is a non-autoregressive, latent-predictive model utilizing **Mamba-2 State Space Duality (SSD)** blocks to maintain linear time and memory scaling ($O(N)$). It employs **Arbitrary Layer Graph Routing (ALGR)** to dynamically route tokens across its physical blocks based on computational complexity.

* **Total Parameter Count:** ~3.64 Billion
* **Vocabulary Size ($V$):** 151,643 (Directly matches the `Qwen/Qwen3.6-27B` tokenizer footprint)
* **Hidden Dimension ($d_{model}$):** 4096
* **Mamba Expansion Factor ($E$):** 2
* **Inner Dimension ($d_{inner}$):** $d_{model} \times E = 8192$
* **SSM State Dimension ($d_{state}$):** 128
* **SSM Head Dimension ($d_{head}$):** 128
* **Number of SSM Heads ($nheads$):** $d_{inner} / d_{head} = 64$
* **1D Convolution Width:** 4
* **Physical Blocks ($L_{physical}$):** 24 identical, re-routable Mamba-2 layers
* **Max Computational Budget ($T_{max}$):** 64 total block iterations per sequence
* **Latent/Concept Dimension ($d_{latent}$):** 1024

---

## II. Exact Tensor Shape Directory

To avoid runtime shape mismatches, Jules must enforce these precise dimensions across all operations:

* **Input Tokens:** `[Batch, Seq_Len]`
* **Initial Hidden State ($H_0$):** `[Batch, Seq_Len, 4096]`
* **Dynamic State Vector ($H_t$):** `[Batch, Seq_Len, 4096]`
* **Mamba-2 Combined Projection:** `[Batch, Seq_Len, 16512]` *(Calculated as $2 \times d_{inner} + 2 \times d_{state} + nheads \rightarrow 16384 + 256 + 64 = 16640$)*
* **SSM Recurrent State ($h_t$):** `[Batch, 64, 128, 128]`
* **Routing Tracker Matrix:** `[Batch, Seq_Len, 1]`
* **Router Probability Output:** `[Batch, Seq_Len, 25]` *(24 physical blocks + 1 exit state)*
* **Target Latent Vector ($Y_{latent}$):** `[Batch, 1024]`
* **Decoder Output Tokens:** `[Batch, Max_Output_Tokens, 151643]`

---

## III. Core Component Blueprint

### 1. The Multi-Hop Graph Router

Every physical Mamba-2 block owns an independent linear routing head. This head inspects the token vectors post-recurrency and determines whether a token sequence requires a loop-back, a self-repeat, a step-forward, or an early exit.

```
       [Hidden State H_t from Block N]
                     │
                     ├───► [Micro-Router N] ───► Softmax Logits
                     │                                 │
                     │  ┌──────────────────────────────┴──────────────────────────────┐
                     │  ▼ (Loop Back)            ▼ (Self Repeat)        ▼ (Next / Exit)
                     │  Route to Block N-k       Route to Block N       Route to N+1 or Decoder
                     │  [State Updates]          [State Updates]        [State Updates]
                     ▼
             [Add Iteration & Global Step Embeddings]

```

#### Reference PyTorch Implementation Structure:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaGraphRouter(nn.Module):
    def __init__(self, d_model=4096, num_blocks=24):
        super().__init__()
        self.routing_head = nn.Linear(d_model, num_blocks + 1)
        
    def forward(self, h, global_steps, max_budget=64):
        # h shape: [Batch, Seq_Len, 4096]
        # global_steps shape: [Batch, Seq_Len, 1]
        logits = self.routing_head(h) 
        
        # Terminal state enforcement
        mask = (global_steps >= max_budget).float()
        logits[:, :, :-1] = logits[:, :, :-1] * (1.0 - mask) - (mask * 1e9)
        logits[:, :, -1] = logits[:, :, -1] * (1.0 - mask) + (mask * 1e9)
        
        return F.softmax(logits, dim=-1)

```

### 2. State Augmentation Layers

To prevent vector saturation when recycling blocks, Jules must implement continuous learned embedding matrices to update the state before entering any block:

* `Embedding_global`: Maps integers $1 \dots 64 \rightarrow 4096$
* `Embedding_block`: Maps block indices $1 \dots 24 \rightarrow 4096$

$$H_{augmented} = H_{current} + \text{Embedding}_{global}(\text{steps}) + \text{Embedding}_{block}(\text{target\_idx})$$

### 3. Dual-Stage Latent Speculative Decoder

* **Stage 1 (Non-Autoregressive Draft Canvas):** A projection network that maps the final 1024-dimensional latent vector across a complete text canvas (`Max_Output_Tokens = 256`) simultaneously.
* **Stage 2 (Causal Speculative Sweep):** A 2-layer causal autoregressive transformer block that processes the Stage 1 output canvas in parallel, correcting syntax and code logic bugs (e.g., mismatched brackets or variables) without re-invoking the Mamba-2 reasoning blocks.

---