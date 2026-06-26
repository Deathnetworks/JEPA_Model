# Training Architecture: Hybrid Mamba2-JEPA

This document details the training methodology for the 8B JEPA World Model. Because the architecture abandons standard transformer attention in favor of a Mamba2 backbone with dynamic routing and a Joint-Embedding Predictive Architecture (JEPA) head, standard Hugging Face `Trainer` loops are insufficient.

The training pipeline (`train_jepa_world_model.py`) is a custom PyTorch XPU implementation designed to solve two specific problems:

1. Processing "infinite" context reasoning traces without exceeding 32GB VRAM.
2. Preventing "latent collapse" when balancing generative text prediction with conceptual alignment.

---

## ♾️ 1. Chunked State-Passing (Truncated BPTT)

Frontier reasoning traces (e.g., from Claude Opus or Qwen 3.7 Max) can stretch beyond 40,000 tokens. To train on these without triggering an XPU Out-Of-Memory (OOM) exception, the model implements **Chunked State-Passing**, a variant of Truncated Backpropagation Through Time (TBPTT).

### The Mechanism

1. **Sequence Chunking:** Incoming variable-length sequences are padded to a multiple of `4096` tokens and split into discrete chunks.
2. **Forward Pass & State Extraction:** As Chunk $N$ passes through the Mamba2 blocks, the final recurrent hidden state ($h_t$) is intercepted.
3. **Graph Detachment:** To prevent the PyTorch autograd graph from tracking backward into the previous chunk (which would consume VRAM infinitely), the state is detached: `h_t = h_t.detach()`.
4. **State Re-injection:** The detached $h_t$ is passed as the initial state constraint for Chunk $N+1$.

This allows the model to "remember" the logic from token $0$ all the way to token $40,000$ while only calculating gradients for a rigid $4096$-token window at any given time.

---

## ⚖️ 2. The Tri-Partite Loss Function

Training a hybrid model requires satisfying three completely different architectural objectives simultaneously. The total loss $\mathcal{L}_{total}$ is calculated as:

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_{JEPA}(t) \mathcal{L}_{JEPA} + \lambda_{Route} \mathcal{L}_{Route}$$

### A. Next-Token Prediction ($\mathcal{L}_{CE}$)

* **Function:** Standard Cross-Entropy Loss.
* **Target:** The `qwen_tokens` (the frontier model's response).
* **Purpose:** Teaches the Mamba2 backbone and base MLPs how to generate syntactically correct text and code.

### B. JEPA Latent Alignment ($\mathcal{L}_{JEPA}$)

* **Function:** Cosine Embedding Loss (`F.cosine_embedding_loss`).
* **Target:** The 1024-d `target_concept` generated offline by BGE-Large.
* **Purpose:** Forces the model's internal representation (the average-pooled output of the routers) to geometrically align with the true mathematical meaning of the target response, rather than just memorizing tokens.

### C. Router Load Balancing ($\mathcal{L}_{Route}$)

* **Function:** Z-Loss / Sparsity Penalty.
* **Target:** The internal probability distributions of the dynamic layer looping routers.
* **Purpose:** Deep routed networks suffer from "routing collapse" (lazy routing where it skips all layers or loops infinitely). This penalty enforces a soft maximum loop constraint (e.g., `max_loops=4`) and ensures execution depth varies dynamically based on token complexity.

---

## 📈 3. Dynamic Loss Scheduling (Preventing Latent Collapse)

If $\mathcal{L}_{CE}$ and $\mathcal{L}_{JEPA}$ are weighted equally at step 0, the model will experience **Latent Collapse**. The Cross-Entropy gradients will overwhelm the network, causing it to completely ignore the JEPA concept vectors.

To fix this, $\lambda_{JEPA}$ is tied to a **Dynamic Exponential Scheduler**:

* **Step 0:** $\lambda_{JEPA} = 0.01$ (Allows the model to learn basic token syntax first).
* **Warmup Phase (e.g., first 10% of steps):** Scales exponentially.
* **Post-Warmup:** $\lambda_{JEPA} = 1.0$ (The JEPA head acts as a rigid anchor, forcing the text generation to follow the conceptual world model).

---

## 💻 4. XPU Hardware Implementation Rules

To successfully compile and run this 8B architecture on the Intel Arc Pro B70, the training script enforces strict hardware flags:

* **Backend Compilation:** `torch.compile(backend="inductor", mode="max-autotune")` allows the native SYCL compiler to aggressively fuse the custom routing and JEPA projection kernels.
* **Optimizer States:** Uses `bitsandbytes.optim.AdamW8bit`. A standard 32-bit Adam optimizer for an 8B model requires ~64GB of VRAM. 8-bit Adam compresses this to ~16GB, leaving room for the batch activations.
* **Precision:** Enforces `torch.autocast(device_type="xpu", dtype=torch.bfloat16)` to prevent NaN spiking during deep routing loops.
* **Gradient Accumulation:** Uses a micro-batch size of `1` (one 4096 chunk) to protect memory, with `accumulation_steps=16` before calling `optimizer.step()` to simulate a massive batch size for stable JEPA cosine alignment.