# PART 2: Master Execution Plan

```
┌──────────────────────────────────┐      ┌──────────────────────────────────┐
│   Phase 1: Environment & Arch    │ ───► │  Phase 2: Frontier Ingestion     │
│   - Setup Native PyTorch XPU     │      │  - Asynchronous Stream Pipeline  │
│   - Initialize 8B Mamba2 Core    │      │  - Cache Vector Maps (.pt)       │
└──────────────────────────────────┘      └──────────────────────────────────┘
                                                            │
                                                            ▼
┌──────────────────────────────────┐      ┌──────────────────────────────────┐
│   Phase 4: LRE Native Runtime    │ ◄─── │    Phase 3: Hybrid Training      │
│   - Port Tokenizer & Execution   │      │  - Chunked State-Passing TBPTT   │
│   - Native SYCL/XPU Inference    │      │  - Tri-Partite Loss Dynamics     │
└──────────────────────────────────┘      └──────────────────────────────────┘

```

## Phase 1: Environment & Architecture Initialization

1. **Native XPU Workspace Alignment:** Establish a clean Python 3.10+ virtual environment. Force-install modern upstream PyTorch wheels compiled explicitly for Intel hardware configurations:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

```


2. **Memory Footprint Baselining:** Initialize the 32 physical blocks of the 6144-hidden-dimension Mamba2 architecture. Bind `bitsandbytes.optim.AdamW8bit` as the core optimization driver to keep the base weights and gradient states compressed under 16GB, leaving the remainder of the 32GB allocation wide open for token activation processing.
3. **Hardware Targeting Verification:** Enforce strict device casting hooks (`device = torch.device("xpu")`) across all tensor instantiation routines. Do not declare any legacy CUDA or third-party compatibility layers.
4. **Kernel Fusion Verification:** Execute an immediate tracking test using `torch.compile(model, backend="inductor")` to ensure the upstream SYCL compiler can cleanly optimize and merge the custom layer graph routing hooks into high-throughput device execution streams.

---

## Phase 2: High-Speed Frontier Ingestion (`extract_frontier_data.py`)

1. **Multi-Domain Ingestion Setup:** Construct the data streaming framework to load the three definitive target blocks (Curated Frontier Traces, Massive General Knowledge Backbone, and Code Syntax/Mechanics) directly from Hugging Face.
2. **Streaming Ram Countermeasures:** Enforce `streaming=True` on all `load_dataset` declarations to stream records dynamically into host memory, discarding them immediately post-vectorization to keep disk space and active RAM usage baseline-flat.
3. **Polymorphic Target Parsing:** Execute a structural normalization mapping function to transparently unpack incoming dataset footprints (`messages`, `conversations`, `instruction/output`) down to uniform `(prompt, response)` strings.
4. **XPU Acceleration Embedding Vectorization:** Process the prompt strings using the `Qwen/Qwen2.5-7B-Instruct` vocabulary on the host CPU. Concurrently, route the targeted responses through `BAAI/bge-large-en-v1.5` executing natively on the XPU to cache 1024-dimensional $L_2$-normalized continuous concept handles.
5. **Resilient Local Sharding:** Package the processed unpadded token sequences and target concepts into uniform sequence blocks, dumping out compressed `.pt` shards to the local storage environment every 1,000 clean entries.

---

## Phase 3: The Hybrid Training Loop (`train_jepa_world_model.py`)

1. **Chunked State-Passing (TBPTT Execution):** Establish an unpadded streaming dataloader that segments incoming long-context sequences into deterministic `4096`-token processing chunks.
2. **State Decoupling Routine:** Pass the internal recurrent Mamba2 states across sequential block processing operations. Prior to compiling gradients for Chunk $N+1$, invoke `h = h.detach()` to cap autograd memory overhead at a flat $O(1)$ window relative to total sequence length.
3. **Tri-Partite Loss Evaluation:** Calculate the exact combined update pressure at every gradient step:
* **$\mathcal{L}_{CE}$:** Cross-Entropy prediction errors mapped over the `qwen_tokens`.
* **$\mathcal{L}_{JEPA}$:** `F.cosine_embedding_loss` tracking the distance between the projection head and the continuous `target_concept`.
* **$\mathcal{L}_{Route}$:** A Router Z-loss penalty enforcing uniform load balancing across the ALGR block up to a strict maximum loop ceiling ($\text{max\_loops} = 4$).


4. **Dynamic Loss Weight Modulation:** Implement an exponential scaling schedule for the alignment factor $\lambda_{JEPA}(t)$, starting at `0.01` and scaling linearly to `1.0` during the first 10% of global optimization steps to protect the core networks from structural latent space collapse.
5. **Gradient Accumulation Loop:** Map a single sequence chunk to device memory at any isolated point, accumulating gradient updates across 16 steps before executing the optimizer step.

---

## Phase 4: Native Inference Integration (Latent Runtime Engine)

1. **C++17 Engine Scaffolding:** Instantiate the pure execution architecture for the Latent Runtime Engine (LRE) to bypass bloated, high-overhead runtime dependencies.
2. **Zero-Copy Memory-Mapped Arrays:** Bind the 8B model configuration layers directly into execution space via virtual file allocation maps (`mmap`), allowing active pages to pull directly from fast storage vectors.
3. **SYCL Unified Memory Implementation:** Map the fixed-size `[96, 128, 128]` Mamba2 internal recurrent matrix structures directly into accelerator memory space using native Intel OneAPI primitives, eliminating the overhead of Key-Value data allocation entirely.
4. **ALGR Jump Execution Routing:** Build a clean, low-overhead C++ execution table to manage block traversal based on active router outputs, allowing the runtime to cleanly exit to output projection arrays or scale operations dynamically on the fly.