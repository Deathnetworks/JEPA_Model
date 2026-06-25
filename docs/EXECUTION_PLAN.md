# PART 2: Master Execution Plan

```
┌──────────────────────────────────┐      ┌──────────────────────────────────┐
│   Phase 1: Environment & Arch    │ ───► │  Phase 2: Teacher Distillation   │
│   - Install Native XPU Backend   │      │  - Extract Target Vector Maps    │
│   - Instantiate 4B Mamba Modules │      │  - Cache PyTorch Binaries (.pt)  │
└──────────────────────────────────┘      └──────────────────────────────────┘
                                                            │
                                                            ▼
┌──────────────────────────────────┐      ┌──────────────────────────────────┐
│   Phase 4: Decoder & Harness     │ ◄─── │     Phase 3: The JEPA Loop       │
│   - Train Speculative Overwrite  │      │  - Simulated 4-bit QAT Training  │
│   - Live Rust/Cargo Verifications│      │  - Apply Alignment/Route Losses  │
└──────────────────────────────────┘      └──────────────────────────────────┘

```

## Phase 1: Environment & Architecture Initialization

1. **Repository Setup:** Initialize the GitHub repository containing `model_architecture.py`, `dataset_preparation.py`, `teacher_distillation.py`, and `train_latent_loop.py`.
2. **Hardware Targeting:** Enforce native Intel graphics acceleration. Jules must ensure all data-loading tensors explicitly target `device = torch.device("xpu")`. No CUDA dependencies or wrappers are permitted.
3. **Sanity Check Execution:** Before launching heavy loops, run the hardware verification script to confirm access to the full 33.46 GB allocation on the Intel Arc Pro B70.

## Phase 2: Teacher Distillation & Target Generation

1. **Ingest Source Datasets:** Execute `dataset_preparation.py` to stream code-reasoning repositories (`AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset` and `TheAgenticAI/Agentic-Reasoning`).
2. **Extract Teacher Representations:** Use a resource-optimized instance of `Qwen/Qwen3.6-27B` to process incoming software prompt code. Collect both the final text answers and their respective sentence-level mathematical profiles.
3. **Establish Latent Targets:** Map Qwen's text representations through a dense projection layer down to a frozen `1024` dimensions to serve as our target concept vectors ($Y_{target}$). Cache the resulting tensors directly to disk inside `F:\JEPA_Model\data\` as binary `.pt` files to eliminate redundant processing during training.

## Phase 3: Engine Training (The JEPA Loop)

1. **Instantiate 4-bit QAT Configuration:** Initialize the 4B Mamba-2 weights using simulated 4-bit quantization weights. The micro-routers, layer normalization matrices, and tracking heads must remain pinned at native FP16/BF16 precision to avoid gradient degradation.
2. **Forward Graph Routing Execution:** Pass the processed source text arrays into the model. Track layer traversal via a tracking matrix, incrementing values at every routing jump.
3. **Calculate Compound Objective Metrics:** Optimize the parameters by combining three distinct error tracking steps:
* **Latent Alignment Loss:** $\text{MSE}(H_{final}, Y_{target}) + (1.0 - \text{CosineSimilarity}(H_{final}, Y_{target}))$
* **Efficiency Regularization:** Introduce a penalty factor scaled by $\gamma = 0.001$ against total global sequence hops to incentivize swift execution paths.
* **Backpropagation:** Run standard parameter optimization updates based on the cumulative loss values.



## Phase 4: Decoder Symbiosis & Harness Integration

1. **Freeze Core Blocks:** Freeze all internal weights inside the 24 Mamba-2 layers and their corresponding micro-routers.
2. **Train Speculative Decoding:** Present the frozen concept states to the Dual-Stage Decoder. Optimize cross-entropy outputs until Stage 1 and Stage 2 can reproduce identical target text sequences matching the teacher's original vocabulary structures.
3. **Execution Harness Connection:** Pipe the text canvas directly into a local file manipulation handler. Write out the generated Rust files and pass them directly to an automated shell command running `cargo check`.
4. **Runtime Error Looping:** If the compilation process returns syntax errors, read the error stream directly back into the model's primary input stream. This prompts the graph router to loop through additional physical blocks to identify structural fixes without forcing the user to manually engineer a prompt correction.

---