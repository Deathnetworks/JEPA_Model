# PART 2: Master Execution Plan

```
┌──────────────────────────────────┐      ┌──────────────────────────────────┐
│   Phase 1: Environment & Arch    │ ───► │  Phase 2: Teacher Distillation   │
│   - Install Native XPU Backend   │      │  - Extract Target Vector Maps    │
│   - Instantiate 8B Mamba Modules │      │  - Cache PyTorch Binaries (.pt)  │
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
1. **Repository Setup:** Verify Intel XPU access.
2. **Accelerate Initialization:** Configure Hugging Face `accelerate` for CPU offloading of optimizer states to accommodate the 8B parameters.
3. **Hardware Targeting:** Enforce native Intel graphics acceleration. Jules must ensure all data-loading tensors explicitly target `device = torch.device("xpu")`. No CUDA dependencies or wrappers are permitted.
4. **Sanity Check Execution:** Before launching heavy loops, run the hardware verification script to confirm access to the full 33.46 GB allocation on the Intel Arc Pro B70.

## Phase 2: Teacher Distillation & Target Generation

1. **Ingest Source Datasets:** Execute `dataset_preparation.py` to stream code-reasoning repositories (`AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset` and `TheAgenticAI/Agentic-Reasoning`).
2. **Dataset Tagging:** Instead of dumping all data into one matrix, `dataset_preparation.py` segments data into `logic_set.pt` and `agentic_set.pt`.
3. **Concept Extraction:** Extract teacher responses and encode them into mathematical targets, caching them to disk.
4. **Establish Latent Targets:** Map Qwen's text representations through a dense projection layer down to a frozen `1024` dimensions to serve as our target concept vectors ($Y_{target}$). Cache the resulting tensors directly to disk inside `F:\JEPA_Model\data\` as binary `.pt` files to eliminate redundant processing during training.

## Phase 3: Engine Training (The JEPA Loop)

1. **Resumption Protocol:** On startup, the script searches for `checkpoint_latest/`. If found, it loads weights, optimizer states, and the specific curriculum phase/epoch.
2. **Instantiate 4-bit QAT Configuration:** Initialize the 8B Mamba-2 weights using simulated 4-bit quantization weights. The micro-routers, layer normalization matrices, and tracking heads must remain pinned at native FP16/BF16 precision to avoid gradient degradation.
3. **Forward Graph Routing Execution:** Pass the processed source text arrays into the model. Track layer traversal via a tracking matrix, incrementing values at every routing jump.
 - **Sub-Phase 3A (Logic Training):** Train the 8B Mamba Engine exclusively on `logic_set.pt`. 
 - **Sub-Phase 3B (Agentic Training):** Shift training to `agentic_set.pt`, heavily penalizing excessive routing loops via the efficiency loss parameter to force fast tool-use decisions.
4. **Calculate Compound Objective Metrics:** Optimize the parameters by combining three distinct error tracking steps:
* **Latent Alignment Loss:** $\text{MSE}(H_{final}, Y_{target}) + (1.0 - \text{CosineSimilarity}(H_{final}, Y_{target}))$
* **Efficiency Regularization:** Introduce a penalty factor scaled by $\gamma = 0.001$ against total global sequence hops to incentivize swift execution paths.
* **Backpropagation:** Run standard parameter optimization updates based on the cumulative loss values.



## Phase 4: Decoder Symbiosis & Harness Integration

1. **Freeze Core Blocks:** Freeze all internal weights inside the 32 Mamba-2 layers and their corresponding micro-routers.
2. **Train Speculative Decoding:** Train the Stage 1 and Stage 2 decoders to map the perfect latent concepts to precise programming syntax constraints (Rust, C++, etc.).
3. **Execution Harness Connection:** Pipe the text canvas directly into a local file manipulation handler. Write out the generated Rust files and pass them directly to an automated shell command running `cargo check`.
4. **Runtime Error Looping:** If the compilation process returns syntax errors, read the error stream directly back into the model's primary input stream. This prompts the graph router to loop through additional physical blocks to identify structural fixes without forcing the user to manually engineer a prompt correction.

## Phase 5: Autonomous Inference Harness
1. **Execution Loop:** Run the pipeline. Extract output, compile via subprocess, and route errors back to the model's latent state.

---