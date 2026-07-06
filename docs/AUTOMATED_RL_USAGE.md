# Automated Reinforcement Learning Pipeline

This document explains the usage and inner workings of `src/train_rl_loop.py`, which is the orchestration script for executing the Group Relative Policy Optimization (GRPO) loop over the Mamba-JEPA model.

## 🚀 Overview
The RL training loop leverages test-time optimization principles combined with verifier-driven rewards. Instead of relying purely on large language model (LLM) critics, this system dynamically streams Hugging Face multi-language instruction sets and utilizes real system tools (like `rustc`) to generate deterministic reward signals.

## 🛠 Features

1. **Continuous Checkpointing**: Uses `accelerator.load_state()` and `accelerator.save_state()` allowing you to seamlessly restart training. State metadata (epochs, global steps) is tracked in `checkpoint_rl/rl_training_state.txt`.
2. **Streamed Datasets**: Fetches dynamic streams directly from datasets like `ise-uiuc/Magicoder-OSS-Instruct-75K` using the native `.skip()` mechanism to maintain maximum performance without RAM exhaustion.
3. **8-bit Optimizers**: Deploys `bitsandbytes.optim.AdamW8bit` specifically optimized for the 32GB constraints of Intel XPU setups.
4. **XPU Empty Cache Control**: Actively flattens VRAM curves during group generation processes via explicit garbage collection inside the iteration loop.

## 🏃 Running the Training Loop

Start the training script by simply invoking it via Python:

```bash
python src/train_rl_loop.py \
    --epochs 1 \
    --dataset ise-uiuc/Magicoder-OSS-Instruct-75K \
    --group_size 4
```

### Parameters & Optimal Usage
*   `--epochs` (default: 1): Number of total passes across the streamed generator.
    *   **Optimal Usage**: 1-2 epochs are recommended since the engine relies on diverse datasets. More passes over the same synthetic prompts may result in diminishing returns.
*   `--dataset` (default: `ise-uiuc/Magicoder-OSS-Instruct-75K`): Hugging Face path to the dataset used for prompt curriculum.
    *   **Recommended Datasets**:
        *   For general code problem translation and logic reasoning: `nuprl/MultiPL-E`
        *   For varied real-world tasks and HTML/API creation: `ise-uiuc/Magicoder-OSS-Instruct-75K`
        *   For advanced coding evaluation / Python logic: `google-research-datasets/mbpp`
        *   For competitive advanced creative reasoning: `deepmind/code_contests`
*   `--group_size` (default: 4): The number of independent samples drawn per step for the GRPO advantage calculation.
    *   **Optimal Usage**: The optimal group size for Intel B70 32GB GPU is 4. This size balances sufficient distribution variance to form reliable GRPO Advantage vectors with memory capacity, avoiding OOM events on a single 32GB node.

## 🔍 Code Breakdown

### The DataLoader (`stream_rl_prompts`)
This function dynamically pulls data from Hugging Face via the network and yields row `instruction` prompts. It uses `dataset.skip(start_idx)` for O(1) state resumption during interrupted sessions without creating an internal loop bottleneck.

### `train_rl_loop` Initialization
Here the Hugging Face `Accelerator` and Intel `XPU` settings are invoked. The 8B base state `MambaJEPAEngine` and `ClosedLoopLatentDecoder` weights are loaded from `jepa_engine.pth` and `latent_decoder.pth`.

```python
    model = MambaJEPAEngine()
    decoder = ClosedLoopLatentDecoder()
```
*Note: If the code detects a CPU environment, the hyper-parameters drop significantly to avoid immediate out-of-memory errors.*

### The Training Step
For each prompt in the dataset generator, the script generates a multi-dimensional continuous tensor array for inputs.
1. The script utilizes the GRPO Engine (defined in `GRPOLearningEngine.py`).
2. Generates `group_size` (e.g., 4) different outputs.
3. Calculates standard PPO deviations and verifies via real deterministic processes (rustc/python compilers).
4. Emits `torch.autocast(device_type="xpu", dtype=torch.bfloat16)` context execution over the loss generator.
5. Issues an `accelerator.backward()` call combined with a manual `torch.xpu.empty_cache()` step to maintain stability.
