# JEPA_Model: 8B Mamba2-JEPA Hybrid Reasoning Engine

An advanced, locally-trained 8B reasoning model built from scratch on the Intel Arc Pro B70 (32GB VRAM). This architecture departs from standard autoregressive transformers by combining a Mamba2 state-space backbone with dynamic layer looping routers and a Joint-Embedding Predictive Architecture (JEPA) alignment head.

Instead of generating synthetic data locally, this pipeline operates via a high-speed featurization engine that streams, extracts, and vectorizes bleeding-edge reasoning traces from frontier models (Claude Opus/Fable, GPT-5.5, Gemini 3.5, Qwen 3.7) to train the student's latent world model.

---

## 🚀 Architectural Overview

### 1. The 8B Student Model

* **Backbone:** Mamba2 State Space Model (SSM) for linear-time sequence processing.
* **Routing:** Dynamic Layer Looping Routers, allowing the model to dynamically allocate compute depth based on token complexity. Includes **Token-Level Active Masking (MoR)** for gating computation updates and **Spectral Injection Constraints** to guarantee spectral radius stability.
* **Alignment Head:** A JEPA projector that maps the pooled router states to a 1024-dimensional continuous latent space, trained via Cosine Embedding loss against offline BGE-Large concept vectors.
* **Decoder:** A `ClosedLoopLatentDecoder` implementing **Delta-JEPA Latent Difference Decoding** and an **FC-RL Foresight Success Estimator** for world model Q-value estimations.

### 2. Hardware Optimization (Intel XPU)

Fully optimized for the **Intel Arc Pro B70 (32GB)** using native upstream PyTorch SYCL/XPU backends.

* **No IPEX Dependency:** Relies strictly on modern `torch` upstream XPU wheels (`--index-url https://download.pytorch.org/whl/xpu`).
* **Memory Management:** Utilizes 8-bit AdamW (`bitsandbytes.optim`), mixed `bfloat16` precision, and gradient accumulation to fit the 8B model and optimizer states comfortably within 32GB VRAM.

---

## 🛠 Pipeline Stages

### Phase 1: Frontier Data Featurization (`extract_frontier_data.py`)

Replaces legacy local generation scripts. This asynchronous script streams top-tier instruction-following, reasoning, and coding datasets directly from Hugging Face.

**Capabilities:**

* Streams massive datasets (`fineweb-edu`, `LMSYS`, `WildChat`, `kernelbench`, etc.) without exhausting local RAM.
* Filters for the highest-quality traces (e.g., `claude-fable-5`, `gpt-5.5`, `qwen3.7-max`).
* Normalizes polymorphic dataset schemas into standard `(prompt, response)` pairs.
* Tokenizes text into unpadded 1D tensors using the highly compressed `Qwen/Qwen2.5-7B-Instruct` vocabulary.
* Executes `BAAI/bge-large-en-v1.5` natively on the XPU to generate 1024-d $L_2$ normalized concept vectors.

**Output Schema:**
Saves chunked `.pt` shards containing dictionaries of variable-length 1D tensors:

```python
{
    "input_tokens": torch.Tensor,   # Qwen-tokenized prompt
    "qwen_tokens": torch.Tensor,    # Qwen-tokenized frontier response
    "target_concept": torch.Tensor  # 1024-d BGE representation
}

```

### Phase 2: Hybrid Training (`train_latent_loop.py` & `train_decoder.py`)

Trains the 8B student from scratch using a specialized objective to handle "infinite" reasoning traces, broken into a two-stage process.

**Core Mechanisms:**

1. **Truncated BPTT (Chunked State-Passing):** Ingests massive traces (up to 64k+ tokens) by chunking sequences to 4096 tokens. The Mamba2 hidden state ($h_t$) is detached and passed across chunk boundaries, providing infinite context capacity without VRAM explosions.
2. **Tri-Partite Loss Function (`train_latent_loop.py`):**
* $\mathcal{L}_{CE}$: Next-token Cross-Entropy loss on the sequence.
* $\mathcal{L}_{JEPA}$: Cosine Embedding loss forcing the internal latent representation to match the BGE `target_concept`.
* $\mathcal{L}_{Route}$: A Z-loss penalty to prevent the dynamic looping routers from collapsing or exceeding maximum loop thresholds.
3. **Delta-JEPA Latent Difference Decoding (`train_decoder.py`):** Reconstructs the code structure from the geometric displacement of thoughts rather than heavy contrastive matrices.

### Phase 3: GRPO & RLVR Fine-Tuning (`GRPOLearningEngine.py`)

A Group Relative Policy Optimization (GRPO) loop combined with Reinforcement Learning from Verifiable Rewards (RLVR). 

**Capabilities:**
* Evaluates generated outputs via a deterministic compilation/execution check (e.g., `rustc`).
* Uses a DAPPO Router Policy Optimization that penalizes excessive computational loops, driving efficiency.
* Avoids memory-heavy Critic neural networks, utilizing pure verifiable performance metrics.

---

## ⚙️ Installation & Setup

### Requirements

* Intel Arc Pro B70 (or equivalent XPU with 32GB+ VRAM)
* Linux / WSL2 environment
* Python 3.10+

### Environment Setup

Install the native PyTorch XPU stack and required dependencies:

```bash
# Install PyTorch XPU Native Wheels
pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# Install core dependencies
pip install transformers datasets accelerate bitsandbytes

```

---

## 🏃 Usage

### 1. Extract Frontier Data

Run the extraction script to begin pulling and vectorizing data. This will create `.pt` shards in your `data/` directory.

```bash
python extract_frontier_data.py

```

*Note: This process streams data. Interrupting it via `Ctrl+C` is safe; it chunks saves every 1,000 samples and will resume appropriately.*

### 2. Train the JEPA World Model

Execute the training loop for the latent core. Ensure your `.pt` data shards are available in the target directory.

```bash
python src/train_latent_loop.py
```

### 3. Train the Decoder

Train the Closed-Loop Latent Decoder on the extracted concepts.

```bash
python src/train_decoder.py
```