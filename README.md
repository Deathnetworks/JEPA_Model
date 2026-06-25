# Mamba2-Latent-Loop-8B Engine

This repository contains an 8 Billion parameter, non-autoregressive, latent-predictive reasoning engine utilizing **Mamba-2 State Space Duality (SSD)** and **Arbitrary Layer Graph Routing (ALGR)**. 

Optimized for native Intel XPU compute (specifically the Intel Arc Pro B70), this engine bypasses traditional Autoregressive (AR) generation bottlenecks. It thinks entirely in abstract mathematical concepts within a continuous latent space before translating the final resolved thought into text. 

The engine is trained via mathematical knowledge distillation from the highly concise and reasoning-enhanced `Jackrong/Qwopus3.6-27B-v2` teacher model.

---

## 🧠 Architecture Overview

* **Parameter Count:** ~8.2 Billion
* **Core Engine:** 32 Mamba-2 SSD Blocks. Eliminates the $O(N^2)$ attention matrix explosion, allowing massive codebases to be ingested within a fixed memory footprint.
* **The Graph Router (ALGR):** Every block contains an independent micro-router. Instead of traversing layers sequentially, the model mathematically evaluates its state and dynamically loops back, repeats, or skips layers until it resolves the optimal latent concept (up to a 64-hop computational budget).
* **Dual-Stage Latent Speculative Decoder:** 1. *NAR Draft Canvas:* Projects the 1024-dimensional latent concept simultaneously across a 256-token canvas.
  2. *Causal Sweep:* A 2-layer autoregressive transformer that sweeps the canvas to strictly enforce causal code syntax without regenerating the core reasoning steps.
* **Continuous Resumption:** Utilizes Hugging Face `accelerate` for DeepSpeed/ZeRO-3 CPU offloading to fit 8B training optimizer states into 32GB VRAM. Automatically saves to and resumes from `checkpoint_latest/`.

---

## 🛠️ Hardware & Environment Requirements

This codebase is strictly built for **Native Intel XPU Compute** on Windows. It actively forbids NVIDIA CUDA primitives.

* **Required GPU:** Intel Arc Pro B70 (or equivalent Intel dGPU with 32GB+ VRAM).
* **OS:** Windows 10/11
* **Shell:** Elevated Admin PowerShell
* **RAM:** High-capacity system RAM is required to support optimizer state offloading during the 8B training loop.

### 1. Initialize the Environment
```powershell
# Create and activate the virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch natively optimized for Intel GPUs
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/xpu](https://download.pytorch.org/whl/xpu)

# Install supporting ML libraries and native Intel extensions
pip install transformers datasets accelerate huggingface_hub intel_extension_for_pytorch

```

### 2. Verify Hardware Acceleration

Run the test script to ensure PyTorch recognizes the full 33.46 GB VRAM of the Arc B70.

```powershell
python tests/check_xpu.py

```

---

## 🚀 Execution Pipeline (The Orchestrator)

The entire dataset caching, distillation, and curriculum training process is automated via the master PowerShell script. Ensure you have at least 80GB to 100GB of free disk space for the Hugging Face model caches and the extracted `.pt` binary matrices.

Execute the pipeline natively:

```powershell
.\run_pipeline.ps1

```

### Pipeline Stages

* **Stage 0 (Pre-flight Downloader):** Connects to the Hugging Face Hub to cache the teacher (`Jackrong/Qwopus3.6-27B-v2`), the Concept Encoder (`BAAI/bge-large-en-v1.5`), and the Curriculum datasets to local disk.
* **Stage 1 (Dataset Preparation & Tagging):** Parses the instruction datasets, segments them for the curriculum, and chunks tokens into strict `[Batch, 2048]` memory-mapped `.pt` cache files (`logic_`, `agentic_`, `creative_`).
* **Stage 2 (Teacher Distillation):** Loads Qwopus in 4-bit NF4 double-precision alongside the BGE Concept Encoder. Extracts Qwopus's dense reasoning and encodes it into 1024-dimensional Target Concept Vectors.
* **Stage 3 (JEPA Loop Curriculum Training):** Trains the 8B Mamba-2 Engine using ZeRO-3 offloading. Runs sequentially through three distinct curriculum phases:
* **3A (Logic):** Trains on polyglot coding logic (`Magicoder-OSS-Instruct-75K`).
* **3B (Agentic):** Trains on strict tool-use triggers (`Agentic-Chain-of-Thought` & `Agentic-Reasoning`).
* **3C (Creative):** Trains on fluid prose formatting (`OpenHermes-2.5`).


* **Stage 4 (Decoder Training):** Freezes the Mamba engine blocks and symbiotically trains the Dual-Stage Decoder to map the latent concepts back into strict multi-language syntax.
* **Stage 5 (Autonomous Inference):** Launches the agent loop. Features a local `rustc` subprocess compiler that verifies outputs and routes errors mathematically back into the latent engine for self-correction.

---

## 📂 Repository File Mapping

* `run_pipeline.ps1`: The master PowerShell orchestrator.
* `src/download_models.py`: Pre-fetches models and datasets to avoid VRAM-crashing network timeouts.
* `src/dataset_preparation.py`: Formats and prefixes SFT data for Curriculum Learning.
* `src/teacher_distillation.py`: Memory-aggressive script bridging the 27B teacher to the Concept Encoder.
* `src/model_architecture.py`: The master PyTorch module housing the 8B Mamba loops, state embeddings, ALGR routers, and Dual-Stage decoders.
* `src/train_latent_loop.py`: The native XPU QAT training loop utilizing `accelerate` offloading and auto-resumption.
* `src/train_decoder.py`: The XPU training loop for the text translation heads.
* `src/inference_harness.py`: The final deployment agent featuring automatic compilation and feedback loops.
* `tests/check_xpu.py`: Hardware environment verifier.
* `tests/test_shapes.py`: Dimensionality and tensor boundary verifications for the 8B engine.
* `docs/ARCHITECTURE_SPEC.md`: Immutable tensor math, dimensions, and layer constraints.
* `docs/EXECUTION_PLAN.md`: Phase-by-phase development blueprints.

```

```