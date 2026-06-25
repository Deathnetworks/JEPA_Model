# Mamba2-Latent-Loop-8B Engine

This repository contains an 8 Billion parameter, non-autoregressive, latent-predictive reasoning engine utilizing **Mamba-2 State Space Duality (SSD)** and **Arbitrary Layer Graph Routing (ALGR)**. 

Optimized for Intel Arc Pro B70 (XPU) via CPU offloading, this engine uses a multi-stage curriculum learning pipeline to perfectly distill complex coding and agentic reasoning from massive teacher models.

## 🚀 Features
* **8B Mamba-2 Core:** Capable of massive context ingestion with linear $O(N)$ scaling.
* **Curriculum Distillation:** Separates logic tracking from syntax generation to prevent polyglot mode collapse.
* **Auto-Resumption:** Built-in fault tolerance. Training can be interrupted and resumed exactly where it left off, recovering all optimizer states from CPU RAM/Disk.
* **Autonomous Compiler Feedback:** The inference harness compiles its own Rust/C++ outputs and routes errors mathematically back into the latent engine for self-correction.

---

## 🧠 Architecture Overview

Standard LLMs suffer from Autoregressive (AR) generation bottlenecks. This engine solves that by thinking entirely in abstract mathematical concepts before translating the final thought into text.

* **Parameter Count:** ~8 Billion (Simulated 4-bit precision base)
* **Core Engine:** 24 Mamba-2 SSD Blocks. Eliminates the $O(N^2)$ attention matrix explosion, allowing massive codebases to be ingested within a fixed memory footprint.
* **The Graph Router:** Every block contains a micro-router. Instead of passing through layers sequentially, the model mathematically evaluates its state and dynamically loops back, repeats, or skips layers until it finds the optimal latent concept.
* **Dual-Stage Latent Speculative Decoder:** 1. *NAR Draft Canvas:* Projects the 1024-dimensional latent concept simultaneously across a 256-token canvas.
  2. *Causal Sweep:* A 2-layer autoregressive transformer that sweeps the canvas to strictly enforce causal code syntax without regenerating the reasoning steps.
* **Autonomous Feedback:** The inference harness includes a local `rustc` subprocess compiler. If the model generates broken code, the compiler error is fed directly back into the latent loop for autonomous architectural repair.

---

## 🛠️ Hardware & Environment Requirements

This codebase is strictly built for **Native Intel XPU Compute** on Windows. It actively forbids NVIDIA CUDA primitives. 

* **Required GPU:** Intel Arc Pro B70 (or equivalent Intel dGPU with 32GB+ VRAM).
* **OS:** Windows 10/11
* **Shell:** Elevated Admin PowerShell

### 1. Initialize the Environment
```powershell
# Create and activate the virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch natively optimized for Intel GPUs
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/xpu](https://download.pytorch.org/whl/xpu)

# Install supporting ML libraries
pip install transformers datasets accelerate huggingface_hub

```

### 2. Verify Hardware Acceleration

Run the test script to ensure PyTorch recognizes the full 33.46 GB VRAM of the Arc B70.

```powershell
python tests/check_xpu.py

```

---

## 🚀 Execution Pipeline (The Orchestrator)

The entire training and distillation process is fully automated via `run_pipeline.ps1`. Ensure you have at least 80GB of free disk space for the Hugging Face model caches and the `.pt` binary matrices.

Execute the pipeline natively:

```powershell
.\run_pipeline.ps1

```

### Pipeline Stages

* **Stage 0 (Pre-flight Downloader):** Connects to Hugging Face to cache `Qwen/Qwen3.6-27B`, `BAAI/bge-large-en-v1.5`, and the agentic SFT datasets to local disk to prevent network timeouts.
* **Stage 1 (Dataset Preparation):** Parses the JSONL datasets, handles continuous sequence padding, and chunks the tokens into `[Batch, 2048]` memory-mapped `.pt` files.
* **Stage 2 (Teacher Distillation):** Loads Qwen in 4-bit precision alongside the BGE Concept Encoder. Extracts Qwen's text and encodes it into 1024-dimensional Target Concept Vectors.
* **Stage 3 (JEPA Loop Training):** Trains the 8B Mamba-2 Engine using Quantization Aware Training (QAT). Optimizes for Latent Alignment (MSE + Cosine Similarity) while penalizing excessive routing loops.
* **Stage 4 (Decoder Training):** Freezes the Mamba engine and symbiotically trains the Dual-Stage Decoder to translate the mathematical concept vectors back into strict text syntax.
* **Stage 5 (Autonomous Inference):** Launches the agent.

---

## 📂 Repository File Mapping

* `run_pipeline.ps1`: The master pipeline orchestrator.
* `src/download_models.py`: Safely fetches models and datasets without VRAM loads.
* `src/dataset_preparation.py`: Converts raw SFT logic into properly padded token matrices.
* `src/model_architecture.py`: The master PyTorch module housing the Mamba loops, state embeddings, ALGR routers, and Dual-Stage decoders.
* `src/teacher_distillation.py`: Memory-aggressive script bridging the 27B teacher to the Concept Encoder.
* `src/train_latent_loop.py`: The native XPU QAT training loop for the core engine.
* `src/train_decoder.py`: The XPU training loop for the text translation heads.
* `src/inference_harness.py`: The final deployment agent. Features regex code extraction and automated `subprocess` compiler feedback loops.
* `docs/ARCHITECTURE_SPEC.md`: Immutable tensor math and layer constraints.
* `docs/EXECUTION_PLAN.md`: Phase-by-phase development blueprints.

```