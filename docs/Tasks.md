# JEPA_Model v2 Task Diary

## Instructions for AI Agents
**READ THIS BEFORE MODIFYING THIS DOCUMENT**
This document serves as a persistent task diary, todo list, and planning ledger for the JEPA_Model v2 transition.
- **Do not overwrite or delete completed tasks.** Instead, use strikethrough (`~~task~~`) to mark them as completed.
- Maintain a high level of detail, including pseudo-code, assumptions, and specific implementation notes.
- If plans change or expand, append new tasks or sub-sections rather than replacing the old ones.
- Keep track of the "why" and "how" we are implementing things here. This is our memory ledger.

---

## High-Level Plan: From Current State to Improved Version ("JEPA_Model v2")

### Phase 1: Foundations
- [x] ~~**Config & CLI**~~: Implement configuration management using standard YAML and JSON via `argparse` (leaving structure open to migrate to Hydra/OmegaConf if needed later). Move hardcoded paths into configs.
- [x] ~~**Dependencies**~~: Update `requirements.txt`. Add `pytest`, `mamba-ssm`, `datasets`, `transformers`, `accelerate`, `bitsandbytes`, `wandb`, `einops`. Pin versions where appropriate. Create a basic `setup.py` for standard installation.
- [x] ~~**Licensing & Docs**~~: Add an MIT LICENSE file. Expand `README.md` (architecture diagram, quickstart, benchmarks). Update existing `docs/*.md`.
- [x] ~~**Versioning/CI**~~: Setup `.github/workflows/ci.yml` for basic CI (lint, tests on CPU).

### Phase 2: Code Cleanup & Refactoring
- [x] ~~**Modularize Model**~~: Split `src/model_architecture.py` into:
  - `src/model/backbone/`
  - `src/model/routing/`
  - `src/model/heads/`
  - `src/model/decoder/`
  - Use `torch.nn.Module` best practices.
- [x] ~~**Optimize Mamba Core (SYCL/oneAPI)**~~:
  - Implement `OptimizedMambaBlock` using a custom SYCL kernel.
  - Create `src/kernels/sycl_selective_scan.cpp` and `src/kernels/selective_scan_kernel.sycl` ported from CUDA `mamba-ssm` reference.
  - Implement a Python fallback in `OptimizedMambaBlock` for when SYCL kernel isn't available.
- [x] ~~**Routing & Stability**~~: Add gradient clipping per-router. Add monitoring (avg loops, halt rates) and log to `wandb`.
- [x] ~~**JEPA/Decoder**~~: Add ablation flags, tie weights properly, support teacher-forcing.
- [x] ~~**Training Scripts**~~:
  - Refactor `train_latent_loop.py` to use chunking, separate pretrain -> alignment -> RL stages with resume logic.
  - Add validation loop and early stopping.
- [x] ~~**Data Pipeline**~~: Refactor `extract_frontier_data.py` with `datasets` streaming, filtering, and multiprocessing.

### Phase 3: Hardware & Efficiency
- [x] ~~**Multi-Backend**~~: Stub CUDA/ROCm configs, while keeping primary focus on XPU.
- [x] ~~**Memory/Inference**~~: Add `torch.compile` modes, activation checkpointing, mixed precision guards.
- [x] ~~**Pipeline Orchestration**~~: Update `run_pipeline.ps1` to handle the multi-stage training (pretrain, align, RL) gracefully, with ability to resume and exit on errors.

### Phase 4 & 5: Advanced Features, Evaluation, Tests & Packaging
- [x] ~~**Testing**~~: Transition existing tests to `pytest` and add new ones (unit tests for router halting/spectral norms).
- [x] ~~**Evaluation**~~: Add benchmark scripts (GSM8K, HumanEval stubs).
- [x] ~~**Packaging**~~: Hugging Face integration stubs, logging via Wandb, example notebooks.

---

## Detailed Task Implementations (Pseudo-code & Notes)

### SYCL Kernel Port (Phase 2.1)
*Goal*: Port selective scan from `mamba-ssm` (CUDA) to SYCL for Intel Arc Pro B70.
*Pseudo-code snippet for Python Fallback Wrapper*:
```python
import torch
import torch.nn as nn

try:
    from kernels.sycl_selective_scan import sycl_selective_scan
    USE_SYCL_KERNEL = True
except ImportError:
    USE_SYCL_KERNEL = False

class OptimizedMambaBlock(nn.Module):
    def __init__(self, d_model, d_state, use_sycl_kernel=True):
        super().__init__()
        # Check XPU availability alongside the flag
        self.use_sycl_kernel = use_sycl_kernel and USE_SYCL_KERNEL and hasattr(torch, 'xpu') and torch.xpu.is_available()
        # ... init layers ...

    def forward(self, x):
        if self.use_sycl_kernel:
            return sycl_selective_scan(x, ...)
        else:
            # Fallback logic
            return legacy_forward(x)
```
