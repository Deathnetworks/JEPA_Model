# Workflow Rules for JEPA_Model

## Project Goals
- Maintain a local 8B reasoning model using Mamba2-JEPA architecture.
- Optimize for Intel Arc Pro B70 (32GB VRAM) via PyTorch XPU backend.
- Decoupled training pipeline (`train_latent_loop.py`, `train_decoder.py`, `GRPOLearningEngine.py`).

## Core Principles
- Memory Efficiency: Enforce chunked state-passing to handle long context.
- Stability: Rely on fixed-point halting and spectral injection constraints.
- Delta-JEPA: Reconstruct structures directly from geometric thought displacement.
- Verifiable Reinforcement Learning: Use compiler execution rather than neural reward models for GRPO fine-tuning.

## "Known Stable" Zones
- `model_architecture.py` (Mamba2LatentLoop8B, ClosedLoopLatentDecoder, MambaJEPAEngine)
- Datasets pipeline defined in `extract_frontier_data.py`.
