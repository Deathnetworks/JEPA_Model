# Mamba2-Latent-Loop-4B Engine

This repository contains the implementation codebase for a non-autoregressive, latent-predictive model utilizing Mamba-2 State Space Duality (SSD) blocks and Arbitrary Layer Graph Routing (ALGR). The model runs natively on Intel hardware acceleration (Intel Arc Pro B70 GPU).

## 🤖 Agent Instructions (Jules)
When executing or modifying this codebase, strictly adhere to the constraints and specifications located in the configuration files:
* Core architectural constraints, layers, and tensor shapes: See `docs/ARCHITECTURE_SPEC.md`
* Step-by-step implementation phases and goals: See `docs/EXECUTION_PLAN.md`

### Hard Development Constraints
1. **Hardware Target:** You must exclusively target native Intel GPU compute. Use `device = torch.device("xpu")`. Do NOT use or fall back to CUDA primitives.
2. **Precision:** Utilize simulated 4-bit Quantization Aware Training (QAT) for the core Mamba blocks while maintaining the router layers and decoder heads in FP16/BF16.
3. **No Placeholders:** Write complete python implementations. Do not use inline `pass` blocks or leave comments telling the user to fill in the code later.

## 🛠️ Local Environment Setup

Ensure you are using an elevated Admin PowerShell session within a Windows environment.

```powershell
# Setup virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install native Intel PyTorch extension wheel
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/xpu](https://download.pytorch.org/whl/xpu)
pip install transformers datasets accelerate