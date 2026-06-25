import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"XPU Available: {torch.xpu.is_available()}")

if torch.xpu.is_available():
    print(f"Device Name: {torch.xpu.get_device_name(0)}")
    print(f"Total VRAM: {torch.xpu.get_device_properties(0).total_memory / 1e9:.2f} GB")