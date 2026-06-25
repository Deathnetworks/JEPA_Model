# Migration Path: PyTorch to Intel OpenVINO

## The Challenge
Standard Autoregressive (AR) models are easily converted to OpenVINO. However, our **Arbitrary Layer Graph Routing (ALGR)** uses dynamic `while` loops that break standard JIT tracing. OpenVINO handles static graphs best.

## Step 1: Model Preparation (Eliminating Data-Dependent Python Loops)
Before exporting, the PyTorch model's `forward` pass must be refactored to use `torch.cond` or `torch.where` instead of Python `while` loops, or traced using PyTorch 2.0's `torch.export` (PT2 Export) which captures dynamic control flows better than the legacy `torch.jit.trace`.

## Step 2: ONNX / PT2 Export
1. **Export the Mamba Engine:**
   ```python
   import torch
   # Export using TorchScript (Scripting, not tracing, due to the while loop)
   scripted_engine = torch.jit.script(engine)
   torch.onnx.export(scripted_engine, input_tokens, "mamba_engine.onnx", opset_version=17)

```

2. **Export the Decoder:**
The decoder is a standard transformer and can be easily traced to ONNX.

## Step 3: OpenVINO Model Optimizer (MO)

Use the OpenVINO toolkit to convert the `.onnx` files into the OpenVINO Intermediate Representation (IR) `.xml` and `.bin` files.

```bash
ovc mamba_engine.onnx --compress_to_fp16 --output_dir openvino_models/
ovc decoder.onnx --compress_to_fp16 --output_dir openvino_models/

```

## Step 4: Native C++ / Python Inference API

Write a lightweight OpenVINO inference script using the `openvino.runtime` API.

* Load the IR models.
* Target the `GPU` device (which maps directly to your Arc B70).
* OpenVINO's runtime will automatically compile the execution graph into optimized SYCL/Level-Zero machine code for the Intel GPU.