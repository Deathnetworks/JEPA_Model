# Migration Path: PyTorch to Intel OpenVINO (Native XPU)

## The Challenge

Migrating the **Mamba2-Latent-Loop-8B** to OpenVINO presents two distinct challenges compared to standard Transformer models:

1. **Dynamic Control Flow:** The Arbitrary Layer Graph Routing (ALGR) utilizes dynamic jumps and conditional loops, whereas OpenVINO traditionally optimizes static execution graphs.
2. **Stateful Matrix Persistence:** Unlike Transformers that append to a Key-Value cache, Mamba2 maintains a fixed recurrent matrix ($h_t$). This state must persist across inference calls without constantly copying data back and forth to the host CPU.

## Step 1: Model Preparation (Stateful Refactoring)

Before exporting, the PyTorch model's `forward` pass must explicitly expose the Mamba2 recurrent state as an input and output.

Refactor the execution wrapper to accept the previous state and return the updated state. Avoid purely data-dependent Python `while` loops; instead, rely on PyTorch 2.x `torch.cond` or `torch.export` which natively capture dynamic control flow.

```python
# Refactored for Stateful Tracing
def forward(self, input_tokens, h_prev):
    # h_prev shape: [Batch, 96, 128, 128]
    logits, h_next = self.mamba2_algr_core(input_tokens, h_prev)
    return logits, h_next

```

## Step 2: Direct OpenVINO Conversion (Bypassing ONNX)

OpenVINO 2024.x+ natively ingests PyTorch models, completely eliminating the need for the ONNX intermediate step. You can trace the model directly using `ov.convert_model`.

```python
import torch
import openvino as ov

# 1. Instantiate dummy inputs for the trace
dummy_tokens = torch.zeros((1, 4096), dtype=torch.long)
dummy_state = torch.zeros((1, 96, 128, 128), dtype=torch.bfloat16)

# 2. Convert PyTorch model natively
ov_model = ov.convert_model(
    model, 
    example_input=(dummy_tokens, dummy_state)
)

```

### Applying Stateful Transformations

To prevent the host application from manually passing the `[96, 128, 128]` tensor back and forth during every step, OpenVINO allows you to fuse the state internally using the `MakeStateful` transformation:

```python
from openvino.runtime.passes import MakeStateful

# Maps the output state tensor directly back to the input state tensor internally
MakeStateful({"h_prev": "h_next"}).run_on_model(ov_model)

```

## Step 3: Weight Compression (NNCF)

An 8B model stored in 16-bit precision requires ~16GB of VRAM. To maximize the execution speed and memory bandwidth on the Intel Arc Pro B70, compress the weights to INT8 or INT4 using the Neural Network Compression Framework (NNCF).

```python
import nncf

# Compress weights to INT8 to halve the memory footprint to ~8GB
compressed_ov_model = nncf.compress_weights(
    ov_model, 
    mode=nncf.CompressWeightsMode.INT8_ASYM
)

# Save the optimized Intermediate Representation (IR) files (.xml / .bin)
ov.save_model(compressed_ov_model, "openvino_models/mamba2_8b_int8.xml")

```

## Step 4: Native C++ / Python Inference API

Write a lightweight OpenVINO inference script using the `openvino.runtime` API.

Because we applied the `MakeStateful` transformation, the OpenVINO runtime manages the Mamba2 state automatically.

```python
import openvino as ov

core = ov.Core()

# Target the Arc B70 natively via SYCL execution
compiled_model = core.compile_model("openvino_models/mamba2_8b_int8.xml", device_name="GPU")

# The state is automatically held in GPU memory. 
# You only need to pass the incoming text tokens.
infer_request = compiled_model.create_infer_request()

# Execute Chunk 1
outputs_1 = infer_request.infer({0: tokens_chunk_1})

# Execute Chunk 2 (State from Chunk 1 is automatically applied)
outputs_2 = infer_request.infer({0: tokens_chunk_2})

# Reset the state when processing a completely new user prompt
infer_request.reset_state()

```