# Architectural Specification: Latent Runtime Engine (LRE)

## I. Core Philosophy
The Latent Runtime Engine (LRE) is a C++17 bare-metal inference runtime designed explicitly for Non-Autoregressive, Graph-Routed Mamba-2 models. 

Unlike `llama.cpp` (which is bottlenecked by sequential Key-Value cache updating for standard Transformers), LRE focuses on **State Matrix Augmentation** and **Dynamic Layer Jumps**.

## II. System Architecture Overview

### 1. The Frontend (C++ / Python Bindings)
* Handles Tokenization (SentencePiece/BPE implementation).
* Exposes an asynchronous API to pass text/tokens to the runtime.
* Returns raw text strings back to the user.

### 2. The Execution Graph (The ALGR Director)
This is the "Brain" of the C++ engine. 
* It does NOT execute layers 1 to N sequentially.
* It maintains a `RuntimeState` struct (containing the 6144-d continuous vector).
* It executes Block $N$, evaluates the Softmax Logit from the micro-router, and uses a C++ `switch` statement to dispatch the tensor to the memory address of Block $N+1$, Block $N-k$, or the exit node.

### 3. Hardware Abstraction Layer (HAL) & Compute Backends
The runtime must swap compute kernels dynamically at compile time or runtime. 

#### Backend Implementations:
1.  **SYCL / DPC++ (Primary):** * Native support for Intel Arc (XPU) and discrete accelerators. Uses Intel OneAPI Math Kernel Library (oneMKL) for extreme matrix multiplication speed.
2.  **CUDA (NVIDIA):** * `cuBLAS` implementation for the heavy Mamba 1D convolutions and SSD state scans.
3.  **HIP / ROCm (AMD):** * Source-to-source translation from CUDA. 
4.  **Metal (Apple Silicon):** * Utilizing Metal Performance Shaders (MPS) to execute the SSD parallel scan algorithm on Unified Memory.
5.  **Vulkan Compute (Universal/Fallback):** * Uses Vulkan compute shaders for generic execution on ARM NPUs (Snapdragon, edge devices) or unsupported discrete GPUs.

## III. Memory Management (Zero-Copy Paradigm)
* **Weight Mmap:** The 8B parameters are memory-mapped (`mmap`) directly from disk. Only the layers currently executing in the ALGR loop are paged into VRAM/Unified Memory.
* **No KV Cache:** Because Mamba-2 is a State Space Model, we do not append tokens to a growing cache. We update a fixed-size `[96, 128, 128]` state matrix. Memory usage remains constant regardless of whether the prompt is 10 words or 100,000 words.

## IV. The Dual-Stage Decoder Pipeline in C++
Once the ALGR loop finishes, the `RuntimeState` vector is passed to the Decoder struct.
1.  **Draft Kernel:** A massive, highly parallel matrix multiplication that projects `[1024]` -> `[256 * 6144]` in a single GPU dispatch.
2.  **Sweep Kernel:** A custom causal attention kernel that performs exactly two passes over the 256 tokens.
3.  **Argmax Extraction:** CPU pulls the final IDs and decodes them to string.

## V. Future Roadmap
* **Phase 1:** Build CPU-only math primitives (reference implementation).
* **Phase 2:** Implement SYCL Backend for Intel Arc validation.
* **Phase 3:** Write the ALGR Director logic for dynamic layer jumping.
* **Phase 4:** Add CUDA/Metal backends via CMake toggles.
* **Phase 5:** Implement INT4 / INT8 weight quantization loaders.