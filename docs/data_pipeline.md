Here is the updated documentation for the data extraction phase, which would logically sit as `docs/data_pipeline.md` (or similar) in your repository. It explicitly details the shift from local synthetic generation to the high-speed frontier streaming approach.

---

# Data Featurization & Ingestion Pipeline

This document outlines the architecture and execution of `extract_frontier_data.py`, the core ingestion engine for the JEPA 8B world model.

Instead of relying on slow, compute-bound local autoregressive generation, this pipeline streams pre-distilled reasoning traces and instructions from top-tier frontier models directly from Hugging Face. It normalizes, tokenizes, and vectorizes this data using native PyTorch XPU acceleration to create the structural `.pt` shards required for training.

## 🗂️ The Dataset Matrix

The ingestion queue is divided into three highly curated domains to ensure the student model learns advanced reasoning, broad factual grounding, and strict structural mechanics.

### 1. Curated Frontier Traces (Reasoning & Agentic Flow)

Extracts multi-turn logic, tool use, and complex problem-solving directly from leading proprietary and open-weights models.

* **Target Models:** Claude (Opus 4.7/4.8, Fable 5), GPT-5.5, Gemini 3.1 Pro/3.5 Flash, Qwen 3.6/3.7, GLM 5.2.
* **Repositories:** `Complete-FABLE.5-traces-2M`, `gpt-5.5-agent`, `Qwen3.6-35B-A3B-Tool-Calling`, `kernelbench-mega-traces`, and other high-density datastores.

### 2. General Knowledge & Instruction Following

Provides the foundational factual core and aligns the model's conversational flow.

* **Repositories:** `fineweb-edu` (scientific/factual density), `ultrafeedback_clean` (human alignment), `OpenHermes-2.5`, `reasoning-base-20k`.

### 3. Code Mechanics & Grammar Rules

Forces the JEPA latent space to understand abstract syntax trees, compiler logic, and deterministic state changes—critical for low-level system understanding.

* **Repositories:** `starcoder2-instruct`, `python-execution-traces`, `CodeFeedback-Filtered-Instruction`.

## ⚙️ Technical Architecture

### Streaming & Memory Safety

To prevent local storage and RAM exhaustion, the pipeline enforces `streaming=True` across all network calls. Data is processed iteratively in memory. Malformed schemas or network timeouts trigger a safe bypass rather than a hard crash, ensuring multi-hour extraction jobs run cleanly.

### Polymorphic Schema Parsing

Because the datasets originate from disparate organizations, the script utilizes a dynamic schema parser (`extract_text_pair()`). It automatically unifies formats (e.g., `messages`, `instruction/output`, `conversations`) into standard `(prompt, response)` pairs.

### Native XPU Vectorization

The pipeline executes a dual-processing loop optimized for the Intel Arc architecture:

1. **Host CPU (Tokenization):** The `Qwen/Qwen2.5-7B-Instruct` tokenizer compresses the raw text into unpadded 1D `torch.long` tensors.
2. **GPU/XPU (Vectorization):** The target response is piped into `BAAI/bge-large-en-v1.5` executing natively on the XPU. The pipeline extracts the `[CLS]` token, applies $L_2$ normalization, and yields a precise 1024-dimensional continuous latent vector.

## 💾 Output Schema

Data is serialized into uniformly batched `.pt` shards (e.g., `distilled_frontier_chunk_X.pt`) every 1,000 processed samples. This allows the TBPTT training loop to stream data efficiently from the NVMe/SSD without memory spiking.

Each shard contains a dictionary structured as follows:

```python
{
    "input_tokens": torch.Tensor,   # 1D Qwen-tokenized prompt
    "qwen_tokens": torch.Tensor,    # 1D Qwen-tokenized frontier target response
    "target_concept": torch.Tensor  # 1D Normalized BGE representation [1024]
}

```

---