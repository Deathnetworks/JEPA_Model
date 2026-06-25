import os
import gc
import glob
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_device():
    """
    Ensure the target device is the Intel Arc Pro B70 GPU (xpu).
    Strictly forbids CUDA primitives.
    """
    try:
        import intel_extension_for_pytorch as ipex
        logging.info("Successfully imported intel_extension_for_pytorch.")
    except ImportError:
        logging.warning("intel_extension_for_pytorch not found. Attempting native 'xpu' device assignment.")

    device = torch.device("xpu")
    logging.info(f"Targeting native Intel GPU compute via device: {device}")
    return device

def clear_memory():
    """
    Aggressive memory management to ensure the 33.46 GB VRAM constraint is respected
    when running a 27B model and an encoder concurrently.
    """
    gc.collect()
    if hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"):
        torch.xpu.empty_cache()

def main():
    device = setup_device()

    input_dir = Path(r"F:\JEPA_Model\data")
    output_dir = Path(r"F:\JEPA_Model\distilled_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Concept Encoder
    encoder_name = "BAAI/bge-large-en-v1.5"
    logging.info(f"Loading Concept Encoder: {encoder_name}")
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    encoder_model = AutoModel.from_pretrained(encoder_name)
    encoder_model.to(device)
    encoder_model.eval()

    # 2. Load Teacher Model (Qwen) in 4-bit Precision
    teacher_name = "Jackrong/Qwopus3.6-27B-v2"
    logging.info(f"Loading Teacher Model: {teacher_name}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)
    if teacher_tokenizer.pad_token_id is None:
        if teacher_tokenizer.eos_token_id is not None:
            teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
        else:
            teacher_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load with 4-bit precision directly to the xpu
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        quantization_config=quantization_config,
        device_map={"": "xpu"},
        trust_remote_code=True
    )
    teacher_model.eval()

    # 3. Process Chunked .pt Files
    pt_files = glob.glob(str(input_dir / "*.pt"))
    if not pt_files:
        logging.warning(f"No .pt files found in {input_dir}. Ensure dataset_preparation.py has been run.")
        return

    batch_size = 4
    max_new_tokens = 256

    for file_path in pt_files:
        logging.info(f"Processing file: {file_path}")
        try:
            # Shape: [chunk_size, seq_len]
            data_chunk = torch.load(file_path, map_location="cpu", weights_only=True)
        except TypeError:
            data_chunk = torch.load(file_path, map_location="cpu")
        except Exception as e:
            logging.error(f"Failed to load {file_path}: {e}")
            continue

        num_samples = data_chunk.shape[0]

        all_input_tokens = []
        all_target_concepts = []
        all_qwen_tokens = []

        for i in range(0, num_samples, batch_size):
            batch_input_ids = data_chunk[i:i+batch_size].to(device)
            input_len = batch_input_ids.shape[1]

            # A. Generate teacher response text tokens
            with torch.no_grad():
                outputs = teacher_model.generate(
                    input_ids=batch_input_ids,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=teacher_tokenizer.pad_token_id,
                    eos_token_id=teacher_tokenizer.eos_token_id,
                    do_sample=False
                )

            # Extract only the generated tokens (exclude the prompt)
            qwen_generated_tokens = outputs[:, input_len:]

            # Pad generated tokens up to max_new_tokens to maintain rigid matrix shapes
            if qwen_generated_tokens.shape[1] < max_new_tokens:
                pad_len = max_new_tokens - qwen_generated_tokens.shape[1]
                pad_tensor = torch.full(
                    (qwen_generated_tokens.shape[0], pad_len),
                    teacher_tokenizer.pad_token_id,
                    dtype=qwen_generated_tokens.dtype,
                    device=device
                )
                qwen_generated_tokens = torch.cat([qwen_generated_tokens, pad_tensor], dim=1)
            else:
                qwen_generated_tokens = qwen_generated_tokens[:, :max_new_tokens]

            # Decode to string format for the Concept Encoder
            qwen_generated_text = teacher_tokenizer.batch_decode(qwen_generated_tokens, skip_special_tokens=True)

            # B. Encode teacher text into mathematical Concept Vectors
            encoder_inputs = encoder_tokenizer(
                qwen_generated_text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                encoder_outputs = encoder_model(**encoder_inputs)
                # BGE utilizes the [CLS] token at index 0 for sequence representation
                cls_embeddings = encoder_outputs.last_hidden_state[:, 0, :]
                # L2 Normalize the vectors for Cosine Similarity alignment in Phase 3
                concept_vectors = F.normalize(cls_embeddings, p=2, dim=1)  # Target Shape: [Batch, 1024]

            # C. Store the results locally on CPU to free up GPU VRAM
            all_input_tokens.append(batch_input_ids.cpu())
            all_target_concepts.append(concept_vectors.cpu())
            all_qwen_tokens.append(qwen_generated_tokens.cpu())

            # D. Aggressive VRAM cleanup
            del batch_input_ids, outputs, qwen_generated_tokens, encoder_inputs, encoder_outputs, cls_embeddings, concept_vectors
            clear_memory()

        if len(all_input_tokens) > 0:
            # Construct the distilled binary mapping matrix
            distilled_matrix = {
                "input_tokens": torch.cat(all_input_tokens, dim=0),       # Shape: [chunk_size, seq_len]
                "target_concept": torch.cat(all_target_concepts, dim=0),  # Shape: [chunk_size, 1024]
                "qwen_tokens": torch.cat(all_qwen_tokens, dim=0)          # Shape: [chunk_size, 256]
            }

            out_file = output_dir / f"distilled_{Path(file_path).name}"
            torch.save(distilled_matrix, out_file)
            logging.info(f"Saved distilled matrices to {out_file}. Concept Vector Shape: {distilled_matrix['target_concept'].shape}")

            del distilled_matrix, all_input_tokens, all_target_concepts, all_qwen_tokens
            clear_memory()

if __name__ == "__main__":
    main()
