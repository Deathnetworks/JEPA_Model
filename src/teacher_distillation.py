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
    teacher_tokenizer.padding_side = 'left'
    
    # --- ADD THIS PRINT CHECK ---
    print(f"DEBUG: Tokenizer padding side is: {teacher_tokenizer.padding_side}")    
    # ----------------------------
    
    if teacher_tokenizer.pad_token_id is None:
        if teacher_tokenizer.eos_token_id is not None:
            teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
        else:
            teacher_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
    # 1. Define memory constraints (Set to ~20GB to stay safe)
    max_memory_mapping = {0: "30GiB", "cpu": "15GiB"}
    offload_dir = r"F:\JEPA_Model\offload_cache"
    os.makedirs(offload_dir, exist_ok=True)
    
    print(f"DEBUG: Max memory settings: {max_memory_mapping}")

    # 2. Updated Teacher Model Load
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        quantization_config=quantization_config,
        device_map="auto",                     # Changed from {"": "xpu"} to allow spilling
        max_memory=max_memory_mapping,         # Enforce the RAM cap
        offload_folder=offload_dir,            # Explicitly route overflow to the SSD
        low_cpu_mem_usage=True,                # Reduce CPU memory spike
        trust_remote_code=True        
    )
    teacher_model.eval()
    
    logging.info("Compiling models for XPU (this may take a few minutes)...")
    try:
        teacher_model = torch.compile(teacher_model, backend="inductor")
        encoder_model = torch.compile(encoder_model, backend="inductor")
        logging.info("Compilation successful.")
    except Exception as e:
        logging.error(f"Torch compilation failed (this is optional, proceeding anyway): {e}")

    # 3. Process Chunked .pt Files
    pt_files = glob.glob(str(input_dir / "*.pt"))
    if not pt_files:
        logging.warning(f"No .pt files found in {input_dir}. Ensure dataset_preparation.py has been run.")
        return

    batch_size = 1
    max_new_tokens = 65536  # Expanded ceiling to support ultra-long agent reasoning traces

    for file_path in pt_files:
        # Define where the output should be
        out_file = output_dir / f"distilled_{Path(file_path).name}"
        
        # --- NEW RESUME LOGIC ---
        if out_file.exists():
            logging.info(f"Skipping {Path(file_path).name} - already processed.")
            continue
        # ------------------------
        
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
            
            # --- ADD THIS: Create an Attention Mask ---
            pad_token_id = teacher_tokenizer.pad_token_id
            attention_mask = (batch_input_ids != pad_token_id).long().to(device)
            # ------------------------------------------
            
            input_len = batch_input_ids.shape[1]

            # A. Generate teacher response text tokens
            with torch.no_grad():
                outputs = teacher_model.generate(
                    input_ids=batch_input_ids,
                    max_new_tokens=max_new_tokens,
                    attention_mask=attention_mask, 
                    pad_token_id=teacher_tokenizer.pad_token_id,
                    eos_token_id=teacher_tokenizer.eos_token_id,
                    do_sample=False,
                    repetition_penalty=1.15,  # Discourages infinite loop repetitions (Scenario 0)
                    cache_implementation="quantized",  # Quantize KV Cache to protect VRAM
                    cache_config={"backend": "quanto", "nbits": 8}
                )

            # Extract only the generated tokens (exclude the prompt)
            qwen_generated_tokens = outputs[:, input_len:]

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

            # C. Store the results locally on CPU as unbatched 1D tensors to preserve unique lengths
            all_input_tokens.append(batch_input_ids.squeeze(0).cpu())
            all_target_concepts.append(concept_vectors.squeeze(0).cpu())
            all_qwen_tokens.append(qwen_generated_tokens.squeeze(0).cpu())

            # D. Aggressive VRAM cleanup
            del batch_input_ids, attention_mask, outputs, qwen_generated_tokens, encoder_inputs, encoder_outputs, cls_embeddings, concept_vectors
            clear_memory()
            
            # --- ADD THIS: Progress Tracker ---
            processed_in_file = i + batch_size
            percent = (processed_in_file / num_samples) * 100
            logging.info(f"Progress: {percent:.2f}% | Current Item: {processed_in_file}/{num_samples}")
            # ----------------------------------

        if len(all_input_tokens) > 0:
            # Construct the distilled structure as a dictionary of lists to avoid tensor shape mismatch crashes
            distilled_matrix = {
                "input_tokens": all_input_tokens,      # List of 1D tensors (varying length)
                "target_concept": all_target_concepts,  # List of 1D tensors (shape: [1024])
                "qwen_tokens": all_qwen_tokens          # List of 1D tensors (varying length)
            }

            out_file = output_dir / f"distilled_{Path(file_path).name}"
            torch.save(distilled_matrix, out_file)
            logging.info(f"Saved distilled lists to {out_file}.")

            del distilled_matrix, all_input_tokens, all_target_concepts, all_qwen_tokens
            clear_memory()

if __name__ == "__main__":
    main()