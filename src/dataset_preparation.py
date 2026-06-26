import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path
import logging

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
        logging.warning("intel_extension_for_pytorch not found. Device assignment to 'xpu' will be attempted using native PyTorch.")

    device = torch.device("xpu")
    logging.info(f"Targeting native Intel GPU compute via device: {device}")
    return device

def prepare_text_from_item(item, dataset_name, tokenizer):
    """
    Extracts the prompt and formats it using the model's native chat template.
    We DO NOT include the Assistant's answer because we want the Teacher model 
    to generate its own fresh reasoning trace.
    """
    messages = []
    
    if dataset_name == "AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset":
        messages.append({"role": "system", "content": item.get('system', '')})
        messages.append({"role": "user", "content": item.get('user', '')})
        
    elif dataset_name in ["TheAgenticAI/Agentic-Reasoning", "teknium/OpenHermes-2.5"]:
        system = item.get('system', item.get('system_prompt', ''))
        if system:
            messages.append({"role": "system", "content": system})
            
        conversations = item.get('conversations', [])
        # Only grab the User's prompt, ignore the Assistant's pre-written answer
        for conv in conversations:
            if conv.get('from', '') == 'human':
                messages.append({"role": "user", "content": conv.get('value', '')})
                break # We only need the first prompt for generation
                
    elif dataset_name == "ise-uiuc/Magicoder-OSS-Instruct-75K":
        messages.append({"role": "system", "content": "You are an exceptionally intelligent coding assistant."})
        messages.append({"role": "user", "content": item.get('problem', '')})

    # Let the tokenizer wrap the text in the correct <|im_start|> tags
    # add_generation_prompt=True adds the final <|im_start|>assistant tag to trigger generation
    if messages:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text.strip()
    return ""

def process_and_save_datasets(
    model_name="Jackrong/Qwopus3.6-27B-v2",
    save_dir=r"F:\JEPA_Model\data",
    chunk_size=1,
    max_samples_per_dataset=25000,
    max_seq_len=4096 # Updated to 4K to capture complex instructions
):
    # Path configuration
    save_path = Path(save_dir)

    marker_file = save_path / ".prep_completed"
    if marker_file.exists():
        logging.info("Found .prep_completed marker. Datasets are already cached. Skipping preparation.")
        return

    device = setup_device()

    save_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left" 

    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logging.info("Added [PAD] token as no pad or eos token was found.")

    datasets_to_process = [
        "AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset",
        "TheAgenticAI/Agentic-Reasoning",
        "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "teknium/OpenHermes-2.5"
    ]

    for dataset_name in datasets_to_process:
        logging.info(f"Processing dataset: {dataset_name}")

        try:
            dataset = load_dataset(dataset_name, split="train", streaming=True)
        except Exception as e:
            logging.error(f"Failed to load dataset {dataset_name}: {e}")
            continue

        buffer = []
        file_index = 0
        samples_processed = 0

        for item in dataset:
            if max_samples_per_dataset is not None and samples_processed >= max_samples_per_dataset:
                break

            text = prepare_text_from_item(item, dataset_name, tokenizer)
            if not text:
                continue

            # Tokenize without truncation
            tokens = tokenizer(
                text,
                truncation=False,
                return_tensors="pt"
            )

            input_ids = tokens["input_ids"].squeeze(0) # 1D tensor

            # HARD LIMIT FILTER: Skip prompts that are too long for the Arc B70 VRAM
            if len(input_ids) > max_seq_len:
                logging.warning(f"Skipping sample: Prompt length {len(input_ids)} exceeds safe memory budget ({max_seq_len}).")
                continue

            # Append the raw, unpadded, variable-length tensor directly
            buffer.append(input_ids)
            samples_processed += 1

            # Periodically write to disk (Save as a LIST of tensors)
            if len(buffer) >= chunk_size:
                if dataset_name in ["AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset", "TheAgenticAI/Agentic-Reasoning"]:
                    prefix = "agentic_"
                elif dataset_name == "ise-uiuc/Magicoder-OSS-Instruct-75K":
                    prefix = "logic_"
                elif dataset_name == "teknium/OpenHermes-2.5":
                    prefix = "creative_"
                else:
                    prefix = "unknown_"

                out_file = save_path / f"{prefix}set_{file_index}.pt"
                
                # Save the list of variable-length tensors directly
                torch.save(buffer, out_file)
                logging.info(f"Saved {out_file} with {len(buffer)} items ({samples_processed} source items processed)")

                buffer = []
                file_index += 1

        # Flush remaining buffer
        if buffer:
            if dataset_name in ["AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset", "TheAgenticAI/Agentic-Reasoning"]:
                prefix = "agentic_"
            elif dataset_name == "ise-uiuc/Magicoder-OSS-Instruct-75K":
                prefix = "logic_"
            elif dataset_name == "teknium/OpenHermes-2.5":
                prefix = "creative_"
            else:
                prefix = "unknown_"

            out_file = save_path / f"{prefix}set_{file_index}.pt"
            torch.save(buffer, out_file)
            logging.info(f"Saved final {out_file} with {len(buffer)} items")

    # Mark as completed
    marker_file.touch()

if __name__ == "__main__":
    process_and_save_datasets()