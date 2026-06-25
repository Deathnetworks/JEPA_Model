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

def prepare_text_from_item(item, dataset_name):
    """
    Extracts and formats text appropriately from different dataset schemas.
    """
    text = ""
    if dataset_name == "AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset":
        system = item.get('system', '')
        user = item.get('user', '')
        assistant = item.get('assistant', '')
        text = f"System: {system}\n\nUser: {user}\n\nAssistant: {assistant}"
    elif dataset_name == "TheAgenticAI/Agentic-Reasoning":
        system = item.get('system', '')
        text = f"System: {system}\n\n"
        conversations = item.get('conversations', [])
        for conv in conversations:
            role = conv.get('from', '')
            value = conv.get('value', '')
            text += f"{role.capitalize()}: {value}\n\n"
    elif dataset_name == "ise-uiuc/Magicoder-OSS-Instruct-75K":
        problem = item.get('problem', '')
        solution = item.get('solution', '')
        text = f"System: You are an exceptionally intelligent coding assistant.\n\nUser: {problem}\n\nAssistant: {solution}"
    elif dataset_name == "teknium/OpenHermes-2.5":
        system = item.get('system_prompt', '')
        text = f"System: {system}\n\n"
        conversations = item.get('conversations', [])
        for conv in conversations:
            role = conv.get('from', '')
            value = conv.get('value', '')
            text += f"{role.capitalize()}: {value}\n\n"
    return text.strip()

def process_and_save_datasets(
    model_name="Qwen/Qwen3.6-27B",
    save_dir=r"F:\JEPA_Model\data",
    seq_len=2048,
    chunk_size=5000,
    max_samples_per_dataset=20000
):
    device = setup_device()

    # Path configuration
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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

            text = prepare_text_from_item(item, dataset_name)
            if not text:
                continue

            # Tokenize without truncation to preserve the entire reasoning chain
            tokens = tokenizer(
                text,
                truncation=False,
                return_tensors="pt"
            )

            input_ids = tokens["input_ids"].squeeze(0) # 1D tensor of variable length

            # Chunking into multiple seq_len segments
            for i in range(0, len(input_ids), seq_len):
                chunk = input_ids[i:i+seq_len]
                if len(chunk) < seq_len:
                    # Pad to match standard seq_len for continuous tensor architecture
                    pad_len = seq_len - len(chunk)
                    pad_tensor = torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)
                    chunk = torch.cat([chunk, pad_tensor])

                # Keep CPU tensors for saving to disk, move to XPU during training DataLoader
                buffer.append(chunk)

            samples_processed += 1

            # Periodically write to disk
            if len(buffer) >= chunk_size:
                tensor_chunk = torch.stack(buffer) # [chunk_size, seq_len]

                if dataset_name in ["AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset", "TheAgenticAI/Agentic-Reasoning"]:
                    prefix = "agentic_"
                elif dataset_name == "ise-uiuc/Magicoder-OSS-Instruct-75K":
                    prefix = "logic_"
                elif dataset_name == "teknium/OpenHermes-2.5":
                    prefix = "creative_"
                else:
                    prefix = "unknown_"

                out_file = save_path / f"{prefix}set_{file_index}.pt"

                # We save locally as .pt which acts as a cache for standard DataLoader
                torch.save(tensor_chunk, out_file)
                logging.info(f"Saved {out_file} with shape {tensor_chunk.shape} ({samples_processed} source items processed)")

                buffer = []
                file_index += 1

        # Flush remaining buffer
        if buffer:
            tensor_chunk = torch.stack(buffer)

            if dataset_name in ["AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset", "TheAgenticAI/Agentic-Reasoning"]:
                prefix = "agentic_"
            elif dataset_name == "ise-uiuc/Magicoder-OSS-Instruct-75K":
                prefix = "logic_"
            elif dataset_name == "teknium/OpenHermes-2.5":
                prefix = "creative_"
            else:
                prefix = "unknown_"

            out_file = save_path / f"{prefix}set_{file_index}.pt"
            torch.save(tensor_chunk, out_file)
            logging.info(f"Saved final {out_file} with shape {tensor_chunk.shape}")

if __name__ == "__main__":
    process_and_save_datasets()
