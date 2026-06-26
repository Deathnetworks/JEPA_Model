import os
import re
import json
import logging
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATASET_QUEUE = {
    # 1. Curated Frontier Model Traces (Reasoning & Agentic Flow)
    "frontier_traces": [
        "Crownelius/Complete-FABLE.5-traces-2M",
        "armand0e/claude-fable-5-claude-code",
        "ansulev/claude_mythos_distilled_25k",
        "ox-ox/mythos-character-distillation",
        "11-47/claude_opus_4.8_max_thinking_5k_v2",
        "Quaxicron/claude-opus-4.8-pi-traces",
        "angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k",
        "TeichAI/lordx64-claude-opus-4.7-max-cleaned",
        "Jackrong/Claude-opus-4.7-TraceInversion-5000x",
        "AletheiaResearch/GPT-5.5-Codex",
        "armand0e/gpt-5.5-agent",
        "armand0e/gpt-5.5-chat",
        "hotdogs/uka-glm-5.2",
        "AletheiaResearch/GLM-5.2-Bench",
        "armand0e/qwen3.7-max-pi-traces",
        "tomaarsen/zelo-scores-10kx100-qwen3.6-27b",
        "zake7749/Qwen3.6-35B-A3B-Tool-Calling",
        "khazarai/qwen3.6-plus-high-reasoning-500x",
        "caiovicentino1/Qwen3.6-35B-A3B-mcr-stage-b",
        "armand0e/minimax-m3-claude-code-traces",
        "Infatoshi/kernelbench-mega-traces",
        "Roman1111111/gemini-3.1-pro-hard-high-reasoning",
        "PhysEdit/pawbench-gemini-expansion-20260619",
        "TTS-AGI/dramabox-gemini-finetune",
        "mfielding92/gemini-3.1-pro-2048-reasoning-1100x",
        "benchflow/skillsbench-leaderboard",
        "evaleval/EEE_datastore"
    ],

    # 2. Massive General Knowledge, Instruction Following, & Creative Core
    "general_knowledge": [
        "HuggingFaceFW/fineweb-edu",
        "HuggingFaceH4/ultrafeedback_clean",
        "technium/OpenHermes-2.5",
        "KingNish/reasoning-base-20k"
    ],

    # 3. Code Syntax & Language Grammar Rules (For the JEPA World Model)
    "code_mechanics": [
        "bigcode/starcoder2-instruct",
        "iamtarun/python-execution-traces",
        "m-a-p/CodeFeedback-Filtered-Instruction"
    ]
}

def setup_device():
    try:
        import intel_extension_for_pytorch as ipex
        logging.info("intel_extension_for_pytorch imported.")
    except ImportError:
        pass

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device

def sanitize_dataset_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def stringify_complex(obj):
    if isinstance(obj, (dict, list)):
        try:
            return json.dumps(obj)
        except:
            return str(obj)
    return str(obj)

def extract_text_pair(row, dataset_name):
    """
    Polymorphic parser to extract prompt/response pairs from diverse dataset schemas.
    Returns (prompt: str, response: str) or (None, None) if unavailable.
    """
    prompt, response = None, None

    try:
        # Common schemas
        if "messages" in row:
            messages = row["messages"]
            if isinstance(messages, list) and len(messages) >= 2:
                # E.g. [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
                prompt_parts = []
                for msg in messages:
                    if not isinstance(msg, dict): continue
                    role = str(msg.get("role", "")).lower()
                    content = stringify_complex(msg.get("content", ""))
                    if role in ["system", "user", "human", "prompter"]:
                        prompt_parts.append(f"{role}: {content}")
                    elif role in ["assistant", "bot", "model"]:
                        response = content
                        break
                if prompt_parts and response:
                    prompt = "\n".join(prompt_parts)
                    return prompt, response

        if "conversations" in row:
            conversations = row["conversations"]
            if isinstance(conversations, list) and len(conversations) >= 2:
                prompt_parts = []
                for conv in conversations:
                    if not isinstance(conv, dict): continue
                    from_role = str(conv.get("from", "")).lower()
                    val = stringify_complex(conv.get("value", ""))
                    if from_role in ["human", "user", "system"]:
                        prompt_parts.append(f"{from_role}: {val}")
                    elif from_role in ["gpt", "assistant", "bot"]:
                        response = val
                        break
                if prompt_parts and response:
                    prompt = "\n".join(prompt_parts)
                    return prompt, response

        # Instruction / Output
        if "instruction" in row and "output" in row:
            instruction = stringify_complex(row["instruction"])
            if "input" in row and row["input"]:
                instruction += "\n" + stringify_complex(row["input"])
            return instruction, stringify_complex(row["output"])

        # Prompt / Response
        if "prompt" in row and "response" in row:
            return stringify_complex(row["prompt"]), stringify_complex(row["response"])

        if "prompt" in row and "completion" in row:
            return stringify_complex(row["prompt"]), stringify_complex(row["completion"])

        # Text only (fineweb-edu, etc.)
        if "text" in row:
            text = stringify_complex(row["text"])
            # Artificial split: take first half as prompt, second as response for auto-encoding tasks
            mid = len(text) // 2
            if mid > 0:
                prompt = text[:mid]
                response = text[mid:]
                return prompt, response

    except Exception as e:
        logging.debug(f"Failed to parse row: {e}")

    return None, None

def process_datasets(save_dir=r"F:\JEPA_Model\data\shards", chunk_size=1000):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    device = setup_device()

    tokenizer_id = "Qwen/Qwen2.5-7B-Instruct"
    encoder_id = "BAAI/bge-large-en-v1.5"

    logging.info(f"Loading tokenizer: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    logging.info(f"Loading encoder: {encoder_id}")
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_id)
    encoder_model = AutoModel.from_pretrained(encoder_id).to(device)
    encoder_model.eval()

    for domain, datasets in DATASET_QUEUE.items():
        for dataset_name in datasets:
            logging.info(f"Starting ingestion for: {dataset_name} in domain: {domain}")

            safe_name = sanitize_dataset_name(dataset_name)
            chunk_id = 0
            buffer = []

            # Dataset specific args
            load_args = {"path": dataset_name, "split": "train", "streaming": True}
            if dataset_name == "HuggingFaceFW/fineweb-edu":
                load_args["name"] = "sample-10BT"

            try:
                ds = load_dataset(**load_args)
            except Exception as e:
                logging.warning(f"Failed to load dataset {dataset_name}: {e}. Trying without 'train' split...")
                try:
                    ds = load_dataset(dataset_name, streaming=True)
                    if "train" in ds:
                        ds = ds["train"]
                    else:
                        ds = next(iter(ds.values()))
                except Exception as e2:
                    logging.error(f"Failed again to load {dataset_name}: {e2}. Skipping.")
                    continue

            # Limit total rows to avoid infinite loops on huge streaming datasets if we don't want to process everything
            # Let's say process up to 100k pairs for massive sets like fineweb, others bounded by natural length or 1M.
            max_rows = 50000 if dataset_name == "HuggingFaceFW/fineweb-edu" else 250000
            rows_processed = 0

            for row in ds:
                if rows_processed >= max_rows:
                    break

                try:
                    prompt, response = extract_text_pair(row, dataset_name)
                    if not prompt or not response:
                        continue

                    # 1 & 2: Tokenize input and response
                    input_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)["input_ids"].squeeze(0)
                    qwen_tokens = tokenizer(response, return_tensors="pt", truncation=True, max_length=2048)["input_ids"].squeeze(0)

                    # Skip empty sequences
                    if input_tokens.numel() == 0 or qwen_tokens.numel() == 0:
                        continue

                    # 3: Compute target concept on XPU
                    enc_inputs = encoder_tokenizer(response, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)

                    with torch.no_grad():
                        enc_outputs = encoder_model(**enc_inputs)
                        cls_embedding = enc_outputs.last_hidden_state[:, 0, :]
                        target_concept = F.normalize(cls_embedding, p=2, dim=1).squeeze(0).cpu()

                    buffer.append({
                        "input_tokens": input_tokens.cpu(),
                        "qwen_tokens": qwen_tokens.cpu(),
                        "target_concept": target_concept
                    })
                    rows_processed += 1

                    if len(buffer) >= chunk_size:
                        shard_path = save_path / f"{domain}_{safe_name}_{chunk_id}.pt"
                        torch.save(buffer, shard_path)
                        logging.info(f"Saved {shard_path} with {len(buffer)} items.")
                        buffer = []
                        chunk_id += 1

                        # Clear VRAM cache hook
                        if hasattr(torch.xpu, 'empty_cache'):
                            torch.xpu.empty_cache()
                        elif torch.cuda.is_available():
                            torch.cuda.empty_cache()

                except Exception as e:
                    logging.warning(f"Error processing row in {dataset_name}: {e}")
                    continue

            # Save any remaining in buffer
            if buffer:
                shard_path = save_path / f"{domain}_{safe_name}_{chunk_id}.pt"
                torch.save(buffer, shard_path)
                logging.info(f"Saved final {shard_path} with {len(buffer)} items.")

            logging.info(f"Finished {dataset_name}. Total processed pairs: {rows_processed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and shard frontier datasets.")
    parser.add_argument("--save_dir", type=str, default=r"F:\JEPA_Model\data\shards")
    parser.add_argument("--chunk_size", type=int, default=1000)
    args = parser.parse_args()

    process_datasets(save_dir=args.save_dir, chunk_size=args.chunk_size)
