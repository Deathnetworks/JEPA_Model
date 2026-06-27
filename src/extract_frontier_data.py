import os
import time
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
    # 1. Curated Frontier Model Traces (Reasoning, Alignment & Agentic Flow)
    "frontier_traces": [
        "Crownelius/Complete-FABLE.5-traces-2M",
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
        "mfielding92/gemini-3.1-pro-2048-reasoning-1100x",
        "benchflow/skillsbench-leaderboard",
        "evaleval/EEE_datastore",
        # Parallel Translation Alignments belong here!
        "FradSer/DeepSeek-R1-Distilled-Translate-en-zh_CN-39k",
        "FradSer/DeepSeek-R1-Distilled-Translate-en-zh_CN-39k-Alpaca-GPT4",
        "bh2821/LightNovel5000",
        "Skyhigh-2203/MiMo-2.5-Pro-Reasoning-Traces-Hard",
        "Glint-Research/Fable-5-traces",
        "owenqwenllmwine/bigpi-1",
        "sornnakub/Fable-5-traces",
        "shijunhao/Fable-5-traces",
        "TeichAI/DeepSeek-v4-Pro-Agent",
        "kira/Fable-5-traces",
        "ansulev/DeepSeek-v4-Pro-Agent",
        "hardcoremoore/DeepSeek-v4-Pro-Agent",
        "ronaldcmz/DeepSeek-v4-Pro-Agent",
        "ororai/ORORAi",
        "julien-c/synthtraces",
        "armand0e/teich-test-v1",
        "choucsan/mimo-claude-code-traces-1k",
        "AletheiaResearch/GLM-5.2-Agent",
        "AletheiaResearch/GPT-5.5-Codex",
        "Infatoshi/kernelbench-hard-traces",
        "Quaxicron/Fable-5-traces",
        "cfahlgren1/Fable-5-traces"
    ],

    # 2. Massive General Knowledge, Instruction Following, & Creative Core
    "general_knowledge": [
        "HuggingFaceFW/fineweb-edu",
        "HuggingFaceH4/ultrafeedback_clean",
        "teknium/OpenHermes-2.5",
        "teknium/openhermes",
        "KingNish/reasoning-base-20k",
        "Salesforce/wikitext",
        "banned-historical-archives/banned-historical-archives",
        "allenai/c4",
        "stanfordnlp/imdb",
        "legacy-datasets/wikipedia",
        "bookcorpus/bookcorpus",
        "fse/paranmt-300",
        "Skylion007/openwebtext",
        "evaluate-metric/xnli",
        "liwu/MNBVC",
        "wdndev/webnovel-chinese",
        # Moved basic comprehension datasets here
        "rajpurkar/squad",
        "google/boolq"
    ],

    # 3. Code Syntax & Language Grammar Rules (For the JEPA World Model)
    "code_mechanics": [
        "m-a-p/CodeFeedback-Filtered-Instruction",
        "deepmind/code_contests",
        "code-search-net/code_search_net",
        "bigcode/starcoder2-instruct",
        "iamtarun/python-execution-traces",
        "bigcode/the-stack",
        "Salesforce/wikisql",
        "gaianet/learn-rust",
        "semeru/code-code-translation-java-csharp",
        "MehdiFe/csharp-instruction-Dataset",
        "microsoft/LCC_csharp",
        "AlgorithmicResearchGroup/arxiv_cplusplus_research_code",
        "Infatoshi/kernelbench-mega-traces",
        "jedisct1/security-audits",
        "randomanon000/coding-sessions"
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

def find_key_recursive(data, target_key):
    if isinstance(data, dict):
        if target_key in data:
            return data[target_key]
        for key, value in data.items():
            result = find_key_recursive(value, target_key)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_key_recursive(item, target_key)
            if result is not None:
                return result
    return None

def extract_all_strings(data):
    strings = []
    if isinstance(data, str):
        strings.append(data)
    elif isinstance(data, dict):
        for val in data.values():
            strings.extend(extract_all_strings(val))
    elif isinstance(data, list):
        for item in data:
            strings.extend(extract_all_strings(item))
    return strings

def extract_qa_pair(row, dataset_name):
    """
    Polymorphic parser to extract prompt/response pairs from diverse dataset schemas.
    Returns (prompt: str, response: str) or (None, None) if unavailable.
    """
    prompt, response = None, None

    try:
        # Aggressive recursive search for 'messages' or 'conversations'
        messages = find_key_recursive(row, "messages")
        if not messages:
            messages = find_key_recursive(row, "conversations")

        if messages and isinstance(messages, list) and len(messages) >= 2:
            prompt_parts = []
            response_content = None

            # Find the first user message
            for msg in messages:
                if not isinstance(msg, dict): continue
                role = str(msg.get("role", msg.get("from", ""))).lower()
                content = stringify_complex(msg.get("content", msg.get("value", "")))

                if role in ["system", "user", "human", "prompter"] and not prompt_parts:
                    prompt_parts.append(f"{role}: {content}")
                    break # just get the first one for prompt

            # Find the last assistant message
            for msg in reversed(messages):
                if not isinstance(msg, dict): continue
                role = str(msg.get("role", msg.get("from", ""))).lower()
                content = stringify_complex(msg.get("content", msg.get("value", "")))

                if role in ["assistant", "bot", "model", "gpt"]:
                    response_content = content
                    break

            if prompt_parts and response_content:
                prompt = "\n".join(prompt_parts)
                return prompt, response_content

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
            
        # Add this rule inside your extract_qa_pair function in extract_frontier_data.py
        if "zh" in row and "en" in row:
            return stringify_complex(row["zh"]), stringify_complex(row["en"])

        if "source" in row and "target" in row:
            # Verify if it's the Chinese translation partition
            return stringify_complex(row["source"]), stringify_complex(row["target"])

        # Text only (fineweb-edu, etc.)
        if "text" in row:
            text = stringify_complex(row["text"])
            # Artificial split: take first half as prompt, second as response for auto-encoding tasks
            mid = len(text) // 2
            if mid > 0:
                prompt = text[:mid]
                response = text[mid:]
                return prompt, response

        # Ultimate fallback: Find the longest string in the entire row
        all_strings = extract_all_strings(row)
        if all_strings:
            longest_string = max(all_strings, key=len)
            if len(longest_string) > 10:
                mid = len(longest_string) // 2
                prompt = longest_string[:mid]
                response = longest_string[mid:]
                return prompt, response

    except Exception as e:
        logging.debug(f"Failed to parse row: {e}")

    return None, None


def process_datasets(save_dir=r"F:\JEPA_Model\data\shards", chunk_size=1000):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    device = setup_device()

    tokenizer_id = "Qwen/Qwen2.5-7B-Instruct"
    encoder_id = "BAAI/bge-m3"

    logging.info(f"Loading tokenizer: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    logging.info(f"Loading encoder: {encoder_id}")
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_id)
    encoder_model = AutoModel.from_pretrained(encoder_id).to(device)
    encoder_model.eval()

    start_time = time.time()
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
            # Add this specific check for MNBVC
            elif dataset_name == "liwu/MNBVC":
                load_args["name"] = "web_novel" # or "novel" based on your preference

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
            # Set limits based on domain and dataset size
            if domain == "general_knowledge":
                max_rows = 20_000_000
            elif dataset_name == "wdndev/webnovel-chinese":
                max_rows = 5_000_000   # Solid base for Chinese prose structures
            elif domain == "frontier_traces":
                max_rows = 500_000
            else:
                max_rows = 1_000_000

            rows_processed = 0

            # Check for existing shards to resume
            existing_shards = list(save_path.glob(f"{domain}_{safe_name}_*.pt"))
            if existing_shards:
                # Extract chunk IDs
                chunk_ids = []
                for shard in existing_shards:
                    match = re.search(r"_(\d+)\.pt$", shard.name)
                    if match:
                        chunk_ids.append(int(match.group(1)))

                if chunk_ids:
                    highest_chunk_id = max(chunk_ids)
                    chunk_id = highest_chunk_id + 1
                    rows_to_skip = chunk_id * chunk_size

                    if rows_to_skip > 0:
                        logging.info(f"Resuming {dataset_name}: Fast-forwarding {rows_to_skip} rows...")
                        ds = ds.skip(rows_to_skip)
                        rows_processed = rows_to_skip

            try:
                # Wrap the iterator to catch ArrowInvalid and other dataset-level errors
                ds_iterator = iter(ds)
                while True:
                    if rows_processed >= max_rows:
                        break

                    try:
                        row = next(ds_iterator)
                    except StopIteration:
                        break
                    except Exception as e:
                        logging.error(f"Dataset {dataset_name} iterator crashed: {e}. Moving to next dataset.")
                        break

                    try:
                        prompt, response = extract_qa_pair(row, dataset_name)
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
                            elapsed = time.time() - start_time
                            throughput = len(buffer) / elapsed if elapsed > 0 else 0
                            logging.info(f"Saved {shard_path} with {len(buffer)} items. Speed: {throughput:.2f} samples/sec")
                            start_time = time.time()
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
            except Exception as e:
                logging.error(f"Critical error during {dataset_name} processing: {e}")

            # Save any remaining in buffer
            if buffer:
                shard_path = save_path / f"{domain}_{safe_name}_{chunk_id}.pt"
                torch.save(buffer, shard_path)
                elapsed = time.time() - start_time
                throughput = len(buffer) / elapsed if elapsed > 0 else 0
                logging.info(f"Saved final {shard_path} with {len(buffer)} items. Speed: {throughput:.2f} samples/sec")
                start_time = time.time()

            logging.info(f"Finished {dataset_name}. Total processed pairs: {rows_processed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and shard frontier datasets.")
    parser.add_argument("--save_dir", type=str, default=r"F:\JEPA_Model\data\shards")
    parser.add_argument("--chunk_size", type=int, default=1000)
    args = parser.parse_args()

    process_datasets(save_dir=args.save_dir, chunk_size=args.chunk_size)
