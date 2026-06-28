import os
import sys

# Force Windows to treat all incoming network file streams as UTF-8 globally
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import re
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path

import functools
print = functools.partial(print, flush=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
for handler in logging.root.handlers:
    handler.flush = sys.stdout.flush
    
# SILENCE CODES: Completely suppress network verbosity layers unless a critical failure occurs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# =====================================================================
# 1. Unified Dataset Queue Layout
# =====================================================================
DATASET_QUEUE = {
    # 1. Curated Frontier Model Traces (Reasoning, Alignment & Agentic Flow)
    "frontier_traces": [
        "Crownelius/Complete-FABLE.5-traces-2M",
        "Qwen/AgentWorldBench",
        "nvidia/HelpSteer2",
        "nvidia/HelpSteer2-Pref",
        "nvidia/Nemotron-Math-Proofs-v2",
        "nvidia/Nemotron-RL-InverseIFEval-v1",
        "nvidia/Nemotron-RL-CFBench-v1",
        "nvidia/Nemotron-RL-Multichallenge-v1",
        "nvidia/Nemotron-RL-Math-v2",
        "nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1",
        "nvidia/Nemotron-SFT-Math-v4",
        "nvidia/Nemotron-SFT-Math-v3",
        "nvidia/compute-eval",
        "nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1",
        "nvidia/Nemotron-SFT-Agentic-v2",
        "nvidia/Nemotron-Agentic-v1",
        "nvidia/Nemotron-RL-ReasoningGym-v1",
        "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1",
        "nvidia/Nemotron-SFT-Instruction-Following-Chat-v3",
        "nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1",
        "nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
        "nvidia/Nemotron-SFT-ARC-AGI-v1",
        "nvidia/Nemotron-RL-ARC-AGI-v1",
        "armand0e/qwen3.7-max-pi-traces",
        "mfielding92/gemini-3.1-pro-2048-reasoning-1100x",
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
        "AletheiaResearch/GLM-5.2-Agent",
        "AletheiaResearch/GPT-5.5-Codex",
        "Quaxicron/Fable-5-traces",
        "cfahlgren1/Fable-5-traces",
        "ansulev/claude_mythos_distilled_25k",
        "ox-ox/mythos-character-distillation",
        "11-47/claude_opus_4.8_max_thinking_5k_v2",
        "Quaxicron/claude-opus-4.8-pi-traces",
        "angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k",
        "TeichAI/lordx64-claude-opus-4.7-max-cleaned",
        "Jackrong/Claude-opus-4.7-TraceInversion-5000x",
        "AletheiaResearch/GLM-5.2-Bench",
        "armand0e/gpt-5.5-agent",
        "armand0e/gpt-5.5-chat",
        "hotdogs/uka-glm-5.2",
        "armand0e/minimax-m3-claude-code-traces",
        "Infatoshi/kernelbench-mega-traces",
        "Roman1111111/gemini-3.1-pro-hard-high-reasoning",
        "FradSer/DeepSeek-R1-Distilled-Translate-en-zh_CN-39k",
        "FradSer/DeepSeek-R1-Distilled-Translate-en-zh_CN-39k-Alpaca-GPT4",
        "WebNovelTrans/kunpeng-sentencepairs-5m-instruction"
    ],
    "general_knowledge": [
        "HuggingFaceFW/fineweb-edu",
        "nvidia/Nemotron-Pretraining-Legal-v1",
        "nvidia/Nemotron-Pretraining-Specialized-v1.2",
        "nvidia/Nemotron-SFT-Multilingual-v2",
        "nvidia/Nemotron-SFT-Safety-v2",
        "nvidia/Nemotron-SpecializedDomains-Finance-v1",
        "nvidia/Nemotron-SFT-Science-v2",
        "nvidia/Nemotron-RL-Science-v1",
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
        "rajpurkar/squad",
        "google/boolq",
        "AlgorithmicResearchGroup/arxiv_s2orc_parsed",
        "AlgorithmicResearchGroup/s2orc_full",
        "AlgorithmicResearchGroup/s2orc-cs-enriched"
    ],
    "code_mechanics": [
        "nvidia/Nemotron-Pretraining-Code-v3",
        "nvidia/Open-SWE-Traces",
        "nvidia/Nemotron-SFT-SWE-v3",
        "nvidia/SWE-Zero-openhands-trajectories",
        "nvidia/Nemotron-SWE-v1",
        "nvidia/Nemotron-SFT-SWE-v2",
        "nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1",
        "nvidia/Nemotron-SFT-OpenCode-v1",
        "nvidia/Nemotron-SFT-CUDA-v1",
        "nvidia/Nemotron-Competitive-Programming-v1",
        "nvidia/Nemotron-SFT-Competitive-Programming-v2",
        "nvidia/Nemotron-RL-SysBench-v1",
        "Infatoshi/kernelbench-hard-traces",
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
        "AlgorithmicResearchGroup/arxiv_cplusplus_research_code"
    ]
}

SPECIAL_LOAD_ARGS = {
    "HuggingFaceFW/fineweb-edu": {"name": "sample-100BT", "split": "train"},
    "liwu/MNBVC": {"name": "web_novel", "split": "train"},
    "Qwen/AgentWorldBench": {"path": "text", "data_files": "hf://datasets/Qwen/AgentWorldBench/*_test.jsonl", "split": "train"},
    "jedisct1/security-audits": {"name": "all", "split": "train"},
    "nvidia/compute-eval": {"split": "eval"},
    "nvidia/HelpSteer2-Pref": {"path": "nvidia/HelpSteer2", "data_files": "preference/preference.jsonl.gz", "split": "train"}
}

def setup_device():
    try:
        import intel_extension_for_pytorch as ipex
        logging.info("Intel Extension for PyTorch (IPEX) initialization checked.")
    except ImportError:
        pass
    if torch.xpu.is_available():
        device_name = "xpu"
    elif torch.cuda.is_available():
        device_name = "cuda"
    else:
        device_name = "cpu"
    logging.info(f"Execution engine mapped to backend device target: [{device_name.upper()}]")
    return torch.device(device_name)

def sanitize_dataset_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def stringify_complex(obj):
    if isinstance(obj, str): return obj
    if isinstance(obj, (list, dict)):
        try: return json.dumps(obj, ensure_ascii=False)
        except Exception: return str(obj)
    return str(obj)

def find_key_recursive(data, target_key):
    if isinstance(data, dict):
        if target_key in data: return data[target_key]
        for key, value in data.items():
            result = find_key_recursive(value, target_key)
            if result is not None: return result
    elif isinstance(data, list):
        for item in data:
            result = find_key_recursive(item, target_key)
            if result is not None: return result
    return None

def extract_all_strings(data):
    strings = []
    if isinstance(data, str): strings.append(data)
    elif isinstance(data, dict):
        for val in data.values(): strings.extend(extract_all_strings(val))
    elif isinstance(data, list):
        for item in data: strings.extend(extract_all_strings(item))
    return strings

def extract_qa_pair(row, dataset_name):
    try:
        if isinstance(row, dict) and "text" in row:
            text_val = row["text"]
            if isinstance(text_val, str) and text_val.strip().startswith("{"):
                try:
                    unpacked = json.loads(text_val)
                    if isinstance(unpacked, dict): row = unpacked
                except Exception: pass
        
        # S2ORC Structured Extraction Intercept
        if isinstance(row, dict) and "abstract" in row and ("full_text" in row or "text" in row):
            body_key = "full_text" if "full_text" in row else "text"
            if row["abstract"] and row[body_key]:
                return stringify_complex(row["abstract"]), stringify_complex(row[body_key])
                
        if "zh" in row and "en" in row: return stringify_complex(row["zh"]), stringify_complex(row["en"])
        if "prompt" in row and "completion" in row: return stringify_complex(row["prompt"]), stringify_complex(row["completion"])
        if "current_prompt" in row and "response" in row: return stringify_complex(row["current_prompt"]), stringify_complex(row["response"])

        messages = None
        if isinstance(row, dict) and "responses_create_params" in row:
            params = row["responses_create_params"]
            if isinstance(params, dict) and "input" in params: messages = params["input"]

        if not messages and isinstance(row, dict):
            for k, v in row.items():
                if isinstance(v, list) and len(v) >= 1 and isinstance(v[0], (dict, str)):
                    messages = v
                    break

        if not messages: messages = find_key_recursive(row, "messages")
        if not messages: messages = find_key_recursive(row, "conversations")
        if not messages: messages = find_key_recursive(row, "trajectory")

        if messages and isinstance(messages, list) and len(messages) >= 1:
            prompt_parts, response_parts = [], []
            for idx, msg in enumerate(messages):
                if isinstance(msg, str):
                    role = "system_event" if idx % 2 == 0 else "agent_action"
                    content = msg
                elif isinstance(msg, dict):
                    role = str(msg.get("role", msg.get("from", msg.get("uid", "turn")))).lower()
                    content = stringify_complex(msg.get("content", msg.get("value", msg.get("text", ""))))
                else:
                    continue
                if idx == 0 or (idx == 1 and role in ["user", "human", "prompter"] and len(prompt_parts) <= 1):
                    prompt_parts.append(f"{role}: {content}")
                else:
                    response_parts.append(f"[{role}]: {content}")
            if prompt_parts:
                prompt_str = "\n".join(prompt_parts)
                response_str = "\n".join(response_parts) if response_parts else "completed"
                return prompt_str, response_str

        if "instruction" in row and "output" in row:
            prompt = stringify_complex(row["instruction"])
            if "input" in row and row["input"] and isinstance(row["input"], str): 
                prompt += f"\n\nContext: {stringify_complex(row['input'])}"
            return prompt, stringify_complex(row["output"])

        if "prompt" in row and "response" in row: return stringify_complex(row["prompt"]), stringify_complex(row["response"])
        if "question" in row and "answer" in row: return stringify_complex(row["question"]), stringify_complex(row["answer"])
        if "role" in row and "content" in row: return stringify_complex(row["role"]), stringify_complex(row["content"])

        if "text" in row:
            text = stringify_complex(row["text"])
            if len(text) > 10:
                mid = len(text) // 2
                return text[:mid], text[mid:]

        all_strings = extract_all_strings(row)
        if all_strings:
            longest_string = max(all_strings, key=len)
            if len(longest_string) > 10:
                mid = len(longest_string) // 2
                return longest_string[:mid], longest_string[mid:]
    except Exception: pass
    return None, None

def process_datasets(save_dir=r"F:\JEPA_Model\data\shards", chunk_size=10000):
    scratch_path = Path(save_dir)
    scratch_path.mkdir(parents=True, exist_ok=True)
    device = setup_device()

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    encoder_id = "BAAI/bge-m3"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_id)
    encoder_model = AutoModel.from_pretrained(encoder_id).to(device)
    encoder_model.eval()

    start_time = time.time()
    for domain, datasets in DATASET_QUEUE.items():
        for dataset_name in datasets:
            safe_name = sanitize_dataset_name(dataset_name)
            lock_path = f"gdrive:JEPA_Shards/locks/{domain}_{safe_name}.lock"
            lock_claimed = False
            
            try:
                # 1. GLOBAL DRIVE LOCK CHECK
                try:
                    lock_check = subprocess.run(["rclone", "lsf", lock_path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                    if lock_check.stdout.strip():
                        logging.info(f"[{device.type.upper()}] YIELD: Dataset {dataset_name} is locked by another instance. Skipping...")
                        continue
                except Exception: pass
                    
                # 2. ASSERT CLOUD EXCLUSIVE CLAIM
                logging.info(f"[{device.type.upper()}] CLAIMING exclusive workspace access for: {dataset_name}")
                try:
                    local_lock_file = scratch_path / f"active_local_work.lock"
                    local_lock_file.touch()
                    subprocess.run(["rclone", "copyto", str(local_lock_file), lock_path], stderr=subprocess.DEVNULL)
                    lock_claimed = True
                except Exception as e:
                    logging.error(f"Failed to push cloud lock token: {e}")

                logging.info(f"[{device.type.upper()}] Initiating vector streaming loop for: {dataset_name}")
                buffer = []
                chunk_id = 0

                load_args = SPECIAL_LOAD_ARGS.get(dataset_name, {"path": dataset_name, "split": "train"})
                load_args["streaming"] = True
                
                is_teich_format = False
                try: 
                    ds = load_dataset(**load_args)
                except Exception:
                    try:
                        logging.info(f"Deploying Teich trace fallback parsing sequence for {dataset_name}...")
                        from teich import load_traces
                        ds = load_traces(dataset_name)
                        is_teich_format = True
                    except Exception:
                        logging.error(f"Critical data fetch failure. Skipping {dataset_name}.")
                        continue

                if domain == "general_knowledge": max_rows = 20_000_000
                elif dataset_name == "wdndev/webnovel-chinese": max_rows = 5_000_000   
                elif dataset_name == "nvidia/Nemotron-Pretraining-Code-v3": max_rows = 20_000_000
                elif domain == "frontier_traces": max_rows = 500_000
                else: max_rows = 1_000_000
                    
                rows_processed = 0

                # 3. AUTOMATED DRIVE RESUME CHECK
                try:
                    cloud_files = subprocess.check_output(["rclone", "lsf", "gdrive:JEPA_Shards"], text=True, stderr=subprocess.DEVNULL).splitlines()
                except Exception: cloud_files = []

                if cloud_files:
                    chunk_ids = []
                    for filename in cloud_files:
                        if filename.startswith(f"{domain}_{safe_name}_") and filename.endswith(".pt"):
                            match = re.search(r"_(\d+)\.pt$", filename)
                            if match: chunk_ids.append(int(match.group(1)))
                    if chunk_ids:
                        chunk_id = max(chunk_ids) + 1
                        rows_to_skip = chunk_id * chunk_size
                        if rows_to_skip > 0:
                            logging.info(f"Resuming {dataset_name}: Fast-forwarding core iterator past {rows_to_skip} rows...")
                            if not is_teich_format: ds = ds.skip(rows_to_skip)
                            rows_processed = rows_to_skip

                try:
                    ds_iterator = iter(ds)
                    if is_teich_format and rows_processed > 0:
                        for _ in range(rows_processed):
                            try: next(ds_iterator)
                            except StopIteration: break
                    
                    micro_batch_size = 32
                    staging_pool = []
                    
                    while True:
                        if rows_processed >= max_rows: break
                        try: 
                            row = next(ds_iterator)
                        except StopIteration: 
                            break
                        except Exception as e:
                            logging.warning(f"Transient HTTP packet read drop on {dataset_name}: {e}. Advancing to next record...")
                            continue

                        try:
                            prompt, response = extract_qa_pair(row, dataset_name)
                            if not prompt or not response: continue

                            input_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)["input_ids"].squeeze(0).cpu()
                            qwen_tokens = tokenizer(response, return_tensors="pt", truncation=True, max_length=2048)["input_ids"].squeeze(0).cpu()

                            if input_tokens.numel() == 0 or qwen_tokens.numel() == 0: continue

                            staging_pool.append({
                                "prompt_tokens": input_tokens,
                                "response_tokens": qwen_tokens,
                                "raw_response": response
                            })

                            if len(staging_pool) >= micro_batch_size or rows_processed + len(staging_pool) >= max_rows:
                                texts_to_embed = [item["raw_response"] for item in staging_pool]
                                
                                enc_inputs = encoder_tokenizer(texts_to_embed, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
                                with torch.no_grad():
                                    enc_outputs = encoder_model(**enc_inputs)
                                    cls_embeddings = enc_outputs.last_hidden_state[:, 0, :]
                                    target_concepts = F.normalize(cls_embeddings, p=2, dim=1).cpu()

                                for i, staged_item in enumerate(staging_pool):
                                    buffer.append({
                                        "input_tokens": staged_item["prompt_tokens"],
                                        "qwen_tokens": staged_item["response_tokens"],
                                        "target_concept": target_concepts[i]
                                    })
                                    rows_processed += 1

                                    if len(buffer) >= chunk_size:
                                        sys.stdout.write("\r" + " " * 110 + "\r")
                                        shard_name = f"{domain}_{safe_name}_{chunk_id}.pt"
                                        local_shard_path = scratch_path / shard_name
                                        torch.save(buffer, local_shard_path)
                                        
                                        elapsed = time.time() - start_time
                                        throughput = len(buffer) / elapsed if elapsed > 0 else 0
                                        logging.info(f"Shard Compiled. Shipping {shard_name} via background channel at {throughput:.2f} samples/sec...")
                                        
                                        subprocess.Popen(["rclone", "copyto", str(local_shard_path), f"gdrive:JEPA_Shards/{shard_name}", "--drive-chunk-size", "64M"])
                                        
                                        start_time = time.time()
                                        buffer = []
                                        chunk_id += 1
                                        
                                        if torch.xpu.is_available(): torch.xpu.empty_cache()
                                        elif torch.cuda.is_available(): torch.cuda.empty_cache()

                                current_elapsed = time.time() - start_time
                                current_speed = rows_processed / current_elapsed if current_elapsed > 0 else 0
                                
                                display_name = dataset_name.split('/')[-1] if '/' in dataset_name else dataset_name
                                if len(display_name) > 28:
                                    display_name = display_name[:25] + "..."
                                    
                                ticker_line = f"\r -> [{domain[:8].upper()}] {display_name}: {rows_processed:,} rows... ({current_speed:.0f} rows/sec)"
                                sys.stdout.write(ticker_line.ljust(110)[:110])
                                sys.stdout.flush()

                                staging_pool = []
                        except Exception: continue
                except Exception as e: logging.error(f"Fault inside dataset loop: {e}")

                if buffer:
                    sys.stdout.write("\r" + " " * 110 + "\r")
                    shard_name = f"{domain}_{safe_name}_{chunk_id}.pt"
                    local_shard_path = scratch_path / shard_name
                    torch.save(buffer, local_shard_path)
                    logging.info(f"Flushing tail buffer shard: {shard_name} to cloud folder...")
                    subprocess.run(["rclone", "copyto", str(local_shard_path), f"gdrive:JEPA_Shards/{shard_name}"])
                    if torch.xpu.is_available(): torch.xpu.empty_cache()
                    elif torch.cuda.is_available(): torch.cuda.empty_cache()

                sys.stdout.write("\r" + " " * 110 + "\r")
                logging.info(f"Finished dataset {dataset_name} normally. Total rows processed: {rows_processed:,}")

            finally:
                if lock_claimed:
                    logging.info(f"Releasing atomic cloud lock token for: {dataset_name}")
                    subprocess.run(["rclone", "delete", lock_path], stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and shard frontier datasets.")
    parser.add_argument("--save_dir", type=str, default=r"F:\JEPA_Model\data\shards")
    parser.add_argument("--chunk_size", type=int, default=10000)
    args = parser.parse_args()

    process_datasets(save_dir=args.save_dir, chunk_size=args.chunk_size)