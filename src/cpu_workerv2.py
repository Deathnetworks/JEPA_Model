import os
import re
import time
import logging
import sys
import subprocess
import threading
import uuid
import functools
import shutil
import queue
from pathlib import Path

# Force early low-level runtime configurations
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['HF_TOKEN'] = 'YOUR_HF_TOKEN'
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"

# =====================================================================
# 1. Environment & Secure Rclone Mount
# =====================================================================
print = functools.partial(print, flush=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

logging.info("Installing system dependencies...")
if os.path.exists("/usr/bin/apt-get"):
    subprocess.run(["apt-get", "update", "-y"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    install_res = subprocess.run(["apt-get", "install", "-y", "rclone"], capture_output=True, text=True)
    if install_res.returncode == 0:
        logging.info("System rclone distribution successfully mapped into runtime environment.")
    else:
        logging.error(f"System packaging failure: {install_res.stderr}")

subprocess.run(["pip", "install", "-q", "transformers", "orjson", "datasets", "teich"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

rclone_paths = [
    "/kaggle/input/datasets/deathnetworks/jepa-rclone-config/rclone.conf",
    "/kaggle/input/jepa-rclone-config/rclone.conf"
]
RCLONE_SOURCE = next((p for p in rclone_paths if os.path.exists(p)), None)

if RCLONE_SOURCE:
    os.makedirs("/root/.config/rclone", exist_ok=True)
    subprocess.run(["cp", RCLONE_SOURCE, "/root/.config/rclone/rclone.conf"])
    logging.info(f"Secure Rclone configuration loaded successfully from {RCLONE_SOURCE}")
else:
    if os.path.exists(os.path.expanduser("~/.config/rclone/rclone.conf")):
        logging.info("Using pre-existing local machine rclone workspace credentials mapping.")
    else:
        logging.warning("Gating warning: rclone.conf missing from immediate workspace directories.")

def get_rclone_base_command():
    """Dynamically locates the rclone binary and its config file across context paths."""
    current_dir = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent

    rclone_bin = "rclone"
    if not shutil.which("rclone"):
        for search_path in [script_dir, parent_dir, current_dir]:
            for binary_name in ["rclone.exe", "rclone"]:
                potential_bin = search_path / binary_name
                if potential_bin.exists():
                    rclone_bin = str(potential_bin)
                    break
            if rclone_bin != "rclone":
                break

    base_cmd = [rclone_bin]

    config_found = False
    for search_path in [script_dir, parent_dir, current_dir]:
        potential_config = search_path / "rclone.conf"
        if potential_config.exists():
            base_cmd.extend(["--config", str(potential_config)])
            config_found = True
            break
    return base_cmd

RCLONE_BASE = get_rclone_base_command()
UPLOAD_QUEUE = queue.Queue()

for noisy in [
    "urllib3", "requests", "huggingface_hub", "datasets",
    "transformers", "filelock", "fsspec", "aiohttp", "httpx",
    "httpcore", "hf_xet"
]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

import orjson as json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# 2. Dataset Queue & Data Parsing Utilities
# =====================================================================
DATASET_QUEUE = {
    "frontier_traces": [
        "agentlans/mlabonne-open-perfectblend",
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
        "WebNovelTrans/kunpeng-sentencepairs-5m-instruction",
        "openbmb/UltraData-SFT-2605",
        "mlabonne/open-perfectblend",
        "Crownelius/Complete-FABLE.5-traces-2M",
        "Qwen/AgentWorldBench",
        "nvidia/HelpSteer2",
        "nvidia/HelpSteer2-Pref"
    ],
    "general_knowledge": [
        "idealand/IdeaSeed",
        "HuggingFaceFW/fineweb-edu",
        "nvidia/Nemotron-Pretraining-Legal-v1",
        "nvidia/Nemotron-Pretraining-Specialized-v1.2",
        "nvidia/Nemotron-Pretraining-SFT-v1",
        "nvidia/Nemotron-CC-Math-v1",
        "nvidia/Nemotron-SFT-Multilingual-v2",
        "nvidia/Nemotron-SFT-Safety-v2",
        "nvidia/Nemotron-SpecializedDomains-Finance-v1",
        "nvidia/Nemotron-SFT-Science-v2",
        "nvidia/Nemotron-RL-Science-v1",
        "teknium/OpenHermes-2.5",
        "teknium/openhermes",
        "KingNish/reasoning-base-20k",
        "Salesforce/wikitext",
        "banned-historical-archives/banned-historical-archives",
        "allenai/c4",
        "stanfordnlp/imdb",
        "legacy-datasets/wikipedia",
        "Skylion007/openwebtext",
        "liwu/MNBVC",
        "wdndev/webnovel-chinese",
        "rajpurkar/squad",
        "google/boolq",
        "AlgorithmicResearchGroup/arxiv_s2orc_parsed",
        "AlgorithmicResearchGroup/s2orc-cs-enriched",
        "openbmb/Ultra-FineWeb",
        "openbmb/Ultra-FineWeb-L3",
        "openbmb/UltraData-Math"
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

def stringify_complex(obj):
    if isinstance(obj, str): return obj
    if isinstance(obj, (list, dict)):
        import json as std_json
        try: return std_json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
        except Exception: return str(obj)
    return str(obj)

def sanitize_dataset_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

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

        if "问" in row and "答" in row: return stringify_complex(row["问"]), stringify_complex(row["答"])
        if "Q" in row and "A" in row: return stringify_complex(row["Q"]), stringify_complex(row["A"])

        if isinstance(row, dict) and "abstract" in row and ("full_text" in row or "text" in row):
            body_key = "full_text" if "full_text" in row else "text"
            if row["abstract"] and row[body_key]:
                return stringify_complex(row["abstract"]), stringify_complex(row[body_key])

        if "zh" in row and "en" in row: return stringify_complex(row["zh"]), stringify_complex(row["en"])
        if "prompt" in row and "completion" in row: return stringify_complex(row["prompt"]), stringify_complex(row["completion"])
        if "current_prompt" in row and "response" in row: return stringify_complex(row["current_prompt"]), stringify_complex(row["response"])
        if "instruction" in row and "output" in row:
            prompt = stringify_complex(row["instruction"])
            if "input" in row and row["input"] and isinstance(row["input"], str):
                prompt += f"\n\nContext: {stringify_complex(row['input'])}"
            return prompt, stringify_complex(row["output"])
        if "question" in row and "answer" in row: return stringify_complex(row["question"]), stringify_complex(row["answer"])
        if "prompt" in row and "response" in row: return stringify_complex(row["prompt"]), stringify_complex(row["response"])
        if "role" in row and "content" in row: return stringify_complex(row["role"]), stringify_complex(row["content"])
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
        if messages and isinstance(messages, list) and len(messages) >= 2:
            system_prompt = ""
            if isinstance(messages[0], dict) and str(messages[0].get("role", "")).lower() == "system":
                system_prompt = stringify_complex(messages[0].get("content", "")) + "\n\n"

            last_user = next((stringify_complex(m.get("content")) for m in reversed(messages) if isinstance(m, dict) and str(m.get("role", m.get("from", m.get("uid", "")))).lower() in ["user", "human", "prompter"]), "")
            last_assistant = next((stringify_complex(m.get("content")) for m in reversed(messages) if isinstance(m, dict) and str(m.get("role", m.get("from", m.get("uid", "")))).lower() in ["assistant", "model", "bot"]), "")

            if last_user and last_assistant:
                return f"{system_prompt}User: {last_user}", f"Assistant: {last_assistant}"
        if "text" in row:
            text = stringify_complex(row["text"])
            if len(text) > 100:
                mid = len(text) // 2
                search_window = text[max(0, mid-500):min(len(text), mid+500)]
                split_idx = mid

                if "\n\n" in search_window:
                    split_idx = max(0, mid-500) + search_window.find("\n\n") + 2
                elif "\n" in search_window:
                    split_idx = max(0, mid-500) + search_window.find("\n") + 1

                return text[:split_idx].strip(), text[split_idx:].strip()
            elif len(text) > 10:
                mid = len(text) // 2
                return text[:mid].strip(), text[mid:].strip()
        all_strings = extract_all_strings(row)
        if healthiest_string := [s for s in all_strings if len(s) > 10]:
            longest_string = max(healthiest_string, key=len)
            mid = len(longest_string) // 2
            return longest_string[:mid], longest_string[mid:]

    except Exception: pass
    return None, None

def is_verifiable_logic(text):
    if not isinstance(text, str):
        return False
    # Check for code blocks or standard unit test assertion markers
    has_code_block = bool(re.search(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", text, re.DOTALL))
    has_assertions = bool(re.search(r"(assert_eq!|#\[test\]|def test_|assert |EXPECT_EQ)", text))
    return has_code_block or has_assertions

# =====================================================================
# 3. Fault-Tolerant Exponential Backoff Cloud Bridge
# =====================================================================
def sync_cloud_flush(local_path, shard_name, max_retries=5):
    """Synchronous upload called by consumer thread - absolute destination formatting strictly locked down."""
    retries = 0
    backoff_factor = 3

    strict_destination = f"gdrive:JEPA_Shards/preprocessed/{shard_name}"

    while retries < max_retries:
        try:
            result = subprocess.run(
                RCLONE_BASE + [
                    "copyto", str(local_path), strict_destination,
                    "--drive-chunk-size", "64M",
                    "--retries", "3",
                    "--low-level-retries", "5",
                    "--timeout", "60s"
                ],
                capture_output=True, text=True, timeout=180
            )
            if result.returncode == 0:
                logging.info(f"[IO-WORKER] ✅ Successfully uploaded {shard_name} strictly to target folder (Attempt {retries + 1})")
                try: os.remove(local_path)
                except Exception as de: logging.warning(f"[IO-WORKER] Local scratch cleanup stalled: {de}")
                return
            else:
                logging.error(f"[IO-WORKER] ❌ Upload error for {shard_name} [Attempt {retries + 1}/{max_retries}]: {result.stderr}")
        except subprocess.TimeoutExpired:
            logging.error(f"[IO-WORKER] ❌ Hard timeout limit exceeded for {shard_name} on attempt {retries + 1}")
        except Exception as e:
            logging.error(f"[IO-WORKER] ❌ Operational loop breakdown on {shard_name}: {e}")

        retries += 1
        if retries < max_retries:
            sleep_duration = backoff_factor ** retries
            logging.warning(f"[IO-WORKER] Throttling network link. Backing off execution for {sleep_duration}s...")
            time.sleep(sleep_duration)

    logging.critical(f"[IO-WORKER] 💥 FATAL: Shard {shard_name} dropped out of the pipeline permanently after {max_retries} attempts.")

def upload_consumer_loop():
    """Continuous background queue worker that processes uploads sequentially."""
    while True:
        task = UPLOAD_QUEUE.get()
        if task is None:
            break
        local_path, shard_name = task
        sync_cloud_flush(local_path, shard_name)
        UPLOAD_QUEUE.task_done()

# =====================================================================
# 4. Main Preprocessor Loop (Double-Gate Isolated)
# =====================================================================
def run_preprocessor(chunk_size=10000, output_dir="./processed_shards", domains=None, max_datasets=None, upload_to_gdrive=False, enable_gating=False):
    local_scratch = Path(output_dir)
    local_scratch.mkdir(parents=True, exist_ok=True)

    logging.info("Initializing Tokenizers...")
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    SPECIAL_LOAD_ARGS = {
        "HuggingFaceFW/fineweb-edu": [{"name": "sample-100BT", "split": "train"}],
        "Qwen/AgentWorldBench": [{"path": "text", "data_files": "hf://datasets/Qwen/AgentWorldBench/*_test.jsonl", "split": "train"}],
        "bigcode/starcoder2-instruct": [{"split": "train"}],
        "mfielding92/gemini-3.1-pro-2048-reasoning-1100x": [{"split": "train"}],
        "jedisct1/security-audits": [{"name": "all", "split": "train"}],
        "nvidia/compute-eval": [{"split": "eval"}],
        "nvidia/HelpSteer2-Pref": [
            {"path": "nvidia/HelpSteer2", "data_files": "preference/preference.jsonl.gz", "split": "train"},
            {"path": "nvidia/HelpSteer2", "data_files": "preference/preference.jsonl.gz", "split": "validation"}
        ],
        "nvidia/Nemotron-CC-Math-v1": [
            {"name":"3", "split": "train"},
            {"name":"4plus", "split": "train"},
            {"name":"4pus_MIND", "split": "train"}
        ],
        "nvidia/Nemotron-Pretraining-SFT-v1": [
            {"name":"3", "split": "train"},
            {"name":"4plus", "split": "train"},
            {"name":"4pus_MIND", "split": "train"}
        ],
        "nvidia/Nemotron-Pretraining-Code-v2": [
            {"name":"3", "split": "train"},
            {"name":"4plus", "split": "train"},
            {"name":"4pus_MIND", "split": "train"}
        ],
        "allenai/c4": [
            {"name": "en", "split": "train"},
            {"name": "zh", "split": "train"},
            {"name": "es", "split": "train"},
            {"name": "fr", "split": "train"},
            {"name": "de", "split": "train"},
            {"name": "en", "split": "validation"},
            {"name": "zh", "split": "validation"},
            {"name": "multilingual", "split": "train"},
            {"name": "multilingual", "split": "validation"}
        ],
        "nvidia/Open-SWE-Traces": [
            {"name": "openhands", "split": "minimax_m25"},
            {"name": "openhands", "split": "qwen35_122b"},
            {"name": "sweagent", "split": "minimax_m25"},
            {"name": "sweagent", "split": "qwen35_122b"}
        ],
        "liwu/MNBVC": [
            {"path": "json", "data_files": "hf://datasets/liwu/MNBVC/wiki/**/*.jsonl.gz", "split": "train", "name": "wiki"},
            {"path": "json", "data_files": "hf://datasets/liwu/MNBVC/qa/**/*.jsonl.gz", "split": "train", "name": "qa"},
            {"path": "json", "data_files": "hf://datasets/liwu/MNBVC/news/**/*.jsonl.gz", "split": "train", "name": "news"}
        ],
        "nvidia/Nemotron-Pretraining-Specialized-v1.2": [
            {"name": "Nemotron-Pretraining-Fact-Seeking", "split": "train"},
            {"name": "Nemotron-Pretraining-Moral-Scenarios", "split": "train"},
            {"name": "Nemotron-Pretraining-Generative", "split": "train"},
            {"name": "Nemotron-Pretraining-Multiple-Choice", "split": "train"}
        ],
        "nvidia/Nemotron-Pretraining-Legal-v1": [
            {"name": "Nemotron-Pretraining-Legal-Case-Law-Summary", "split": "train"},
            {"name": "Nemotron-Pretraining-Legal-CaseHOLD", "split": "train"},
            {"name": "Nemotron-Pretraining-Legal-eCFR-QA", "split": "train"},
            {"name": "Nemotron-Pretraining-Legal-GlobalCit", "split": "train"},
            {"name": "Nemotron-Pretraining-Legal-Definition-Classification", "split": "train"},
            {"name": "Nemotron-Pretraining-Legal-LegalBench-CUAD-v2", "split": "train"},
            {"name": "Nemotron-Pretraining-Legal-ToS-Clause-Understanding", "split": "train"}
        ],
        "nvidia/Nemotron-SFT-Science-v2": [
            {"name": "rqa", "split": "train"},
            {"name": "so", "split": "train"},
            {"name": "syn_mcq", "split": "train"}
        ],
        "nvidia/Nemotron-SFT-ARC-AGI-v1": [
            {"name": "large_reasoning_and_tools", "split": "train"},
            {"name": "large_tools_no_reasoning", "split": "train"},
            {"name": "large_reasoning_no_tools", "split": "train"}
        ],
        "Salesforce/wikitext": [
            {"name": "wikitext-103-v1", "split": "train"},
            {"name": "wikitext-103-v1", "split": "validation"},
            {"name": "wikitext-103-v1", "split": "test"}
        ],
        "nvidia/Nemotron-RL-Multichallenge-v1": [
            {"name": "advanced", "split": "train"},
            {"name": "vanilla", "split": "train"}
        ],
        "nvidia/Nemotron-RL-ARC-AGI-v1": [
            {"name": "python_inductive", "split": "train"},
            {"name": "transductive", "split": "train"}
        ],
        "legacy-datasets/wikipedia": [
            {"name": "20220301.en", "split": "train"},
            {"name": "20220301.zh", "split": "train"},
            {"name": "20220301.es", "split": "train"},
            {"name": "20220301.fr", "split": "train"},
            {"name": "20220301.de", "split": "train"}
        ],
        "code-search-net/code_search_net": [
            {"name": "python", "split": "train"},
            {"name": "java", "split": "train"},
            {"name": "javascript", "split": "train"},
            {"name": "go", "split": "train"},
            {"name": "ruby", "split": "train"},
            {"name": "php", "split": "train"}
        ],
        "bigcode/the-stack": [
            {"data_dir": "data/python", "name": "python", "split": "train"},
            {"data_dir": "data/rust", "name": "rust", "split": "train"},
            {"data_dir": "data/cpp", "name": "cpp", "split": "train"},
            {"data_dir": "data/c", "name": "c", "split": "train"},
            {"data_dir": "data/javascript", "name": "javascript", "split": "train"},
            {"data_dir": "data/java", "name": "java", "split": "train"}
        ],
        "deepmind/code_contests": [
            {"split": "train"},
            {"split": "valid"},
            {"split": "test"}
        ],
        "Salesforce/wikisql": [
            {"split": "train"},
            {"split": "validation"},
            {"split": "test"}
        ],
        "nvidia/HelpSteer2": [
            {"split": "train"},
            {"split": "validation"}
        ],
        "nvidia/Nemotron-SFT-Multilingual-v2": [
            {"name": "ko", "split": "train"},
            {"name": "ja", "split": "train"}
        ],
        "openbmb/Ultra-FineWeb": [
            {"split": "en"},
            {"split": "zh"}
        ],
        "openbmb/Ultra-FineWeb-L3": [
            {"name": "Ultra-FineWeb-L3-en-Multi-Style-Synthetic", "split": "train"},
            {"name": "Ultra-FineWeb-L3-zh-Multi-Style-Synthetic", "split": "train"},
            {"name": "Ultra-FineWeb-L3-zh-QA-Synthetic", "split": "train"},
            {"name": "Ultra-FineWeb-L3-en-QA-Synthetic", "split": "train"}
        ],
        "openbmb/UltraData-Math": [
            {"name": "UltraData-Math-L1", "split": "train"},
            {"name": "UltraData-Math-L2-preview", "split": "train"},
            {"name": "UltraData-Math-L3-Conversation-Synthetic", "split": "train"},
            {"name": "UltraData-Math-L3-Conversation-Synthetic", "split": "train"},
            {"name": "UltraData-Math-L3-QA-Synthetic", "split": "train"},
            {"name": "UltraData-Math-L3-Textbook-Exercise-Synthetic", "split": "train"}
        ],
        "openbmb/UltraData-SFT-2605": [
            {"name": "Chinese-general", "split": "think"},
            {"name": "Chinese-general", "split": "no_think"},
            {"name": "Code", "split": "think"},
            {"name": "Code", "split": "no_think"},
            {"name": "IF", "split": "think"},
            {"name": "IF", "split": "no_think"},
            {"name": "Knowledge", "split": "think"},
            {"name": "Knowledge", "split": "no_think"},
            {"name": "Math", "split": "think"},
            {"name": "Math", "split": "no_think"},
            {"name": "Multi-lang-Knowledge", "split": "no_think"},
            {"name": "Multi-lang-Math", "split": "no_think"}
        ]
    }

    if upload_to_gdrive:
        threading.Thread(target=upload_consumer_loop, daemon=True).start()
        logging.info("[IO-SYSTEM] Bounded sequential upload manager initialized successfully.")

    target_queue = []
    for domain, datasets in DATASET_QUEUE.items():
        if domains is None or domain in domains:
            for ds in datasets[:max_datasets] if max_datasets else datasets:
                target_queue.append((domain, ds))

    for domain, dataset_name in target_queue:
        load_configs = SPECIAL_LOAD_ARGS.get(dataset_name, [{"path": dataset_name, "split": "train"}])

        for config_opts in load_configs:
            config_name = config_opts.get("name", "default")
            split_name = config_opts.get("split", "train")

            safe_name = sanitize_dataset_name(f"{dataset_name}_{config_name}_{split_name}")

            # === DOUBLE-GATE DISTRIBUTED LOCKING ===
            if enable_gating:
                try:
                    res_chk = subprocess.run(
                        RCLONE_BASE + ["lsf", "gdrive:JEPA_Shards/locks/", "--timeout", "30s"],
                        capture_output=True, text=True
                    )
                    if res_chk.returncode != 0:
                        if res_chk.returncode == 1 or "not found" in res_chk.stderr.lower():
                            pre_check = []
                        else:
                            raise subprocess.CalledProcessError(res_chk.returncode, res_chk.args, output=res_chk.stdout, stderr=res_chk.stderr)
                    else:
                        pre_check = [f.strip() for f in res_chk.stdout.splitlines()]
                except Exception as le:
                    err_details = getattr(le, 'stderr', str(le)).strip() if getattr(le, 'stderr', None) else str(le)
                    logging.warning(f"⚠️ Gating interface fault reading locks directory for {safe_name}: {err_details}. Defensive yield.")
                    continue

                if any(f.startswith(f"{domain}_{safe_name}_active_") for f in pre_check):
                    logging.info(f"⏭️ YIELD: {safe_name} is locked by another node.")
                    continue

                # Claim phase
                worker_uuid = uuid.uuid4().hex[:8]
                claim_filename = f"{domain}_{safe_name}_claim_{worker_uuid}.lock"
                active_filename = f"{domain}_{safe_name}_active_{uuid.uuid4().hex[:8]}.lock"

                Path("local.lock").write_text("lock_claim", encoding="utf-8")
                subprocess.run(RCLONE_BASE + ["copyto", "local.lock", f"gdrive:JEPA_Shards/locks/{claim_filename}"], stderr=subprocess.DEVNULL)

                time.sleep(4)  # Settling window

                try:
                    res_lks = subprocess.run(
                        RCLONE_BASE + ["lsf", "gdrive:JEPA_Shards/locks/", "--timeout", "30s"],
                        capture_output=True, text=True
                    )
                    if res_lks.returncode != 0:
                        if res_lks.returncode == 1 or "not found" in res_lks.stderr.lower():
                            locks = []
                        else:
                            raise subprocess.CalledProcessError(res_lks.returncode, res_lks.args, output=res_lks.stdout, stderr=res_lks.stderr)
                    else:
                        locks = [f.strip() for f in res_lks.stdout.splitlines()]
                except Exception as le:
                    logging.error(f"⚠️ Network timeout verifying claim pool for {safe_name}: {le}. Evicting local trace node safely.")
                    subprocess.run(RCLONE_BASE + ["delete", f"gdrive:JEPA_Shards/locks/{claim_filename}"], stderr=subprocess.DEVNULL)
                    continue

                if any(f.startswith(f"{domain}_{safe_name}_active_") for f in locks):
                    logging.info(f"⏭️ YIELD: Another node claimed {safe_name}.")
                    subprocess.run(RCLONE_BASE + ["delete", f"gdrive:JEPA_Shards/locks/{claim_filename}"], stderr=subprocess.DEVNULL)
                    continue

                dataset_claims = sorted([f for f in locks if f.startswith(f"{domain}_{safe_name}_claim_")])
                if dataset_claims and dataset_claims[0] != claim_filename:
                    logging.info(f"⏭️ YIELD: Lost claim race for {safe_name}. Winner: {dataset_claims[0]}")
                    subprocess.run(RCLONE_BASE + ["delete", f"gdrive:JEPA_Shards/locks/{claim_filename}"], stderr=subprocess.DEVNULL)
                    continue

                # Upgrade to active
                subprocess.run(RCLONE_BASE + ["copyto", "local.lock", f"gdrive:JEPA_Shards/locks/{active_filename}"], stderr=subprocess.DEVNULL)
                subprocess.run(RCLONE_BASE + ["delete", f"gdrive:JEPA_Shards/locks/{claim_filename}"], stderr=subprocess.DEVNULL)
                Path("local.lock").unlink(missing_ok=True)

                logging.info(f"✅ CLAIMED {safe_name} via Double-Gate protocol.")

            logging.info(f"Processing {safe_name} in domain {domain}...")

            if domain == "frontier_traces":
                max_rows = 500_000
            elif dataset_name == "wdndev/webnovel-chinese" or dataset_name == "WebNovelTrans/kunpeng-sentencepairs-5m-instruction":
                max_rows = 5_000_000
            elif domain == "general_knowledge" or dataset_name == "nvidia/Nemotron-Pretraining-Code-v3":
                max_rows = 20_000_000
            else:
                max_rows = 1_000_000

            buffer = []
            chunk_id = 0
            rows_processed = 0

            load_args = config_opts.copy()
            load_args["streaming"] = True  # Strict enforcement of pure web streaming pipelines
            if "path" not in load_args:
                load_args["path"] = dataset_name

            is_teich_format = False

            try:
                ds = load_dataset(**load_args)
            except Exception as e:
                logging.warning(f"Standard config load failed for {safe_name}: {e}. Trying fallback with name='default'...")
                try:
                    fallback_args_default = load_args.copy()
                    fallback_args_default["name"] = "default"
                    ds = load_dataset(**fallback_args_default)
                except Exception as e_default:
                    logging.warning(f"Default config load failed for {safe_name}: {e_default}. Trying JSON fallback...")
                    try:
                        fallback_args = {
                            "path": "text",
                            "data_files": f"hf://datasets/{dataset_name}/**/*.json*",
                            "split": split_name,
                            "streaming": True
                        }
                        ds = load_dataset(**fallback_args)
                    except Exception:
                        try:
                            logging.info(f"Deploying Teich trace fallback for {safe_name}...")
                            from teich import load_traces
                            ds = load_traces(dataset_name)
                            is_teich_format = True
                        except Exception as e_teich:
                            logging.error(f"Critical data fetch failure for {safe_name}: {e_teich}")
                            if enable_gating:
                                subprocess.run(RCLONE_BASE + ["delete", f"gdrive:JEPA_Shards/locks/{active_filename}"], stderr=subprocess.DEVNULL)
                            continue

            # =====================================================================
            # 4.1 Cloud Defrost Sequential First-Gap Resume Logic with Strict Metadata Guard
            # =====================================================================
            try:
                res_exist = subprocess.run(
                    RCLONE_BASE + ["lsf", "gdrive:JEPA_Shards/preprocessed/"],
                    capture_output=True, text=True
                )
                existing = res_exist.stdout.splitlines() if res_exist.returncode == 0 else []
                existing = [f.strip() for f in existing]
            except Exception:
                existing = []

            found_chunks = []
            for f in existing:
                if f.startswith(f"{domain}_{safe_name}_"):
                    match = re.search(r"_(\d+)\.pt$", f)
                    if match:
                        found_chunks.append(int(match.group(1)))

            # --- DEFROST VERIFICATION TRACK ---
            if found_chunks:
                min_c, max_c = min(found_chunks), max(found_chunks)
                for expected_chunk in range(min_c, max_c + 1):
                    if expected_chunk not in found_chunks:
                        check_file = f"{domain}_{safe_name}_{expected_chunk:06d}.pt"
                        chk_res = subprocess.run(
                            RCLONE_BASE + ["lsf", f"gdrive:JEPA_Shards/preprocessed/{check_file}"],
                            capture_output=True, text=True
                        )
                        if chk_res.returncode == 0 and chk_res.stdout.strip():
                            logging.info(f"[STAGE1-DEFROST] Force-recovered cached index gap: {check_file}")
                            found_chunks.append(expected_chunk)

            # --- CRITICAL FIX: ABSOLUTE CONTINUITY CAPABILITY ---
            resume_chunk = 0
            if found_chunks:
                while resume_chunk in found_chunks:
                    resume_chunk += 1
                chunk_id = resume_chunk
            else:
                chunk_id = 0

            rows_to_skip = chunk_id * chunk_size

            # --- EXTRACTION OF REMOTE DATASET ROW SPLIT COUNTS ---
            ds_total_rows = float('inf')
            if hasattr(ds, 'info') and ds.info is not None:
                splits = getattr(ds.info, 'splits', None)
                if splits and split_name in splits:
                    num_examples = getattr(splits[split_name], 'num_examples', None)
                    if num_examples is not None and num_examples > 0:
                        ds_total_rows = num_examples

            # Absolute Guard Check: Halt instantly if we cross our arbitrary cap
            if rows_to_skip >= max_rows:
                logging.info(f"🛑 [CAP-STOP] Shard index path {chunk_id} satisfies arbitrary allocation constraints ({rows_to_skip}/{max_rows} rows mapped). Skipping.")
                if enable_gating:
                    subprocess.run(RCLONE_BASE + ["delete", f"gdrive:JEPA_Shards/locks/{active_filename}"], stderr=subprocess.DEVNULL)
                continue

            # Absolute Guard Check: Halt instantly if rows to skip exceeds true Hugging Face repository bounds
            if rows_to_skip >= ds_total_rows:
                logging.info(f"🛑 [DATASET-END] Requested resume rows ({rows_to_skip}) exceeds actual total dataset size ({ds_total_rows} examples). Shards complete. Skipping.")
                if enable_gating:
                    subprocess.run(RCLONE_BASE + ["delete", f"gdrive:JEPA_Shards/locks/{active_filename}"], stderr=subprocess.DEVNULL)
                continue

            logging.info(f"Resuming {safe_name} starting at target missing chunk {chunk_id}. Aligning pointers...")

            # --- METADATA EXPLOIT RESILIENCY HANDSHAKE ---
            max_handshake_retries = 3
            handshake_success = False
            ds_iterator = None
            first_row = None

            for attempt in range(max_handshake_retries):
                try:
                    if rows_to_skip > 0 and not is_teich_format:
                        logging.info(f"   [NETWORK-HANDSHAKE] Engaging skip sequence to bypass {rows_to_skip} rows (Attempt {attempt + 1}/{max_handshake_retries})...")
                        skip_start = time.time()

                        optimized_ds = ds.skip(rows_to_skip)
                        ds_iterator = iter(optimized_ds)

                        first_row = next(ds_iterator)
                        logging.info(f"   [NETWORK-HANDSHAKE] Pointer alignment locked in {time.time() - skip_start:.2f}s.")
                    else:
                        ds_iterator = iter(ds)
                        if rows_to_skip > 0 and is_teich_format:
                            for _ in range(rows_to_skip):
                                next(ds_iterator)

                    handshake_success = True
                    break
                except StopIteration:
                    logging.warning(f"   ⚠️ Dataset split contains fewer than {rows_to_skip} total items.")
                    ds_iterator = None
                    handshake_success = True
                    break
                except Exception as e:
                    backoff_sleep = 6 * (attempt + 1)
                    logging.warning(f"   ⚠️ Handshake execution crash on attempt {attempt + 1}: {e}. Re-throttling channel for {backoff_sleep}s...")
                    time.sleep(backoff_sleep)

            if not handshake_success or (ds_iterator is None and rows_to_skip > 0 and not is_teich_format):
                if ds_iterator is None and handshake_success:
                    if enable_gating:
                        subprocess.run(RCLONE_BASE + ["delete", f"gdrive:JEPA_Shards/locks/{active_filename}"], stderr=subprocess.DEVNULL)
                    continue
                logging.error(f"💥 [HANDSHAKE-FAILED] Unable to stabilize network socket for {safe_name} after {max_handshake_retries} connection attempts. Evicting node lock safely.")
                if enable_gating:
                    subprocess.run(RCLONE_BASE + ["delete", f"gdrive:JEPA_Shards/locks/{active_filename}"], stderr=subprocess.DEVNULL)
                continue

            rows_processed = rows_to_skip
            # =====================================================================

            try:
                start_time = time.time()
                raw_batch_size = 512

                while True:
                    if rows_processed >= max_rows:
                        logging.info(f"🏁 [CAP-REACHED] Upper sequence constraint limits touched ({rows_processed}/{max_rows}). Terminating compilation tracker.")
                        break

                    # --- INLINE MID-STREAM ITERATOR FF UPGRADE ---
                    current_shard_name = f"{domain}_{safe_name}_{chunk_id:06d}.pt"
                    if found_chunks and chunk_id in found_chunks:
                        logging.info(f"⏩ [GAP-HEALER] Shard {current_shard_name} verified on Drive. Advancing iterator past cache block...")
                        for _ in range(chunk_size):
                            try: next(ds_iterator)
                            except StopIteration: break
                        rows_processed += chunk_size
                        chunk_id += 1
                        start_time = time.time()
                        continue
                    # ---------------------------------------------

                    valid_prompts = []
                    valid_responses = []

                    # If we have a verified record from the handshake phase, inject it as the first element of our first batch array
                    if first_row is not None:
                        prompt, response = extract_qa_pair(first_row, dataset_name)
                        if domain in ["frontier_traces", "code_mechanics"]:
                            if not is_verifiable_logic(response):
                                pass
                            else:
                                if prompt and response:
                                    valid_prompts.append(prompt)
                                    valid_responses.append(response)
                        else:
                            if prompt and response:
                                valid_prompts.append(prompt)
                                valid_responses.append(response)
                        first_row = None  # Burn container flag immediately

                    for _ in range(raw_batch_size - len(valid_prompts)):
                        try: row = next(ds_iterator)
                        except StopIteration: break
                        except Exception: continue

                        prompt, response = extract_qa_pair(row, dataset_name)

                        if domain in ["frontier_traces", "code_mechanics"]:
                            if not is_verifiable_logic(response):
                                continue # Drop unverifiable data silently

                        if prompt and response:
                            valid_prompts.append(prompt)
                            valid_responses.append(response)

                    if not valid_prompts:
                        break

                    in_encodings = tokenizer(valid_prompts, truncation=True, max_length=2048)["input_ids"]
                    qw_encodings = tokenizer(valid_responses, truncation=True, max_length=2048)["input_ids"]

                    for p_toks, r_toks, raw_resp in zip(in_encodings, qw_encodings, valid_responses):
                        if rows_processed >= max_rows:
                            break

                        in_tensor = torch.tensor(p_toks, dtype=torch.long)
                        qw_tensor = torch.tensor(r_toks, dtype=torch.long)
                        if in_tensor.numel() == 0 or qw_tensor.numel() == 0:
                            continue
                        buffer.append({
                            "prompt_tokens": in_tensor,
                            "response_tokens": qw_tensor,
                            "raw_response": raw_resp
                        })
                        rows_processed += 1
                        if len(buffer) >= chunk_size:
                            shard_name = f"{domain}_{safe_name}_{chunk_id:06d}.pt"
                            local_shard_path = local_scratch / shard_name

                            torch.save(buffer, local_shard_path)

                            elapsed = time.time() - start_time
                            throughput = len(buffer) / elapsed if elapsed > 0 else 0
                            logging.info(f"✅ Saved chunk {shard_name} ({throughput:.2f} rows/sec) | Total rows: {rows_processed}")

                            if upload_to_gdrive:
                                UPLOAD_QUEUE.put((local_shard_path, shard_name))

                            buffer = []
                            chunk_id += 1
                            start_time = time.time()

                    # --- ACTIVE BATCH LOOP PROGRESS HEARTBEAT ---
                    if valid_prompts:
                        logging.info(f"   --> Shard Buffer: {len(buffer)}/{chunk_size} rows collected | Progress: {rows_processed}/{max_rows} dataset rows")

            except Exception as e:
                logging.error(f"Fault on {safe_name}: {e}")

            if buffer:
                shard_name = f"{domain}_{safe_name}_{chunk_id:06d}.pt"
                if not (found_chunks and chunk_id in found_chunks):
                    local_shard_path = local_scratch / shard_name
                    torch.save(buffer, local_shard_path)
                    logging.info(f"✅ Saved final chunk {shard_name} with {len(buffer)} rows")

                    if upload_to_gdrive:
                        UPLOAD_QUEUE.put((local_shard_path, shard_name))

            if enable_gating:
                subprocess.run(RCLONE_BASE + ["delete", f"gdrive:JEPA_Shards/locks/{active_filename}"], stderr=subprocess.DEVNULL)

    logging.info("Preprocessing completed!")

# =====================================================================
# 5. Environment Self-Awareness Selector Entry Point
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Autonomous JEPA Preprocessor Engine")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Rows per shard")
    parser.add_argument("--output-dir", type=str, default="./processed_shards", help="Output directory for .pt shards")
    parser.add_argument("--domains", nargs="+", choices=["frontier_traces", "general_knowledge", "code_mechanics"],
                        help="Specific domains to process (default: all)")
    parser.add_argument("--max-datasets", type=int, help="Limit number of datasets per domain for testing")
    parser.add_argument("--upload-gdrive", action="store_true", help="Upload shards to Google Drive via rclone")
    parser.add_argument("--enable-gating", action="store_true", help="Enable distributed locking via rclone")
    args = parser.parse_args()

    is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle')

    upload = True if is_kaggle else args.upload_gdrive
    gating = True if is_kaggle else args.enable_gating

    if is_kaggle:
        logging.info("🌐 Environment Discovery: Headless Kaggle Cloud Container Node Confirmed.")
        logging.info("--> Activating absolute Double-Gate parameters and Cloud Upload pipelines automatically.")
    else:
        logging.info("🖥️ Environment Discovery: Local Machine Execution Track Confirmed.")

    run_preprocessor(
        chunk_size=args.chunk_size,
        output_dir=args.output_dir,
        domains=args.domains,
        max_datasets=args.max_datasets,
        upload_to_gdrive=upload,
        enable_gating=gating
    )