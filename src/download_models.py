import logging
from huggingface_hub import snapshot_download
from datasets import load_dataset
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

MODELS_TO_DOWNLOAD = [
    "Qwen/Qwen2.5-7B-Instruct",
    "BAAI/bge-large-en-v1.5"
]

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
        "mfielding92/gemini-3.1-pro-2048-reasoning-1100x",
        "rajpurkar/squad",
        "google/boolq",
        "benchflow/skillsbench-leaderboard",
        "evaleval/EEE_datastore"
    ],

    # 2. Massive General Knowledge, Instruction Following, & Creative Core
    "general_knowledge": [
        "HuggingFaceFW/fineweb-edu",
        "HuggingFaceH4/ultrafeedback_clean",
        "technium/OpenHermes-2.5",
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
        "liwu/MNBVC"
    ],

    # 3. Code Syntax & Language Grammar Rules (For the JEPA World Model)
    "code_mechanics": [
        "m-a-p/CodeFeedback-Filtered-Instruction",
        "deepmind/code_contests",
        "code-search-net/code_search_net",
        "bigcode/starcoder2-instruct",
        "iamtarun/python-execution-traces",
        "bigcode/the-stack",
        "bookcorpus/bookcorpus",
        "Salesforce/wikisql",
        "gaianet/learn-rust",
        "semeru/code-code-translation-java-csharp",
        "MehdiFe/csharp-instruction-Dataset",
        "microsoft/LCC_csharp",
        "AlgorithmicResearchGroup/arxiv_cplusplus_research_code",
        "FradSer/DeepSeek-R1-Distilled-Translate-en-zh_CN-39k",
        "FradSer/DeepSeek-R1-Distilled-Translate-en-zh_CN-39k-Alpaca-GPT4",
        "bh2821/LightNovel5000"        
    ]
}

def preflight_download():
    logger.info("Starting Pre-flight Model and Dataset Downloading...")

    # Download models
    logger.info("=== Downloading Models ===")
    for model_id in MODELS_TO_DOWNLOAD:
        logger.info(f"Downloading/Verifying model: {model_id}")
        try:
            snapshot_download(repo_id=model_id, local_files_only=False)
            logger.info(f"Successfully cached model: {model_id}")
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            sys.exit(1)

    # Download datasets
    logger.info("=== Downloading Datasets ===")
    for domain, datasets in DATASET_QUEUE.items():
        for dataset_id in datasets:
            logger.info(f"Downloading/Verifying dataset: {dataset_id} (streaming cache)")
            try:
                # We just load a dummy subset to force cache resolution if not present,
                # though streaming doesn't strictly "download" everything locally beforehand.
                # Just ensuring the dataset is accessible.
                load_args = {"path": dataset_id, "split": "train", "streaming": True}
                if dataset_id == "HuggingFaceFW/fineweb-edu":
                    load_args["name"] = "sample-10BT"
                try:
                    ds = load_dataset(**load_args)
                except Exception:
                    # Retry without train split
                    ds = load_dataset(dataset_id, streaming=True)

                # Fetch one item to test
                next(iter(ds))
                logger.info(f"Successfully verified dataset: {dataset_id}")
            except Exception as e:
                logger.error(f"Failed to verify dataset {dataset_id}: {e}")
                # We don't exit here to allow failure on specific obscure datasets without crashing the whole pipeline

    logger.info("Pre-flight download completed successfully. All accessible assets are cached.")

if __name__ == "__main__":
    preflight_download()
