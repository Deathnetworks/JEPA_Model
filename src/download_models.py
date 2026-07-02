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
    "frontier_traces": [
        "Crownelius/Complete-FABLE.5-traces-2M",
        "Qwen/AgentWorldBench",
        "nvidia/HelpSteer2",
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
