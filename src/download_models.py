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
    "Qwen/Qwen3.6-27B",
    "BAAI/bge-large-en-v1.5"
]

DATASETS_TO_DOWNLOAD = [
    "AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset",
    "TheAgenticAI/Agentic-Reasoning"
]

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
    for dataset_id in DATASETS_TO_DOWNLOAD:
        logger.info(f"Downloading/Verifying dataset: {dataset_id} (train split only)")
        try:
            load_dataset(dataset_id, split="train")
            logger.info(f"Successfully cached dataset: {dataset_id}")
        except Exception as e:
            logger.error(f"Failed to download dataset {dataset_id}: {e}")
            sys.exit(1)

    logger.info("Pre-flight download completed successfully. All assets are cached.")

if __name__ == "__main__":
    preflight_download()
