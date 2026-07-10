import argparse
import logging
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_training_corpus(dataset_name="Salesforce/wikitext", config="wikitext-103-raw-v1"):
    dataset = load_dataset(dataset_name, config, split="train", streaming=True)
    for row in dataset:
        yield row["text"]

def build_tokenizer(vocab_size=32000, dataset_name="Salesforce/wikitext", config="wikitext-103-raw-v1", output_dir="jepa_tokenizer"):
    logging.info(f"Initializing ByteLevel BPE Tokenizer targeting {vocab_size} vocab size...")
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"]
    )

    logging.info("Training from dataset iterator...")
    dataset_iterator = get_training_corpus(dataset_name, config)
    tokenizer.train_from_iterator(dataset_iterator, trainer=trainer)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    logging.info(f"Tokenizer saved to {output_dir}/tokenizer.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--output_dir", type=str, default="jepa_tokenizer")
    args = parser.parse_args()
    build_tokenizer(vocab_size=args.vocab_size, output_dir=args.output_dir)
