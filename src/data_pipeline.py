from datasets import load_dataset
import torch
from transformers import AutoTokenizer

class StreamingFrontierDataset:
    def __init__(self, config):
        self.dataset_name = config['data']['dataset_name']
        self.seq_len = config['data']['seq_len']

        # We load a Qwen tokenizer as the baseline for this project
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        # Assuming we can pad right
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset in streaming mode to save local disk space
        self.dataset = load_dataset(self.dataset_name, streaming=True)

    def get_iterator(self, split='train', skip_items=0):
        # We usually fallback to 'train' if the specific split isn't available
        try:
            ds = self.dataset[split]
        except KeyError:
            ds = self.dataset['train']

        # Fast-forwarding logic native to huggingface streaming dataset
        if skip_items > 0:
            ds = ds.skip(skip_items)

        def _generator():
            for item in ds:
                # Actual tokenization based on common HF dataset structures
                text = item.get('text', item.get('content', item.get('prompt', '')))
                if not text:
                    continue

                tokens = self.tokenizer(
                    text,
                    max_length=self.seq_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                input_ids = tokens['input_ids'].squeeze(0)

                # Standard causal LM labels setup
                labels = input_ids.clone()
                labels[tokens['attention_mask'].squeeze(0) == 0] = -100

                yield {
                    "input_ids": input_ids,
                    "labels": labels
                }

        return _generator()
