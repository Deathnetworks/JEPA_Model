from datasets import load_dataset
import torch

class StreamingFrontierDataset:
    def __init__(self, config):
        self.dataset_name = config['data']['dataset_name']
        self.seq_len = config['data']['seq_len']

        # Load dataset in streaming mode to save local disk space
        self.dataset = load_dataset(self.dataset_name, streaming=True)

    def get_iterator(self, split='train', skip_items=0):
        ds = self.dataset[split]

        # Fast-forwarding logic native to huggingface streaming dataset
        if skip_items > 0:
            ds = ds.skip(skip_items)

        def _generator():
            for item in ds:
                # Mock tokenization/featurization
                yield {
                    "input_ids": torch.randint(0, 1000, (self.seq_len,)),
                    "labels": torch.randint(0, 1000, (self.seq_len,))
                }

        return _generator()
