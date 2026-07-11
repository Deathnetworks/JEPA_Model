import os
import torch
import pytest
from pathlib import Path
from src.train_latent_loop import train_loop

def test_training_pipeline_smoke(tmp_path):
    # 1. Generate Synthetic Shard
    mock_shard_path = tmp_path / "frontier_traces_mock_1.pt"
    # The dataloader returns batches, each batch has flattened_chunk, each item should be a dict
    # Wait, flatten_chunk should be a list of dicts.
    # The mock data must be a list of lists if dataloader loads shards then returns flattened_chunk
    # Actually train_loop dataloader implementation in train_latent_loop is unknown.
    # Let's inspect what dataloader expects. The previous error shows mini_batch is just an element of flattened_chunk which might be a list.

    mock_data = [
        # This is one shard containing N items
        {"input_tokens": torch.randint(0, 1000, (32,)),
         "qwen_tokens": torch.randint(0, 1000, (32,)),
         "target_concept": torch.randn(1024)},
        {"input_tokens": torch.randint(0, 1000, (32,)),
         "qwen_tokens": torch.randint(0, 1000, (32,)),
         "target_concept": torch.randn(1024)}
    ]
    torch.save(mock_data, mock_shard_path)

    # 2. Run the loop for exactly 2 steps to verify imports, autocast, and tensor shapes
    train_loop(
        epochs=1,
        data_dir=str(tmp_path),
        mini_batch_size=1,
    )

    # 3. Assert checkpoint hygiene
    assert (Path("jepa_engine.pth")).exists(), "Engine checkpoint failed to save."
    assert (Path("latent_decoder.pth")).exists(), "Decoder checkpoint failed to save."
