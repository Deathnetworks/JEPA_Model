import argparse
import yaml
import json
import os

def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            return yaml.safe_load(f)
        elif config_path.endswith(".json"):
            return json.load(f)
        else:
            raise ValueError("Unsupported config format. Use .yaml or .json")

def parse_args():
    parser = argparse.ArgumentParser(description="JEPA_Model v2 Training Script")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    # Add overrides if necessary
    parser.add_argument("--batch_size", type=int, help="Override batch size")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    return config

if __name__ == "__main__":
    cfg = parse_args()
    print("Loaded Config:", cfg)
