import logging
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


import argparse
from dino_tracker import DINOTracker
from models.utils import fix_random_seeds
from pathlib import Path

def main(config, data_path, seed):
    # add data path to config
    args = argparse.Namespace(config=config, data_path=data_path)
    fix_random_seeds(seed)
    logging.basicConfig(level=logging.INFO) 
    dino_tracker = DINOTracker(args)
    dino_tracker.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default=Path(__file__).parent / "train.yaml", type=str)
    parser.add_argument("--data-path", default=Path(__file__).parent.parent.parent / "datasets/rendered_mesh_output", type=str)
    parser.add_argument("--seed", default=2, type=int)
    args = parser.parse_args()

    main(args.config, args.data_path, args.seed)
