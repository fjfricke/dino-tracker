import argparse
import logging
from dino_tracker import DINOTracker
from models.utils import fix_random_seeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="./config/train.yaml", type=str)
    parser.add_argument("--data-path", default="./dataset/rendered_mesh_output", type=str)
    parser.add_argument("--seed", default=2, type=int)
    args = parser.parse_args()

    fix_random_seeds(args.seed)
    logging.basicConfig(level=logging.INFO) 
    dino_tracker = DINOTracker(args)
    dino_tracker.train()
