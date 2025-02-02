import argparse
from dino_tracker import DINOTracker
from models.utils import fix_random_seeds
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default=Path(__file__).parent / "train.yaml", type=str)
    parser.add_argument("--data-path", default=Path(__file__).parent.parent.parent / "mesh_videos/dino-tracker-textured", type=str)
    parser.add_argument("--seed", default=2, type=int)
    args = parser.parse_args()

    fix_random_seeds(args.seed)
    dino_tracker = DINOTracker(args)
    dino_tracker.train()
