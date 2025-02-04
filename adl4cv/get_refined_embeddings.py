import argparse
import logging

import numpy as np
from sklearn.decomposition import PCA
import torch
from dino_tracker import DINOTracker
from models.tracker import Tracker
from pathlib import Path
from PIL import Image

from models.utils import fix_random_seeds

def reshape_embeddings(embeddings, h, w):
    """
    Reshape embeddings from (T, C, H, W) to (T, h, w, C).

    Args:
        embeddings (torch.Tensor): Input tensor of shape (T, C, H, W).
        h (int): Target height for reshaped embeddings.
        w (int): Target width for reshaped embeddings.

    Returns:
        torch.Tensor: Reshaped tensor of shape (T, h, w, C).
    """
    T, C, H, W = embeddings.shape
    if H != h or W != w:
        embeddings = torch.nn.functional.interpolate(
            embeddings, size=(h, w), mode="bilinear", align_corners=False
        )
    embeddings = embeddings.permute(0, 2, 3, 1)  # Rearrange to (T, h, w, C)
    return embeddings

def display_pca_features(features, idx, path):

    features = embeddings[idx]
    # Reshape features for PCA
    features_reshaped = features.reshape(-1, features.shape[2]).detach().cpu().numpy()

    # Apply PCA to reduce dimensions to 3
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features_reshaped)

    # Normalize the PCA features to 0-255 for RGB visualization
    features_pca -= features_pca.min(axis=0)
    features_pca /= features_pca.ptp(axis=0)  # Use peak-to-peak (max-min) for normalization
    features_pca *= 255.0
    features_pca = features_pca.astype(np.uint8)

    # Reshape back to image dimensions
    h, w = features.shape[0], features.shape[1]
    feature_img = features_pca.reshape(h, w, 3)

    # # Reshape back to image dimensions
    # h, w = features.shape[2], features.shape[3]
    # feature_img = features_pca.reshape(h, w, 3)

    # Save visualization
    Image.fromarray(feature_img).save(path)

def get_refined_embeddings(config, data_path, seed=2):
    """
    Generate and save refined embeddings for a given dataset.

    Args:
        config (str): Path to the configuration file.
        data_path (str): Path to the dataset.
        seed (int): Random seed for reproducibility.
        h (int): Target height for reshaped embeddings.
        w (int): Target width for reshaped embeddings.
    """
    logging.basicConfig(level=logging.INFO)

    # Initialize the DINO tracker
    class Args:
        def __init__(self, config, data_path, seed):
            self.config = config
            self.data_path = data_path
            self.seed = seed

    args = Args(config=config, data_path=data_path, seed=seed)
    dino_tracker = DINOTracker(args)
    tracker = dino_tracker.get_model()

    # Get refined embeddings
    embeddings, _ = tracker.get_refined_embeddings_cpu()
    # embeddings = reshape_embeddings(embeddings, 512, 512)

    # Save the embeddings
    refined_embeddings_path = Path(tracker.dino_embed_path).parent / "refined_embeddings.pt"
    torch.save(embeddings, refined_embeddings_path)
    print(f"Saved refined embeddings to {refined_embeddings_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/train.yaml", type=str)
    parser.add_argument("--data-path", default="./dataset/rendered_mesh_output", type=str)
    parser.add_argument("--seed", default=2, type=int)
    args = parser.parse_args()

    get_refined_embeddings(
        config=args.config,
        data_path=args.data_path,
        seed=args.seed
    )