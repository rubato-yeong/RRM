"""
    4_get_avg_activation.py

    This script extracts and saves median activation values from a cascaded ViT-based model (e.g., with SAE modules).
    It performs a single forward pass through a batch of ImageNet validation images and stores the median activations
    of features, neurons, and reconstruction errors.

    These median values are used later for relevance-based faithfulness evaluation with Top-K SAE modules.

    Main functionalities:
        - Loads pretrained cascaded model (ViT/DINOv2/CLIP-ViT + TopKSAE)
        - Loads ImageNet validation dataset
        - Performs one forward pass in "median" mode to extract activations
        - Saves the following activations as .pkl files:
            - median_feature_acts_{model}.pkl
            - median_error_acts_{model}.pkl
            - median_neuron_acts_{model}.pkl

    Arguments:
        --model: Model type ['vit', 'dinov2', 'clip_vit']
        --device: Device string (e.g., 'cuda:0')
        --batch_size: Batch size for activation extraction
        --seed: Random seed for reproducibility
        --data_root: Root directory of ImageNet validation set
        --cache_path: Path to SAE activation cache directory
        --sae_ckpt_path: Path to directory containing TopKSAE checkpoints
        --dino_head_ckpt_path: Path to pretrained DINOv2 classification head
        --clip_text_embed_path: Path to CLIP text feature embeddings
        --output_path: Directory where median activation .pkl files will be saved

    Example:
        python 4_get_avg_activation.py \
            --model clip_vit \
            --device cuda:1 \
            --batch_size 1024 \
            --data_root /path/to/imagenet/val \
            --output_path /path/to/output/median_acts \
            --sae_ckpt_path /path/to/checkpoints \
            --clip_text_embed_path /path/to/text_features.pt
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import set_seed
from utils import create_cascaded_model, get_imagenet_transform, ImageNetValDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and save median activations for SAE-based models")

    # Model and computation settings
    parser.add_argument("--model", type=str, required=True, choices=["vit", "dinov2", "clip_vit"], help="Model type")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for processing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Path settings
    parser.add_argument("--data_root", type=str, default="/nfs/home/junhyeok/data/ILSVRC/Data/CLS-LOC/val", help="Path to ImageNet val dataset")  # E.g., /USER/ILSVRC/Data/CLS-LOC/val/
    parser.add_argument("--output_path", type=str, default="./avg_acts", help="Directory to save median activation .pkl files")  # E.g., PROJECT_ROOT/avg_acts/
    parser.add_argument("--cache_path", type=str, default="/nfs/home/junhyeok/neurips2025-camera/cache", help="Path to activation cache")  # E.g., PROJECT_ROOT/cache/
    parser.add_argument("--sae_ckpt_path", type=str, default="", help="Path to Top-K SAE checkpoints")  # E.g., PROJECT_ROOT/checkpoints/imagenet/
    parser.add_argument("--dino_head_ckpt_path", type=str, default="/nfs/home/junhyeok/neurips2025-camera/vision_checkpoints/dinov2_vitb14_reg4_linear_head.pth", help="Dinov2 classification head checkpoint. Refer to: https://github.com/facebookresearch/dinov2")  # E.g., PROJECT_ROOT/vision_heads/dinov2_vitb14_reg4_linear_head.pth
    parser.add_argument("--clip_text_embed_path", type=str, default="/nfs/home/junhyeok/neurips2025-camera/clip_heads/imagenet.pt", help="CLIP text features tensor file. Its shape must be (Number of classes, Embedding dimension) Refer to: https://colab.research.google.com/github/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb#scrollTo=4W8ARJVqBJXs")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    act_model = create_cascaded_model(args, mode="act")

    # Load dataset & dataloader
    dataset = ImageNetValDataset(
        image_dir=Path(args.data_root),
        annotation_dir=os.path.join(Path(args.data_root).parents[2], "Annotations/CLS-LOC/val"),
        transform=get_imagenet_transform(),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    with torch.no_grad():
        for image, label in tqdm(dataloader):  # Sample one batch
            image = image.to(args.device)
            label = label.to(args.device)
            act_model(image, type="median")
            break  # Only one batch needed

    # Save extracted median activations
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, f"median_feature_acts_{args.model}.pkl"), "wb") as f:
        pickle.dump(act_model.feature_acts, f)
    with open(os.path.join(args.output_path, f"median_error_acts_{args.model}.pkl"), "wb") as f:
        pickle.dump(act_model.error_acts, f)
    with open(os.path.join(args.output_path, f"median_neuron_acts_{args.model}.pkl"), "wb") as f:
        pickle.dump(act_model.neuron_acts, f)


if __name__ == "__main__":
    main()
