"""
    1_extract_activations.py

    Extracts and saves intermediate transformer activations from ViT-based models 
    (e.g., ViT, CLIP-ViT, DINOv2) on image datasets like ImageNet.

    Main features:
        - Supports multiple transformer layers (customizable via --layers)
        - Hooks specified layers and saves flattened activation outputs as .npy
        - Supports partial processing via --batch_start_from and --batch_end_at
        - Compatible with different models: ViT (timm), CLIP (HuggingFace), DINOv2 (Facebook)

    Arguments:
        --model: Model type (vit | clip_vit | dinov2)
        --data_root: Path to dataset directory (e.g., /USER/ILSVRC/Data/CLS-LOC/train)
        --output_root: Directory to save the extracted activations (e.g., /USER/activations)
        --layers: List of transformer layer indices to hook
        --split: Dataset split (train or valid)
        --batch_start_from / --batch_end_at: Index range for partial processing

    Output:
        - Stores activations as NumPy `.npy` files, organized by model, split, and layer

    Usage:
        python 1_extract_activations.py --model vit --data_root /path/to/imagenet --output_root /path/to/output
"""


import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import timm
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import CLIPModel, CLIPProcessor

from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ViT activations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size.")
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], help="Layer indices to hook, 0-indexed.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on.")
    parser.add_argument("--batch_start_from", type=int, default=0, help="Start processing from this batch index (0-based).")
    parser.add_argument("--batch_end_at", type=int, default=None, help="Stop processing after this batch index (exclusive, absolute index).")
    parser.add_argument('--model', choices=['vit', 'clip_vit', 'dinov2'], required=True, help='Model type.')
    parser.add_argument('--split', choices=['train', 'valid'], default='train', help='Dataset split to use.')

    parser.add_argument("--data_root", type=str, default="", help="Path to the dataset root.")  # E.g., /USER/ILSVRC/Data/CLS-LOC/val/
    parser.add_argument("--output_root", type=str, default="", help="Directory to save the activations.")  # E.g., PROJECT_ROOT/activations/

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Model setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    processor = None
    if args.model == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
    elif args.model == 'dinov2':
        model_full = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14_reg_lc').to(device)
        model = model_full.backbone
    elif args.model == 'clip_vit':
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch16", use_fast=True)
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch16").to(device)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    model.eval()

    # Dataset and transforms
    if args.model == 'clip_vit':
        def clip_transform(image):
            return processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        transform = clip_transform
    else:
        transform = Compose([
            Resize((224,224)), ToTensor(),
            Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    # Load dataset
    if "ilsvrc" in args.data_root.lower():  # NOTE: Only train split is supported for imagenet
        assert args.split == "train"
        dataset_name = "imagenet"
        from torchvision.datasets import ImageFolder
        dataset = ImageFolder(root=args.data_root, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.data_root}")

    total_len = len(dataset)
    # Apply start and end batch slicing
    start_idx = args.batch_start_from * args.batch_size
    end_idx = total_len if args.batch_end_at is None else min(
        args.batch_end_at * args.batch_size, total_len)
    if start_idx > 0 or end_idx < total_len:
        dataset = Subset(dataset, range(start_idx, end_idx))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=('train' in args.data_root),
        num_workers=4,
        pin_memory=True
    )

    # Register hooks
    batch_hidden_states = {}
    def get_hook(name):
        def hook(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            batch_hidden_states[name] = hs.detach()
        return hook
    if args.model == 'vit':
        blocks = model.blocks
    elif args.model == 'dinov2':
        blocks = model.blocks
    elif args.model == 'clip_vit':
        blocks = model.vision_model.encoder.layers
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    handles = {}
    for layer in args.layers:
        handles[layer] = blocks[layer].register_forward_hook(
            get_hook(layer))
    print(f"Detected {len(blocks)} transformer blocks.")

    # Prepare output dirs
    output_root = os.path.join(args.output_root, dataset_name, args.split, args.model)
    os.makedirs(output_root, exist_ok=True)
    for layer in args.layers:
        os.makedirs(os.path.join(output_root, f"layer{layer}"), exist_ok=True)

    # Inference loop
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            true_idx = args.batch_start_from + batch_idx
            if args.batch_end_at is not None and true_idx >= args.batch_end_at:
                break

            batch_hidden_states.clear()
            
            images = batch[0]
            images = images.to(device)
            if args.model == 'clip_vit':
                _ = model.vision_model(pixel_values=images)
            else:
                _ = model(images)

            for layer in args.layers:
                act = batch_hidden_states[layer]  # [B, T, D]
                B, T, D = act.shape
                flat = act.reshape(B*T, D).half().cpu().numpy()
                save_path = os.path.join(
                    output_root, f"layer{layer}",
                    f"batch_{true_idx:04d}.npy")
                np.save(save_path, flat)

    # Remove hooks
    for h in handles.values(): h.remove()
    print("Done.")


if __name__ == '__main__':
    main()
