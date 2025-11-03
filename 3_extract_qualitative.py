"""
    3_extract_qualitative.py

    This script visualizes the top-k most activating image patches and their class associations
    from a cascaded model composed of ViT-based backbones and sparse autoencoders (TopKSAE).

    Main functionalities:
        - Extracts top-k images that most activate each feature at each transformer layer
        - Computes decoder-to-logit projection ("logit lens") to associate features with class predictions
        - Overlays activation heatmaps on top images and highlights maximally activated patch
        - Optionally saves cropped patch images and red-box overlays per feature (if --save_frags enabled)
        - Produces composite visualizations of (1) activation overlay, (2) original image, (3) cropped patch, and (4) bar chart of top-10 classes promoted by the feature

    Arguments:
        --model: Model type ['vit', 'dinov2', 'clip_vit']
        --target: Feature type to visualize ['sae' or 'residual']
        --layernums: List of transformer layers to process
        --ablate_R: Optional ablation mode for dictionary size ['never', 'low', 'high']
        --save_frags: Whether to save patch-fragments and red-box overlays (default: False)
        --data_root: Path to ImageNet validation set  (e.g., /USER/ILSVRC/Data/CLS-LOC/val)
        --output_path: Directory to save composite visualizations  (e.g., /USER/qualitative_results)
        --sae_ckpt_path: Root directory for pretrained SAE checkpoints  (e.g., /USER/checkpoints/topk_sae/imagenet)
        --cache_path: Root directory for intermediate feature cache (acts/lens)  (e.g., /USER/cache/SAEs/imagenet)
        --acts_file_path: Subdirectory for storing top-k activation maps  (e.g., /USER/cache/patch_activations/)
        --lens_file_path: Subdirectory for storing logit lens projections  (e.g., /USER/cache/logit_lens/)
        --dino_head_ckpt_path: Path to DINOv2 classifier head weights  (e.g., /USER/vision_checkpoints/dinov2_head/dinov2_vitb14_reg4_linear_head.pth)
            > Please refer to: https://github.com/facebookresearch/dinov2
        --clip_text_embed_path: Path to CLIP text features tensor file. Size: [num_classes, dim_text]  (e.g., /USER/cache/text_features.pt)
        --epoch: Epoch of SAE checkpoint (used in filename only)
        --device: Device used for inference, e.g., "cuda:0"
        --batch_size: Batch size for image processing
        --seed: Random seed for reproducibility

    Outputs:
        - Visualizations saved in `output_path/[model]/[target]/L{layer}_F{feature}.png`
        - (Optional) Fragment images in `output_path_frags/[model]/L{layer}/F{feature}/`
        - Intermediate cache files in `acts_file_path` and `lens_file_path`

    Example usage:
        python 3_extract_qualitative.py \
            --model vit \
            --target sae \
            --layernums 10 9 8 \
            --data_root /path/to/imagenet/val \
            --sae_ckpt_path /path/to/checkpoints \
            --acts_file_path /path/to/cache/activations \
            --lens_file_path /path/to/cache/lens \
            --output_path ./visual_results \
            --save_frags
"""


import argparse
import math
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers import CLIPProcessor

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import colormaps
from tqdm import tqdm

from utils import set_seed, ImageNetValDataset, collate_fn_clip
from utils.functions.functions import clip_unnormalize, unnormalize
from utils.functions.load_models import create_cascaded_model


def parse_args():
    parser = argparse.ArgumentParser(description="Visualization for Top-k Activation and Logit Lens")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use, e.g., cuda:0 or cpu")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for dataloader")
    parser.add_argument("--epoch", type=int, default=50, help="Epoch number (if needed for checkpoint naming)")
    parser.add_argument("--model", type=str, required=True, choices=["vit", "dinov2", "clip_vit"], help="Model type.")
    parser.add_argument("--target", type=str, required=True, choices=["sae", "residual"], help="Feature target type for extraction.")
    parser.add_argument("--layernums", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], help="List of layer numbers for feature extraction.")
    parser.add_argument("--ablate_R", type=str, default="never", choices=["never", "low", "high"])
    parser.add_argument("--save_frags", action="store_true", help="Save image fragments for each feature")
    
    parser.add_argument("--data_root", type=str, default="", help="Path to ImageNet val dataset.")  # E.g., /USER/ILSVRC/Data/CLS-LOC/val/
    parser.add_argument("--output_path", type=str, default="", help="Output directory to save visualizations.")  # E.g., PROJECT_ROOT/qualitative_results/
    parser.add_argument("--sae_ckpt_path", type=str, default="", help="Path to TopKSAE checkpoints. Use a path that has created with 2_train_TopKSAE.py.")  # E.g., PROJECT_ROOT/checkpoints/imagenet/
    parser.add_argument("--cache_path", type=str, default="", help="Cache directory used to load sparse autoencoders faster.")  # E.g., PROJECT_ROOT/cache
    parser.add_argument("--acts_file_path", type=str, default="", help="Path to cache patch activations.")  # E.g., PROJECT_ROOT/cache/patch_activations/
    parser.add_argument("--lens_file_path", type=str, default="", help="Path to cache logit lens projections.")  # E.g., PROJECT_ROOT/cache/logit_lens/
    parser.add_argument("--dino_head_ckpt_path", type=str, default="", help="Dinov2 classification head checkpoint. Refer to: https://github.com/facebookresearch/dinov2")  # E.g., PROJECT_ROOT/vision_heads/dinov2_vitb14_reg4_linear_head.pth
    parser.add_argument("--clip_text_embed_path", type=str, default="", help="CLIP text features tensor file. Its shape must be (Number of classes, Embedding dimension) Refer to: https://colab.research.google.com/github/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb#scrollTo=4W8ARJVqBJXs")

    return parser.parse_args()


def generate_topk_activation_file(dataset, cascaded_model, k, layernum, args, save_path=None):
    """
    Extracts top-k activating images and corresponding activation maps for each feature 
    in a given transformer layer.

    This function first runs all images through the cascaded model to compute the maximum 
    activation per feature across the dataset. Then, for each feature, it selects the top-k 
    images that produce the highest activation. It re-infers each of these selected images 
    to extract per-patch activation maps.

    Args:
        dataset (Dataset): A dataset of input images.
        cascaded_model: The cascaded model including ViT and TopKSAE modules.
        k (int): Number of top activating images to retrieve per feature.
        layernum (int): Transformer layer index to process.
        args (Namespace): Argument namespace containing model type, device, etc.
        save_path (str, optional): If provided, saves the resulting dictionary as a pickle file.

    Returns:
        dict: A dictionary mapping each feature index to a list of length-k containing 
              {image index: patch-wise activation values} pairs.
              Format: result[feature_idx] = List[Dict[int, Tensor]]
    """
    max_act_img_indices = []

    # 1. Collect maximum activation value per feature across the dataset
    for img_idx in tqdm(range(len(dataset)), desc=f"Extracting max activations (Layer {layernum})"):
        image, _ = dataset[img_idx]
        if args.model == "clip_vit":
            image = args.processor(images=[image], return_tensors="pt")["pixel_values"][0]
        image = image.unsqueeze(0).to(args.device)
        _ = cascaded_model(image)

        if args.target == "sae":
            z = cascaded_model.sae_models[layernum].intermediates["z"].detach().cpu().squeeze(0)  # [T, F]
        else:  # args.target == "residual"
            z = cascaded_model.vit_blocks[layernum].intermediates["residual_out"].detach().cpu().squeeze(0)  # [T, F]
        max_act = z.max(dim=0)[0]  # [F]
        max_act_img_indices.append(max_act)

    # 2. Compute top-k image indices for each feature (shape: [k, F])
    max_act_img_indices = torch.stack(max_act_img_indices, dim=0)  # [N, F]
    max_act_img_indices = torch.topk(max_act_img_indices, k, dim=0)[1]  # [k, F]
    
    # 3. Re-infer selected images to extract activation maps
    result = {}
    num_features = max_act_img_indices.shape[1]
    result = {f: [] for f in range(num_features)}

    for feature_idx in tqdm(range(num_features), desc=f"Re-inference to get [N, T, F] (Layer {layernum})"):
        indices = max_act_img_indices[:, feature_idx]  # [k]
        for img_idx in indices:
            image, _ = dataset[img_idx.item()]
            if args.model == "clip_vit":
                image = args.processor(images=[image], return_tensors="pt")["pixel_values"][0]
            image = image.unsqueeze(0).to(args.device)
            _ = cascaded_model(image)

            if args.target == "sae":
                z = cascaded_model.sae_models[layernum].intermediates["z"].detach().cpu().squeeze(0)
            else:
                z = cascaded_model.vit_blocks[layernum].intermediates["residual_out"].detach().cpu().squeeze(0)
            result[feature_idx].append({img_idx.item(): z[:, feature_idx]})

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved top-k activations for layer {layernum} to: {save_path}")

    return result


def generate_logit_lens_file(cascaded_model, dataloader, args, save_path=None):
    """
    Computes "logit lens" projection matrices for each transformer layer.

    This function projects each decoder feature vector from the cascaded model into the
    classifier output space (e.g., class logits). It supports ViT, DINOv2, and CLIP-ViT backbones.
    For DINOv2, it first estimates the weighting ratio (alpha) between class token and patch tokens
    using average activation magnitudes. This is used to blend the unembedding weights accordingly.

    Args:
        cascaded_model: The cascaded model consisting of a ViT-based backbone and TopKSAE layers.
        dataloader (DataLoader): Dataloader used to estimate class-to-patch ratios (only used for DINOv2).
        args (Namespace): Arguments including model type, target type, and checkpoint paths.
        save_path (str, optional): If provided, saves the resulting dictionary as a pickle file.

    Returns:
        dict: A dictionary mapping layer numbers to logit projection matrices.
              Each matrix maps decoder features to class logits: [F, C].
    """
    # For mean-pooled models (e.g., DINOv2), compute alpha (class-to-patch activation ratio)
    if args.model == "dinov2":
        cls_to_patch_ratio = {layernum: [] for layernum in range(0, 11)}
        with torch.no_grad():
            for img_idx, (image, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Creating logit lens file for {args.model}"):
                image = image.to(args.device)
                label = label.to(args.device)
                
                # Forward pass
                _ = cascaded_model(image)
                
                for layernum in range(0, 11):
                    if args.target == "sae":
                        acts_topk = cascaded_model.sae_models[layernum].intermediates["z"]  # [B*T, F]
                    elif args.target == "residual":
                        acts_topk = cascaded_model.vit_blocks[layernum].intermediates["residual_out"]
                        
                    acts_topk = acts_topk.reshape(image.shape[0], -1, acts_topk.shape[-1]).squeeze()  # [T, F], assuming B=1

                    cls_token_acts = acts_topk[0, :].mean()
                    patch_token_acts = acts_topk[5:, :].mean()
                    ratio = (cls_token_acts / patch_token_acts).item()
                    cls_to_patch_ratio[layernum].append(ratio)
                
            # Compute average cls-to-patch ratio per layer
            cls_to_patch_ratio = {
                layernum: sum(ratios) / len(ratios)
                for layernum, ratios in cls_to_patch_ratio.items()
            }

    # Project decoder vectors to class logits (i.e., unembedding)
    if args.model == "vit":
        W_unembed = cascaded_model.vit.head.weight  # [C, D]
        D = W_unembed.shape[1]
        if args.target == "residual":
            sae_decoder_vectors = {layernum: torch.eye(D, D, device=W_unembed.device) for layernum in range(0, 11)}  # [D, D]
        else:  # args.target == "sae"
            sae_decoder_vectors = {layernum: F.normalize(cascaded_model.sae_models[layernum].W_dec, p=2, dim=1) for layernum in range(0, 11)}  # [F, D]
        sae_decoder_contributions = {layernum: sae_decoder_vectors[layernum] @ W_unembed.T for layernum in range(0, 11)}  # [F, C]
        
    elif args.model == "dinov2":
        W_unembed = cascaded_model.dinov2.head.weight  # [C, 2*D]
        assert W_unembed.shape[1] % 2 == 0
        D = W_unembed.shape[1] // 2
        W_unembed_cls = W_unembed[:, :D]
        W_unembed_patch = W_unembed[:, D:]
        if args.target == "residual":
            sae_decoder_vectors = {layernum: torch.eye(D, D, device=W_unembed.device) for layernum in range(0, 11)}  # [D, D]
        else:  # args.target == "sae"
            sae_decoder_vectors = {layernum: F.normalize(cascaded_model.sae_models[layernum].W_dec, p=2, dim=1) for layernum in range(0, 11)}  # [F, D]
        
        sae_decoder_contributions = {}
        for layernum in range(0, 11):
            alpha = cls_to_patch_ratio[layernum]
            sae_decoder_contributions[layernum] = alpha * sae_decoder_vectors[layernum] @ W_unembed_cls.T + (1 - alpha) * sae_decoder_vectors[layernum] @ W_unembed_patch.T  # [F, C]

    elif args.model == "clip_vit":
        W_unembed = cascaded_model.head.weight  # [Dt, D]
        D = W_unembed.shape[1]
        if args.target == "residual":
            sae_decoder_vectors = {layernum: torch.eye(D, D, device=W_unembed.device) for layernum in range(0, 11)}  # [D, D]
        else:  # args.target == "sae"
            sae_decoder_vectors = {layernum: F.normalize(cascaded_model.sae_models[layernum].W_dec, p=2, dim=1) for layernum in range(0, 11)}  # [F, D]
        label_embedding = torch.load(args.clip_text_embed_path, map_location=args.device)  # [C, Dt]
        sae_decoder_contributions = {layernum: (sae_decoder_vectors[layernum] @ W_unembed.T) @ label_embedding.T for layernum in range(0, 11)}  # [F, C]
    
    else:
        raise ValueError(f"args.model == {args.model} is not supported.")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(sae_decoder_contributions, f)  # [F, C]
        print(f"Logit lens results saved to: {save_path}")
        
    return sae_decoder_contributions


def visualize_act_img_map(act_img_map, lens_file, layernum, dataset, feature_idx, args,
                          alpha_minmax=(0.0, 0.5), save_path=None, buffer="./buffer"):
    """
    Visualizes the top-k activating image patches for a given feature and transformer layer.

    This function overlays heatmaps on the original image to indicate which patches activate
    the selected feature the most. It also visualizes the most activated patch and shows
    a bar chart of the top-10 classes most strongly associated with this feature (logit lens projection).

    If args.save_frags is enabled, the function will save individual fragment images such as:
    - overlay image with heatmap
    - original image
    - red-box highlighted patch
    - cropped patch

    Args:
        act_img_map (list): A list of dicts containing image indices and per-patch activations.
        lens_file (dict): Dictionary mapping each layer and feature to class contributions.
        layernum (int): Transformer layer number.
        dataset (Dataset): Dataset to retrieve original images.
        feature_idx (int): Feature index to visualize.
        args (Namespace): Parsed command-line arguments.
        alpha_minmax (tuple): Min and max alpha values for overlay transparency.
        save_path (str, optional): If provided, saves final visualization to this path.
        buffer (str): Temporary buffer path to store intermediate images.

    Returns:
        None
    """
    # Normalize input structure to a unified list of dicts
    processed_map = []
    for entry in act_img_map:
        if isinstance(entry, dict) and 'index' in entry and 'act_per_patch' in entry:
            processed_map.append(entry)
        else:
            for img_idx, act in entry.items():
                processed_map.append({'index': img_idx, 'act_per_patch': act})
    act_img_map = processed_map

    num_images = len(act_img_map)
    assert num_images % 2 == 0, "Number of images must be even."
    cmap = colormaps.get_cmap('viridis')
    
    if args.save_frags:
        frag_dir = os.path.join(save_path, f"{args.model}/L{layernum}/F{feature_idx}")
        os.makedirs(frag_dir, exist_ok=True)
    else:
        num_image_per_row = num_images // 2
        fig_images, axes_images = plt.subplots(4, num_image_per_row, figsize=(9.2, 9.2))
        fig_patch, axes_patch = plt.subplots(2, 5, figsize=(5, 3.2))

    for i, entry in enumerate(act_img_map):
        img_idx = entry['index']
        img, label = dataset[img_idx]
        
        if args.model == "clip_vit":
            img_tensor = args.processor(images=img, return_tensors="pt")["pixel_values"][0]
            img_np = clip_unnormalize(img_tensor).cpu().permute(1, 2, 0).numpy()
        else:
            if isinstance(img, torch.Tensor):
                img_np = unnormalize(img).cpu().permute(1, 2, 0).numpy()
            else:
                img_np = np.array(img) / 255.0

        act_per_patch = entry['act_per_patch']
        if args.model == "dinov2":
            act_per_patch = act_per_patch[5:]  # Skip class tokens
        elif args.model in ["vit", "clip_vit"]:
            act_per_patch = act_per_patch[1:]  # Skip class token

        if isinstance(act_per_patch, torch.Tensor):
            act_per_patch = act_per_patch.cpu().numpy()
            
        grid_size = int(math.sqrt(act_per_patch.shape[0]))
        heatmap = act_per_patch.reshape(grid_size, grid_size)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        min_alpha, max_alpha = alpha_minmax
        range_val = heatmap_resized.max() - heatmap_resized.min()
        alpha = (0 if range_val == 0 else min_alpha + (heatmap_resized - heatmap_resized.min()) / (range_val + 1e-6) * (max_alpha - min_alpha))
        heatmap_rgba = cmap(heatmap_resized)

        overlay = np.zeros_like(img_np)
        for c in range(3):
            overlay[:, :, c] = img_np[:, :, c] * (1 - alpha) + heatmap_rgba[:, :, c] * alpha

        max_patch_idx = np.argmax(act_per_patch)
        patch_row = max_patch_idx // grid_size
        patch_col = max_patch_idx % grid_size
        h, w, _ = img_np.shape
        ph, pw = h // grid_size, w // grid_size
        crop = img_np[patch_row*ph:(patch_row+1)*ph, patch_col*pw:(patch_col+1)*pw, :]
        
        if args.model == "clip_vit":
            overlay_uint8 = (overlay * 255).astype(np.uint8)
            overlay = cv2.resize(overlay_uint8, (224, 224), interpolation=cv2.INTER_LINEAR) / 255.0
            crop_uint8 = (crop * 255).astype(np.uint8)
            crop = cv2.resize(crop_uint8, (224, 224), interpolation=cv2.INTER_NEAREST) / 255.0
        
        if args.save_frags:
            overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))
            crop_pil = Image.fromarray((crop * 255).astype(np.uint8))
            origin_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            
            x1, y1 = patch_col * pw, patch_row * ph
            x2, y2 = x1 + pw - 1, y1 + ph - 1

            boxed = origin_pil.copy()
            draw = ImageDraw.Draw(boxed)
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
            
            overlay_pil.save(os.path.join(frag_dir, f"image{i+1}.png"))
            crop_pil.save   (os.path.join(frag_dir, f"patch{i+1}.png"))
            origin_pil.save (os.path.join(frag_dir, f"origin{i+1}.png"))
            boxed.save      (os.path.join(frag_dir, f"redbox{i+1}.png"))
            continue
        
        else:
            row = (i // num_image_per_row) * 2
            col = i % num_image_per_row
            axes_images[row, col].imshow(overlay)
            axes_images[row, col].set_title(f"{col+1+(row//2)*num_image_per_row}", fontsize=10)
            axes_images[row, col].axis('off')

            axes_images[row+1, col].imshow(img_np)
            axes_images[row+1, col].set_title(ImageNetValDataset.to_class_names(label), fontsize=10)
            axes_images[row+1, col].axis('off')

            pr, pc = i // 5, i % 5
            axes_patch[pr, pc].imshow(crop)
            axes_patch[pr, pc].set_title(f"{col+1+(row//2)*num_image_per_row}", fontsize=16)
            axes_patch[pr, pc].axis('off')

    if args.save_frags:
        return
    
    fig_images.suptitle("Maximally Activated Images", fontsize=20)
    fig_images.tight_layout()
    fig_patch.suptitle("Maximally Activated Patches", fontsize=20)
    fig_patch.tight_layout()

    # Plot bar chart for top-k class contributions
    contributions = lens_file[layernum][feature_idx, :]  # [C]
    if isinstance(contributions, torch.Tensor):
        contributions = contributions.detach().cpu().numpy()
    top_k = 10
    top_indices = np.argsort(contributions)[-top_k:][::-1]
    top_values = contributions[top_indices]
    class_names = [ImageNetValDataset.to_class_names(int(idx)) for idx in top_indices]

    fig_bar, ax_bar = plt.subplots(figsize=(4.5, 5.7))
    y_pos = np.arange(len(class_names))
    bars = ax_bar.barh(y_pos, top_values, color='royalblue', edgecolor='black', linewidth=1)
    for bar, class_name in zip(bars, class_names):
        ax_bar.text(0.001, bar.get_y() + bar.get_height() / 2, class_name,
                    color='black', ha='left', va='center', fontsize=12)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(['' for _ in class_names])
    ax_bar.invert_yaxis()
    ax_bar.set_title(f"Class most promoted", fontsize=12)
    for spine in ['top', 'right']:
        ax_bar.spines[spine].set_visible(False)
    fig_bar.tight_layout()

    save_dir = os.path.join(buffer, args.model, args.target, str(layernum))
    os.makedirs(save_dir, exist_ok=True)
    fig_images.savefig(os.path.join(save_dir, "fig_images.png"), dpi=300)
    fig_patch.savefig(os.path.join(save_dir, "fig_patch.png"), dpi=300)
    fig_bar.savefig(os.path.join(save_dir, "fig_barchart.png"), dpi=300)
    plt.close(fig_bar); plt.close(fig_images); plt.close(fig_patch)

    for ax_row in axes_images:
        for ax in ax_row.flatten():
            ax.set_xticks([]); ax.set_yticks([])
    for ax in axes_patch.flatten():
        ax.set_xticks([]); ax.set_yticks([])

    # Combine final visualizations into a single image
    bar_chart_path = os.path.join(save_dir, "fig_barchart.png")
    patch_fig_path = os.path.join(save_dir, "fig_patch.png")
    image_fig_path = os.path.join(save_dir, "fig_images.png")

    bar_chart = Image.open(bar_chart_path).convert("RGB")
    patch_figures = Image.open(patch_fig_path).convert("RGB")
    image_figures = Image.open(image_fig_path).convert("RGB")

    canvas_width = patch_figures.width + image_figures.width + 70
    canvas_height = max(bar_chart.height, patch_figures.height + 100, image_figures.height)
    new_image = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    new_image.paste(bar_chart, (70, 0))
    new_image.paste(patch_figures, (0, bar_chart.height + 100))
    new_image.paste(image_figures, (patch_figures.width + 70, 0))

    if save_path is not None:
        final_dir = os.path.join(save_path, args.model, args.target)
        os.makedirs(final_dir, exist_ok=True)
        final_path = os.path.join(final_dir, f"L{layernum}_F{feature_idx}.png")
        new_image.save(final_path)
        # print(f"Final visualization saved to: {final_path}")

        
def main():
    args = parse_args()
    set_seed(args.seed)
    if args.ablate_R != "never": assert args.model == "vit"

    cascaded_model = create_cascaded_model(args)

    # Load dataset
    if args.model == "clip_vit":
        args.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        vit_transform = None
    else:
        vit_transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    if "val" in args.data_root:  # Loading ImageNet val set
        imagenet_val_path = Path(args.data_root)
        base_dir = imagenet_val_path.parents[2]  # 0-based index
        annotation_path = os.path.join(base_dir, "Annotations/CLS-LOC/val")
        dataset = ImageNetValDataset(
            imagenet_val_path,
            annotation_path,
            transform=vit_transform
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=(collate_fn_clip if args.model == "clip_vit" else None)
        )

    # Extract acts/lens | Visualize
    os.makedirs(args.acts_file_path, exist_ok=True)
    os.makedirs(args.lens_file_path, exist_ok=True)

    for layernum in args.layernums:
        # Construct full path for checking existence only
        if args.ablate_R == "never":
            acts_full_path = Path(f"{args.acts_file_path}/{args.model}/{args.target}/layer{layernum}.pkl")
            lens_full_path = Path(f"{args.lens_file_path}/{args.model}/{args.target}.pkl")
        else:
            acts_full_path = Path(f"{args.acts_file_path}/{args.model}/{args.target}/{args.ablate_R}R/layer{layernum}.pkl")
            lens_full_path = Path(f"{args.lens_file_path}/{args.model}/{args.target}_{args.ablate_R}R.pkl")

        # Activation file loading
        if not acts_full_path.exists():
            print(f"No activation file found for {args.model} (Layer {layernum}). Generating top-k activations...")
            acts_file = generate_topk_activation_file(
                dataset=dataset,
                cascaded_model=cascaded_model,
                k=10,
                layernum=layernum,
                args=args,
                save_path=acts_full_path,
            )
        else:
            print(f"Activation file found for {args.model}. Loading from {acts_full_path}")
            with open(acts_full_path, "rb") as f:
                acts_file = pickle.load(f)

        # Logit lens file loading
        if not lens_full_path.exists():
            print(f"No logit lens file found for {args.model}. Generating logit lens result...")
            lens_file = generate_logit_lens_file(
                cascaded_model=cascaded_model,
                dataloader=dataloader,
                args=args,
                save_path=lens_full_path
            )
        else:
            print(f"Logit lens file found for {args.model}. Loading from {lens_full_path}")
            with open(lens_full_path, "rb") as f:
                lens_file = pickle.load(f)

        if args.save_frags:
            output_path = f"{args.output_path}_frags"
        else:
            output_path = args.output_path
        if not args.ablate_R == "never":
            output_path = f"{output_path}/{args.ablate_R}R"
            
        for feature_idx in tqdm(acts_file.keys(), desc=f"Visualizing features for Layer {layernum}"):
            act_img_map = acts_file[feature_idx]
            visualize_act_img_map(
                act_img_map=act_img_map,
                lens_file=lens_file,
                layernum=layernum,
                dataset=dataset,
                feature_idx=feature_idx,
                args=args,
                alpha_minmax=(0.0, 0.5),
                save_path=output_path,
                buffer=(
                    "./buffer"
                    if args.ablate_R == "never"
                    else f"./buffer/{args.ablate_R}R")
            )


if __name__ == "__main__":
    main()
