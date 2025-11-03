"""
    5_visualize_circuit.py

    This script visualizes normalized relevance (node and edge importance) for cascaded SAE/ViT layers
    on a single image. It generates a graph (using graphviz) where each node shows its relevance score
    and an attached small PNG containing the corresponding input patch and an activation heatmap.

    Supported models:
        - ViT
        - DINOv2
        - CLIP-ViT
    
    Main features:
        - Compute relevance from a model (vit, dinov2, clip_vit) and extract node/edge importance
        - Select top nodes per layer and simplify the graph (edge-based selection)
        - Create per-node 2x4 composite images (patches + heatmaps)
        - Visualize nodes with color encoding (sign + magnitude), and edges with width and color proportional to relevance
        - Save the rendered graph PNG and temporary node images to specified directories

    Arguments:
        --model: Model type (vit, dinov2, clip_vit)
        --data_index: Index of the data sample to visualize
        --method: Relevance computation method ("grad" or "ig")
        --leaf_error: Whether to use predicted error activations for masking
        --libragrad: Use Libragrad mode (if set to False, uses built-in gradients)
        --output_path: Output root directory to save results as pickle files  (e.g., /USER/visualizations/)
        --cache_path: Path for loading cached data (SAE checkpoints, indices, etc.)   (e.g., /USER/cache/SAEs/imagenet/)
        --dino_head_ckpt_path: Path to the pretrained linear classification head for DINOv2
        --clip_text_embed_path: Path to cached CLIP text embeddings (.pt) for logit projection
        --sae_ckpt_path: Path to pretrained Top-K SAE model checkpoints
        --mean_acts_path: Directory containing median activation tensors (used for masking)  (e.g., )
            > Refer to get_avg_activation.py for details on how to generate these tensors
        --data_root: Path to ImageNet validation dataset root directory  (e.g., /USER/Data/CLS-LOC/val)
        --temporal_save_path: Directory to save temporary node images for visualization

    Outputs:
        - Rendered graph PNG at: output_path/imagenet_<class>_<data_index>_<model>.png
        - Temporary node images saved under --temporal_save_path (removed at end unless preserved)

    Example usage:
        python 5_visualize_circuit.py \
            --model vit \
            --data_index 0 \
            --output_path ./visualizations \
            --temporal_save_path ./temp_nodes

    Notes:
        - The script currently processes a single image specified by --data_index. To process multiple images,
        call the script repeatedly or implement batching externally.
        - graphviz installation is required.
"""

import graphviz #! TODO: Write that this is additionally required
from PIL import Image
import cv2
from matplotlib import colormaps
cmap = colormaps.get_cmap('viridis')

import argparse
import os
import pickle
import random
import sys
sys.path.append("..")

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import CLIPProcessor

from utils import set_seed, collate_fn_clip, ImageNetValDataset, get_imagenet_transform, create_cascaded_model, unnormalize
from utils import get_simplified_graph_by_node, get_simplified_graph_by_edge, get_relevance, get_normalized_relevance


def print_args(args):
    print("=" * 50)
    print("Parsed Arguments:")
    print("=" * 50)
    print(f"Model Type: {args.model}")
    print(f"Device: {args.device}")
    print(f"Data Index: {args.data_index}")
    print(f"Relevance Computation Method: {args.method}")
    print(f"Use Libragrad: {args.libragrad}")
    print(f"Use Negative Relevance: {args.graph_negative}")
    print(f"Use Leaf Error: {args.leaf_error}")
    print("=" * 50)
    

def parse_args():
    parser = argparse.ArgumentParser(description="Faithfulness Evaluation Script")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for DataLoader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model", type=str, required=True, choices=["vit", "dinov2", "clip_vit"], help="Model type")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="Device to use for computation")

    parser.add_argument("--data_index", type=int, default=0, help="Index of the data sample to visualize")

    parser.add_argument("--libragrad", action="store_false", help="Use libragrad")
    parser.add_argument("--gamma_rule", type=str, default=None, help="Gamma rule for evaluation")
    parser.add_argument("--method", type=str, default="grad", choices=["grad", "ig"], help="Method for relevance computation")
    parser.add_argument("--graph_negative", "-n", action="store_true", help="Use negative relevance in graph criterion")
    parser.add_argument("--leaf_error", "-l", action="store_false", help="Use leaf error for relevance computation")

    parser.add_argument("--x_width", type=float, default=5.0, help="Figure width for plotting (x dimension)")
    parser.add_argument("--y_width", type=float, default=3.0, help="Figure height for plotting (y dimension)")
    parser.add_argument("--node_vmax", type=float, default=0.2, help="Max colormap value for node heatmap")
    parser.add_argument("--node_vmin", type=float, default=0.0, help="Min colormap value for node heatmap")
    parser.add_argument("--edge_vmax", type=float, default=0.02, help="Max colormap value for edge heatmap")
    parser.add_argument("--edge_vmin", type=float, default=0.0, help="Min colormap value for edge heatmap")

    parser.add_argument("--output_path", type=str, default="", help="Path to save output data")
    parser.add_argument("--cache_path", type=str, default="", help="Path to cache data")
    parser.add_argument("--sae_ckpt_path", type=str, default="", help="Path to SAE checkpoint")
    parser.add_argument("--mean_acts_path", type=str, default="", help="Path to mean activation data")
    parser.add_argument("--data_root", type=str, default="", help="Path to ImageNet validation data")
    parser.add_argument("--dino_head_ckpt_path", type=str, default="", help="Path to DINO head checkpoint")
    parser.add_argument("--clip_text_embed_path", type=str, default="", help="Path to CLIP text embeddings")
    parser.add_argument("--temporal_save_path", type=str, default="", help="Path to save temporal/interactive outputs")

    args = parser.parse_args()
    print_args(args)

    return args


def main():
    args = parse_args()

    set_seed(args.seed)

    # Load the model
    cascaded_model = create_cascaded_model(args)

    # Load the dataset
    dataset = ImageNetValDataset(
        image_dir=Path(args.data_root),
        annotation_dir=os.path.join(Path(args.data_root).parents[2], "Annotations/CLS-LOC/val"),
        transform=get_imagenet_transform() if args.model != "clip_vit" else None,
    )
    if args.model == "clip_vit":
        args.collate_fn = partial(
            collate_fn_clip,
            processor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        )

    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=(args.collate_fn if args.model == "clip_vit" else None)
    )

    # Load mean & base feature acts
    with open(os.path.join(args.mean_acts_path, f"median_feature_acts_{args.model}.pkl"), "rb") as f:
        args.mean_feature_acts = pickle.load(f)
    with open(os.path.join(args.mean_acts_path, f"median_error_acts_{args.model}.pkl"), "rb") as f:
        args.mean_error_acts = pickle.load(f)
    for i in range(11):
        args.mean_feature_acts[i] = args.mean_feature_acts[i].to(args.device) 
        args.mean_error_acts[i] = args.mean_error_acts[i].to(args.device)
    args.mean_neuron_acts = None
        
    image, label = dataloader.dataset[args.data_index]
    image = image.to(args.device)
    label = torch.tensor([label]).to(args.device)
    logits = cascaded_model(image.unsqueeze(0))
    true_error_acts = cascaded_model.get_true_error_acts()

    logit_node = dataset.to_class_names(label.item())

    relevances = get_relevance(
        model=cascaded_model,
        image=image,
        label=label,
        method=args.method,
        target="feature",
        leaf_error=args.leaf_error,
        mean_feature_acts=args.mean_feature_acts,
        mean_error_acts=args.mean_error_acts,
        mean_neuron_acts=args.mean_neuron_acts,
        get_logit=False,
        adjust_logit=True,
        compute_edge=True,
    )

    normalized_relevances = get_normalized_relevance(relevances=relevances, negative_relevance=args.graph_negative)

    node_indices, node_relevances, edge_relevances = get_simplified_graph_by_edge(
        normalized_relevances=normalized_relevances,
        topn=args.graph_topn,
        threshold=None,
        max_node_num=1,
        return_values=True,
    )

    if not os.path.exists(args.temporal_save_path):
        os.makedirs(args.temporal_save_path)

    layer_num = len(node_indices) + 1
    error_nums = {i+1: cascaded_model.sae_models[i].dict_size for i in range(0, layer_num-1)}

    # Visualization
    G = graphviz.Digraph(comment="The Round Table")
    G.graph_attr.update(rankdir="LR")
    G.node_attr.update(shape="box", style="rounded,filled")
    G.engine = 'neato'

    def value_to_color(value, vmax, vmin):
        sign = 1 if value > 0 else -1
        value = (np.abs(value) - vmin) / (vmax - vmin)
        value = np.clip(value, 0, 1)
        cmap = plt.colormaps.get_cmap('RdBu')
        rgba = cmap(0.5 + sign * value / 2) 
        r, g, b, a = [int(255 * x) for x in rgba]
        return f"#{r:02x}{g:02x}{b:02x}"

    def create_2x4_node_image(image_dir, test_patch, test_image, base_length=60, sep=10):
        filenames = [
            "patch1.png", "patch2.png", "patch3.png",
            "image1.png", "image2.png", "image3.png",
            # "origin1.png", "origin2.png", "origin3.png",
        ]
        
        images = []

        tpatch = test_patch.resize((base_length, base_length), resample=Image.LANCZOS)
        timage = test_image.resize((base_length, base_length), resample=Image.LANCZOS)

        for fname in filenames:
            path = os.path.join(image_dir, fname)
            if os.path.exists(path):
                img = Image.open(path).resize((base_length, base_length), resample=Image.LANCZOS)
            else:
                img = Image.new("RGB", (base_length, base_length), color="white")
            images.append(img)

        # concat
        row1 = Image.new("RGB", (base_length * 4 + sep, base_length), color="white")
        row2 = Image.new("RGB", (base_length * 4 + sep, base_length), color="white")

        row1.paste(tpatch, (0, 0))
        row2.paste(timage, (0, 0))

        for i in range(3):
            row1.paste(images[i], ((i + 1) * base_length + sep, 0))
            row2.paste(images[i + 3], ((i + 1) * base_length + sep, 0))

        final_img = Image.new("RGB", (base_length * 4 + sep, base_length * 2))
        final_img.paste(row1, (0, 0))
        final_img.paste(row2, (0, base_length))
        return final_img

    # Node
    for i in range(1, layer_num):
        # node_relevance = normalized_relevances.block[i].node
        node_relevance = node_relevances[i]

        for p, nidx in enumerate(node_indices[i]):
            
            if nidx != error_nums[i]:

                test_origin = unnormalize(image).cpu().numpy().transpose(1, 2, 0)
                test_act = cascaded_model.sae_models[i-1].intermediates['z'][5:, nidx]
                width = int(test_act.shape[0] ** 0.5)
                test_act = test_act.view(width, width)
                test_act = test_act.cpu().detach().numpy()
                row, col = np.unravel_index(np.argmax(test_act), test_act.shape)
                patch_width = test_origin.shape[0] // width
                test_patch = test_origin[row * patch_width: (row + 1) * patch_width, col * patch_width: (col + 1) * patch_width]
                test_patch = Image.fromarray((test_patch * 255).astype(np.uint8))

                min_alpha, max_alpha = 0.0, 0.5
                test_act = (test_act - test_act.min()) / (test_act.max() - test_act.min() + 1e-6)
                test_act = cv2.resize(test_act, (test_origin.shape[1], test_origin.shape[0]), interpolation=cv2.INTER_NEAREST)
                range_val = test_act.max() - test_act.min()
                alpha = (0 if range_val == 0 else min_alpha + (test_act - test_act.min()) / (range_val + 1e-6) * (max_alpha - min_alpha))
                test_act = cmap(test_act)
                test_image = np.zeros_like(test_origin)
                for c in range(3):
                    test_image[:, :, c] = test_act[:, :, c] * alpha + test_origin[:, :, c] * (1 - alpha)
                test_image = Image.fromarray((test_image * 255).astype(np.uint8))

                image_dir = f"/nfs/home/junhyeok/neurips2025/qualitative_results_frags/{args.model}/L{i-1}/F{nidx}"
                node_image = create_2x4_node_image(
                    image_dir=image_dir,
                    test_patch=test_patch,
                    test_image=test_image,
                    base_length=60,
                    sep=10,
                )
                node_image.save(os.path.join(args.temporal_save_path, f"node_image_{i-1}_{nidx}.png"))

                node_name = f"L{i-1}#{nidx}"
                G.node(
                    node_name,
                    label=f'''<
                        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                            <TR><TD><FONT POINT-SIZE="14">{node_name}<BR/>{node_relevance[p].item() * 100:2.1f}%</FONT></TD></TR>
                            <TR><TD><IMG SRC="{os.path.join(args.temporal_save_path, f"node_image_{i-1}_{nidx}.png")}" SCALE="FALSE"/></TD></TR>
                        </TABLE>
                    >''',
                    # label=f"{node_name}\n{node_relevance[p].item() * 100:2.1f}%",
                    pos=f"{args.x_width*i},{-args.y_width*p}!",
                    fixedsize="true",
                    width="3",
                    height="2",
                    fontsize="10",
                    fontcolor="black" if node_relevance[p].item() < 0.8 * args.node_vmax else "white",
                    fontname="helvetica",
                    fillcolor=value_to_color(node_relevance[p].item(), args.node_vmax, args.node_vmin),
                )

            else:

                node_name = f"L{i-1}#E"
                G.node(
                    node_name,
                    label=f"{node_name}\n{node_relevance[p].item() * 100:2.1f}%",
                    pos=f"{args.x_width*i},{-args.y_width*p}!",
                    fixedsize="true",
                    width="3",
                    height="2",
                    fontsize="14",
                    fontcolor="black" if node_relevance[p].item() < 0.8 * args.node_vmax else "white",
                    fontname="helvetica",
                    fillcolor=value_to_color(node_relevance[p].item(), args.node_vmax, args.node_vmin),
                )

    # Logit Node
    G.node(
        "logit",
        label=f"Out#{label.item()}\n{100}%\n\n{logit_node}",
        pos=f"{args.x_width*(i+1)},0!",
        fixedsize="true",
        width="3",
        height="2",
        fontsize="14",
        fontcolor="white",
        fontname="helvetica",
        fillcolor=value_to_color(1.00, args.node_vmax, args.node_vmin),
    )

    # Edge
    for i in range(1, layer_num-1):
        # edge_relevance = normalized_relevances.block[i].edge
        edge_relevance = edge_relevances[i]
        source, target = node_indices[i], node_indices[i+1]

        for p, s in enumerate(source):
            for q, t in enumerate(target):
                source_node_name = f"L{i-1}#{s}" if s != error_nums[i] else f"L{i-1}#E"
                target_node_name = f"L{i}#{t}" if t != error_nums[i+1] else f"L{i}#E"
                G.edge(
                    source_node_name,
                    target_node_name,
                    # label=f"{edge_relevance[p, q].item() * 100:.1f}%", # uncomment to show edge weight
                    # labeldistance="100",
                    # labelfloat="true",
                    # decorate="true",
                    fontsize="10",
                    fontcolor="black",
                    fontname="helvetica",
                    penwidth=f"{min(edge_relevance[p, q].item() * 20, 5):.2f}",
                    # penwidth="1.0",
                    headport="w",
                    tailport="e",
                    arrowhead="none",
                    color=value_to_color(edge_relevance[p, q].item(), args.edge_vmax, args.edge_vmin),
                )

    # Logit Edge
    # edge_relevance = normalized_relevances.block[layer_num-1].edge
    edge_relevance = edge_relevances[layer_num-1]
    source = node_indices[layer_num-1]
    for p, s in enumerate(source):
        source_node_name = f"L{layer_num-2}#{s}" if s != error_nums[layer_num-1] else f"L{layer_num-2}#E"
        G.edge(
            source_node_name,
            "logit",
            # label=f"{edge_relevance[p, 0].item() * 100:.1f}%", # uncomment to show edge weight
            fontsize="10",
            fontcolor="black",
            fontname="helvetica",
            penwidth=f"{min(edge_relevance[p, 0].item() * 20, 5):.2f}",
            # penwidth="1.0",
            headport="w",
            tailport="e",
            arrowhead="none",
            color=value_to_color(edge_relevance[p, 0].item(), args.edge_vmax, args.edge_vmin),
        )
    
    G.render(f"{args.output_path}/imagenet_{logit_node}_{args.data_index}_{args.model}", format="png", cleanup=True)
    for fname in os.listdir(args.temporal_save_path):
        os.remove(os.path.join(args.temporal_save_path, fname))

if __name__ == "__main__":
    main()
