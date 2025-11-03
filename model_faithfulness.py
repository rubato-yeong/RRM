"""
    model_faithfulness.py

    This script evaluates the faithfulness of feature or neuron activations within 
    Vision Transformer (ViT)-based cascaded models using various relevance-based criteria. 
    It systematically masks internal representations (feature/neuron nodes) according to
    predefined rules and measures the impact on model predictions (e.g., logits, probabilities, accuracy).
    Note that to get the final faithfulness score, you should evaluate the AUC of the obtained faithfulness curves. We separated this script to simplify additional analysis of those curves.

    Supported models:
        - ViT
        - DINOv2
        - CLIP-ViT

    Main Features:
        - Relevance-based node ranking using gradient or integrated gradients
        - Graph simplification rules: top-k, top-n, threshold (based on relevance)
        - Supports random masking for baseline comparison
        - Insertion and deletion evaluation (faithfulness curve)
        - Hierarchical attribution and leaf error handling for TopKSAE
        - Supports feature- or neuron-level attribution

    Evaluation metrics:
        - Raw logit, adjusted logit (relative to logit mean), probability, accuracy
        - Node count used at each iteration (for plotting faithfulness curves)

    Arguments:
        --model: Model type (vit, dinov2, clip_vit)
        --evaluation: Faithfulness evaluation mode ("insertion" or "deletion")
        --target: Attribution target ("feature" for SAE, "neuron" for ViT)
        --graph_rule: Relevance-based node selection method ("node", "edge", or "random")
        --graph_criterion: Graph simplification criterion ("topn", "topk", or "threshold")
        --method: Relevance computation method ("grad" or "ig")
        --subset_length: Number of correctly classified samples to evaluate
        --skip_early_layers: Whether to skip the first 1/3 layers during masking
        --leaf_error: Whether to use predicted error activations for masking
        --hierarchical_attribution: Whether to enable hierarchical masking per feature
        --libragrad: Use Libragrad mode (if set to False, uses built-in gradients)
        --adjust_logit: Subtract logit mean when computing adjusted logit
        --output_path: Output root directory to save results as pickle files  (e.g., /USER/faithfulness)
        --cache_path: Path for loading cached data (SAE checkpoints, indices, etc.)   (e.g., /USER/cache/SAEs/imagenet/)
        --dino_head_ckpt_path: Path to the pretrained linear classification head for DINOv2
        --clip_text_embed_path: Path to cached CLIP text embeddings (.pt) for logit projection
        --sae_ckpt_path: Path to pretrained Top-K SAE model checkpoints
        --mean_acts_path: Directory containing median activation tensors (used for masking)  (e.g., )
            > Refer to get_avg_activation.py for details on how to generate these tensors
        --data_root: Path to ImageNet validation dataset root directory  (e.g., /USER/Data/CLS-LOC/val)

    Outputs:
        - Saves per-image evaluation dictionaries as .pkl files containing:
        {
            "node_count": List[int],
            "iter_count": List[float],
            "logit": List[float],
            "adjusted_logit": List[float],
            "probability": List[float],
            "accuracy": List[bool],
        }

    Output path structure:
        output_path/<evaluation>/<target>/<model>/<setting>/imageXXXXX.pkl

    Example usage:
        python model_faithfulness.py \
            --model vit \
            --evaluation deletion \
            --target feature \
            --method grad \
            --graph_rule node \
            --graph_criterion topn \
            --subset_length 500 \
            --output_path ./faithfulness_results
"""


import argparse
import os
import pickle
import random
import sys
sys.path.append("..")

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import CLIPProcessor

from utils import set_seed, collate_fn_clip, ImageNetValDataset, get_imagenet_transform, create_cascaded_model
from utils import get_simplified_graph_by_node, get_simplified_graph_by_edge, get_relevance, get_normalized_relevance


def print_args(args):
    print("=" * 50)
    print("Parsed Arguments:")
    print("=" * 50)
    print(f"Model Type: {args.model}")
    print(f"Device: {args.device}")
    print(f"Evaluation Method: {args.evaluation}")
    print(f"Subset Length: {args.subset_length}")
    print(f"Target: {args.target}")
    print(f"Use Libragrad: {args.libragrad}")
    print(f"Relevance Computation Method: {args.method}")
    print(f"Adjust Logit: {args.adjust_logit}")
    print(f"Graph Rule: {args.graph_rule}")
    print(f"Graph Criterion: {args.graph_criterion}")
    print(f"Use Negative Relevance: {args.graph_negative}")
    print(f"Use Leaf Error: {args.leaf_error}")
    print(f"Hierarchical Attribution: {args.hierarchical_attribution}")
    print("=" * 50)
    

def parse_args():
    parser = argparse.ArgumentParser(description="Faithfulness Evaluation Script")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for DataLoader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model", type=str, required=True, choices=["vit", "dinov2", "clip_vit"], help="Model type")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="Device to use for computation")

    parser.add_argument("--evaluation", "-e", type=str, default="insertion", choices=["insertion", "deletion"], help="Evaluation method")
    parser.add_argument("--subset_length", type=int, default=1500, help="Length of the subset to evaluate")
    parser.add_argument("--target", type=str, default="feature", choices=["feature", "neuron"], help="Target for evaluation")

    parser.add_argument("--skip_early_layers", action="store_true", help="Skip early layers in the model for evaluation")
    parser.add_argument("--gamma_rule", type=str, default=None, help="Gamma rule for evaluation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument("--libragrad", action="store_false", help="Use libragrad")
    parser.add_argument("--method", type=str, default="grad", choices=["grad", "ig"], help="Method for relevance computation")
    parser.add_argument("--adjust_logit", action="store_false", help="Adjust logit value")
    parser.add_argument("--graph_rule", "-r", type=str, default="node", choices=["random", "node", "edge"], help="Graph rule for evaluation")
    parser.add_argument("--graph_criterion", "-c", type=str, default="topn", choices=["topn", "topk", "threshold"], help="Graph criterion for evaluation")
    parser.add_argument("--graph_negative", "-n", action="store_true", help="Use negative relevance in graph criterion")
    parser.add_argument("--leaf_error", "-l", action="store_false", help="Use leaf error for relevance computation")
    parser.add_argument("--hierarchical_attribution", "-x", action="store_true", help="Use hierarchical attribution")

    parser.add_argument("--output_path", type=str, default="", help="Path to save output data")
    parser.add_argument("--cache_path", type=str, default="", help="Path to cache data")
    parser.add_argument("--sae_ckpt_path", type=str, default="", help="Path to SAE checkpoint")
    parser.add_argument("--mean_acts_path", type=str, default="", help="Path to mean activation data")
    parser.add_argument("--data_root", type=str, default="", help="Path to ImageNet validation data")
    parser.add_argument("--dino_head_ckpt_path", type=str, default="", help="Path to DINO head checkpoint")
    parser.add_argument("--clip_text_embed_path", type=str, default="", help="Path to CLIP text embeddings")

    args = parser.parse_args()
    print_args(args)

    return args


def get_correct_indices(vision_model, dataset, args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vision_model.eval()

    correct_indices = []

    dataloader = DataLoader(  # This time, batch_size can be larger than 1.
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=(args.collate_fn if args.model == "clip_vit" else None)
    )

    idx_counter = 0
    total_samples = len(dataset)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Finding correct predictions"):
            images = images.to(device)
            labels = labels.to(device)

            logits = vision_model(images)

            preds = logits.argmax(dim=1)

            for i in range(len(images)):
                if preds[i].item() == labels[i].item():
                    correct_indices.append(idx_counter + i)

            idx_counter += len(images)

    num_correct = len(correct_indices)
    accuracy = num_correct / total_samples * 100
    print(f"\nCorrect predictions: {num_correct}/{total_samples} ({accuracy:.2f}%)")

    return correct_indices


def main():
    args = parse_args()

    set_seed(args.seed)

    #! HARD CODED
    D = 768

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

    # Find correct indices
    correct_indices_path = Path(f"{args.cache_path}/correct_indices/{args.model}.pt")
    correct_indices_path.parent.mkdir(parents=True, exist_ok=True)

    if correct_indices_path.exists():
        print("Loading cached correct indices...")
        with open(correct_indices_path, "rb") as f:
            correct_indices = pickle.load(f)
    else:
        correct_indices = get_correct_indices(cascaded_model, dataset, args)
        with open(correct_indices_path, "wb") as f:
            pickle.dump(correct_indices, f)

    # Subset sampling
    if args.subset_length is not None:
        subset_size = int(args.subset_length)
        assert subset_size <= len(correct_indices), "subset_length is larger than available correct samples!"

        g = torch.Generator()
        g.manual_seed(args.seed)

        rand_indices = torch.randperm(len(correct_indices), generator=g)[:subset_size]
        correct_indices = [correct_indices[i] for i in rand_indices.tolist()]

    subset = Subset(dataset, correct_indices)
    dataloader = DataLoader(
        subset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=(args.collate_fn if args.model == "clip_vit" else None)
    )
    print(f"Total samples selected: {len(subset)}")

    # Load mean & base feature acts
    if args.target == "feature":
        with open(os.path.join(args.mean_acts_path, f"median_feature_acts_{args.model}.pkl"), "rb") as f:
            args.mean_feature_acts = pickle.load(f)
        with open(os.path.join(args.mean_acts_path, f"median_error_acts_{args.model}.pkl"), "rb") as f:
            args.mean_error_acts = pickle.load(f)
        for i in range(11):
            args.mean_feature_acts[i] = args.mean_feature_acts[i].to(args.device) 
            args.mean_error_acts[i] = args.mean_error_acts[i].to(args.device)
        args.mean_neuron_acts = None
        
    elif args.target == "neuron":
        with open(os.path.join(args.mean_acts_path, f"median_neuron_acts_{args.model}.pkl"), "rb") as f:
            args.mean_neuron_acts = pickle.load(f)
        for i in range(11):
            args.mean_neuron_acts[i] = args.mean_neuron_acts[i].to(args.device)
        args.mean_feature_acts = None
        args.mean_error_acts = None
        
    else:
        raise ValueError("Invalid target. Choose either 'feature' or 'neuron'.")
        
        
    node_num = 0
    max_node_num = 0
    if args.target == "feature":
        for sae_model in cascaded_model.sae_models.values():
            node_num += sae_model.dict_size + 1
            if sae_model.dict_size + 1 > max_node_num:
                max_node_num = sae_model.dict_size + 1
    elif args.target == "neuron": #! HARD CODED
        for sae_model in cascaded_model.sae_models.values():
            node_num += D
        max_node_num = D 
    if args.graph_criterion == "topn":
        topn_list = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
        iterator = topn_list
    elif args.graph_criterion == "topk":
        topk_list = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
        iterator = topk_list
    elif args.graph_criterion == "threshold":
        threshold_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.9999, 1]
        iterator = threshold_list
    else:
        raise ValueError("Invalid graph criterion. Choose either 'topk' or 'threshold'.")


    for i, (image, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing images"):

        faithfulness_dict = {
            "node_count": [],
            "iter_count": [],
            "logit": [],
            "adjusted_logit": [],
            "probability": [],
            "accuracy": [],
        }

        image = image[0].to(args.device)
        label = label.to(args.device)
        logits = cascaded_model(image.unsqueeze(0))
        true_error_acts = cascaded_model.get_true_error_acts()
        
        if args.graph_rule != "random" and not args.hierarchical_attribution:

            relevances = get_relevance(
                model=cascaded_model,
                image=image,
                label=label,
                method=args.method,
                target=args.target,
                leaf_error=args.leaf_error,
                mean_feature_acts=args.mean_feature_acts,
                mean_error_acts=args.mean_error_acts,
                mean_neuron_acts=args.mean_neuron_acts,
                get_logit=False,
                adjust_logit=args.adjust_logit,
                compute_edge=False if args.graph_rule == "node" and not args.leaf_error else True,
            )

            normalized_relevances = get_normalized_relevance(relevances=relevances, negative_relevance=args.graph_negative)

        for iter in iterator:

            if iter == 0: # No node
                node_indices = {i+1: torch.tensor([]).to(args.device) for i in range(len(cascaded_model.sae_models))}
            elif iter == 1: # All nodes
                if args.target == "feature":
                    node_indices = {i+1: torch.arange(cascaded_model.sae_models[i].dict_size + 1).to(args.device) for i in range(len(cascaded_model.sae_models))}
                elif args.target == "neuron":
                    node_indices = {i+1: torch.arange(D).to(args.device) for i in range(len(cascaded_model.sae_models))}
            else:
                # Get the node indices
                if args.graph_rule == "random":
                    if args.target == "feature":
                        node_indices = {i+1: torch.tensor(random.sample(range(cascaded_model.sae_models[i].dict_size + 1), int(iter * (cascaded_model.sae_models[i].dict_size + 1)))).to(args.device) for i in range(len(cascaded_model.sae_models))}
                    elif args.target == "neuron":
                        node_indices = {i+1: torch.tensor(random.sample(range(D), int(iter * D))).to(args.device) for i in range(len(cascaded_model.sae_models))}
                elif args.graph_rule == "node":
                    node_indices = get_simplified_graph_by_node(
                        normalized_relevances=normalized_relevances,
                        topn=iter if args.graph_criterion == "topn" else None,
                        topk=iter if args.graph_criterion == "topk" else None,
                        threshold=iter if args.graph_criterion == "threshold" else None,
                        max_node_num=max_node_num if args.graph_criterion == "topn" else None,
                        return_values=False,
                    )
                elif args.graph_rule == "edge":
                    node_indices = get_simplified_graph_by_edge(
                        normalized_relevances=normalized_relevances,
                        topn=iter if args.graph_criterion == "topn" else None,
                        topk=iter if args.graph_criterion == "topk" else None,
                        threshold=iter if args.graph_criterion == "threshold" else None,
                        max_node_num=max_node_num if args.graph_criterion == "topn" else None,
                        return_values=False,
                    )
            
            # Node shift (l -> l-1)
            new_node_indices = {}
            for layer, indices in node_indices.items():
                new_node_indices[layer-1] = indices
            node_indices = new_node_indices
            
            if args.evaluation == "insertion":
                # Reverse the node indices
                new_node_indices = {}
                for layer, indices in node_indices.items():
                    if args.target == "feature":
                        full_node_indices = torch.arange(cascaded_model.sae_models[layer].dict_size + 1).to(args.device)
                    elif args.target == "neuron":
                        full_node_indices = torch.arange(D).to(args.device)
                    mask = ~torch.isin(full_node_indices, indices)
                    new_node_indices[layer] = full_node_indices[mask]
                node_indices = new_node_indices
            
            if args.skip_early_layers:
                # Skip the first 1/3 layers
                for layer in range(round(len(node_indices) / 3)):
                    node_indices[layer] = torch.tensor([])


            logits = cascaded_model(
                image.unsqueeze(0),
                mask_dict=node_indices,
                mean_feature_acts=args.mean_feature_acts,
                mean_error_acts=args.mean_error_acts,
                true_error_acts=true_error_acts if args.leaf_error else None,
                mean_neuron_acts=args.mean_neuron_acts,
            ).squeeze()  # [1000]

            # Get the number of nodes
            node_count = node_num - sum([len(v) for v in node_indices.values()])

            # Get the logit values
            logit = logits[label].item()
            adjusted_logits = (logits[label] - logits.mean()).item()
            probability = torch.softmax(logits, dim=0)[label].item()
            accuracy = (logits.argmax(dim=0) == label).item()

            # Save the results
            faithfulness_dict["node_count"].append(node_count)
            faithfulness_dict["iter_count"].append(iter)
            faithfulness_dict["logit"].append(logit)
            faithfulness_dict["adjusted_logit"].append(adjusted_logits)
            faithfulness_dict["probability"].append(probability)
            faithfulness_dict["accuracy"].append(accuracy)

        # Write the results to a file
        img_idx = subset.indices[i]
        
        setting = f"{args.method}"
        setting = setting + f"_{args.graph_rule}"
        setting = setting + f"_{args.graph_criterion}"
        
        if args.libragrad: setting = "libragrad_" + setting
        if args.graph_negative: setting = setting + "_negative"
        if args.hierarchical_attribution: setting = setting + "_hierarchical"
        if args.adjust_logit: setting = setting + "_adjust_logit"
        
        if args.graph_rule == "random": setting = "random"
        if args.leaf_error and args.target != "neuron": setting = setting + "_leaf_error"
        if args.skip_early_layers: setting = setting + "_skip_early"
        
        save_path = os.path.join(args.output_path, f"{args.evaluation}/{args.target}/{args.model}/{setting}/image{img_idx+1:05d}.pkl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(faithfulness_dict, f)


if __name__ == "__main__":
    main()
