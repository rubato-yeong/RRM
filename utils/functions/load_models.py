from pathlib import Path

import timm
import torch
import torch.nn.functional as F

from utils.modules.vit import CascadedViTWithSAEforGrad, CascadedViTWithSAEforAct
from utils.modules.dinov2 import Dinov2Model, CascadedDinov2WithSAEforGrad, CascadedDinov2WithSAEforAct
from utils.modules.clip_vit import CascadedClipViTWithSAEforGrad, CascadedClipViTWithSAEforAct
from utils.modules.sae import TopKSAE
from utils.functions.functions import custom_format


def load_default_vision_model(args):
    """
    Load a default vision model (ViT, DINOv2, or CLIP ViT) based on `args.model`.

    Args:
        args: Argument namespace that must include `model` (str) and `device` (str or torch.device).
        head_ckpt_path: Path to the pretrained linear head checkpoint for DINOv2 (used only if args.model == 'dinov2').
    """
    if args.model == "vit":
        vision_model = timm.create_model('vit_base_patch16_224', pretrained=True).to(args.device)
    elif args.model == "dinov2":
        model_full = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg_lc', pretrained=True)
        backbone = model_full.backbone.to(args.device)
        vision_model = Dinov2Model(backbone, num_classes=1000, head_ckpt_path=args.dino_head_ckpt_path, device=args.device).to(args.device)
    elif args.model == "clip_vit":
        from transformers import CLIPModel
        vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(args.device)
    return vision_model


def load_default_sae_configs(args):
    """
    Return predefined SAE hyperparameter configurations for each transformer block 
    based on the selected vision model.

    Args:
        args: Argument namespace that must include `model` (str).

    Returns:
        Dictionary mapping layer indices (int) to SAE config dicts with keys:
            - "K": Top-K number of features to be activated.
            - "R": Dictionary size scaling ratio.
    """
    if args.model == "vit":
        config_SAEs = {  # Post-block-indexed
            0: {"K": 4, "R": 0.25},
            1: {"K": 32, "R": 0.5},
            2: {"K": 64, "R": 1},
            3: {"K": 96, "R": 2},
            4: {"K": 96, "R": 2},
            5: {"K": 128, "R": 2},
            6: {"K": 128, "R": 4},
            7: {"K": 160, "R": 4},
            8: {"K": 160, "R": 4},
            9: {"K": 160, "R": 4},
            10: {"K": 128, "R": 2},
        }
    elif args.model == "dinov2":
        config_SAEs = {
            0: {"K": 4, "R": 0.25},
            1: {"K": 16, "R": 0.25},
            2: {"K": 32, "R": 0.5},
            3: {"K": 32, "R": 2},
            4: {"K": 64, "R": 2},
            5: {"K": 96, "R": 2},
            6: {"K": 128, "R": 2},
            7: {"K": 128, "R": 2},
            8: {"K": 128, "R": 2},
            9: {"K": 160, "R": 2},
            10: {"K": 160, "R": 2},
        }
    elif args.model == "clip_vit":
        config_SAEs = {
            0: {"K": 64, "R": 0.5},
            1: {"K": 64, "R": 0.5},
            2: {"K": 96, "R": 1},
            3: {"K": 96, "R": 2},
            4: {"K": 128, "R": 2},
            5: {"K": 128, "R": 2},
            6: {"K": 128, "R": 2},
            7: {"K": 128, "R": 2},
            8: {"K": 160, "R": 2},
            9: {"K": 160, "R": 2},
            10: {"K": 160, "R": 2},
        }
    else:
        raise ValueError(f"Model '{args.model}' is not supported.")
    return config_SAEs


def load_default_sae_args(K, R, L):
    """
    Return a standard set of arguments for initializing a single SAE module.

    Args:
        K: Number of top-k features to use in training.
        R: Dictionary size ratio.
        L: The index of the model layer to which this SAE is attached.
    """
    return {
        "act_size": 768,  # Assuming ViT-B
        "aux_penalty": 0.03125,
        "input_unit_norm": True,
        "l1_coeff": 0.001,
        "n_batches_to_dead": 60,
        "seed": 42,
        "top_k_aux": 256,
        "epoch": 50,
        "top_k": K,
        "dict_size_R": R,
        "model_layer": L,
    }


def load_cascaded_model_for_grad(vision_model, sae_models, args):
    """
    Create a cascaded model by integrating the vision backbone with SAE modules.

    Args:
        vision_model: Pretrained vision model (ViT, DINOv2, or CLIP).
        sae_models: Dictionary mapping block indices to trained SAE models.
        args: Argument namespace that must include `model` and `device`.
    """
    
    if args.model == "vit":
        cascaded_model = CascadedViTWithSAEforGrad(
            vit_model=vision_model,
            sae_models=sae_models, 
            compute_graph=True,
            libragrad=getattr(args, 'libragrad', False),
            gamma_rule=getattr(args, 'gamma_rule', None),
            verbose=getattr(args, 'verbose', False),
        ).to(args.device)
    elif args.model == "dinov2":
        cascaded_model = CascadedDinov2WithSAEforGrad(
            dinov2_model=vision_model,
            sae_models=sae_models,
            compute_graph=True,
            libragrad=getattr(args, 'libragrad', False),
            gamma_rule=getattr(args, 'gamma_rule', None),
            verbose=getattr(args, 'verbose', False),
        ).to(args.device)
    elif args.model == "clip_vit":
        text_features = torch.load(args.clip_text_embed_path, map_location=args.device, weights_only=False)
        text_features = torch.tensor(text_features, dtype=torch.float32).to(args.device)
        text_features = F.normalize(text_features, p=2, dim=-1).to(args.device)
        cascaded_model = CascadedClipViTWithSAEforGrad(
            vit_model=vision_model,
            sae_models=sae_models,
            text_features=text_features,
            compute_graph=True,
            libragrad=getattr(args, 'libragrad', False),
            gamma_rule=getattr(args, 'gamma_rule', None),
            verbose=getattr(args, 'verbose', False),
        ).to(args.device)
    else:
        raise ValueError(f"Model '{args.model}' is not supported.")
    
    return cascaded_model


def load_cascaded_model_for_act(vision_model, sae_models, args):
    if args.model == "vit":
        cascaded_model = CascadedViTWithSAEforAct(
            vit_model=vision_model,
            sae_models=sae_models, 
        ).to(args.device)
    elif args.model == "dinov2":
        cascaded_model = CascadedDinov2WithSAEforAct(
            dinov2_model=vision_model,
            sae_models=sae_models,
        ).to(args.device)
    elif args.model == "clip_vit":
        text_features = torch.load(args.clip_text_embed_path, map_location=args.device, weights_only=False)
        text_features = F.normalize(text_features, p=2, dim=-1).to(args.device)
        cascaded_model = CascadedClipViTWithSAEforAct(
            vit_model=vision_model,
            sae_models=sae_models,
            text_features=text_features,
        ).to(args.device)
    
    return cascaded_model
    


def create_cascaded_model(args, mode="grad"):
    """
    Builds and returns a cascaded vision model that integrates a pretrained backbone with SAE modules.

    This function performs the following steps:
    1. Loads a pretrained vision backbone (ViT, DINOv2, or CLIP) according to `args.model`.
    2. Loads or constructs Sparse Autoencoders (SAEs) for each transformer block using default or ablated configurations.
    3. Optionally applies R-ablation to modify dictionary size scaling (`args.ablate_R`).
    4. Loads cached SAE models if available, otherwise builds them from scratch using stored checkpoints.
    5. Combines the vision model and SAEs into a `Cascaded[Model]WithSAE` wrapper (e.g., `CascadedViTWithSAE`).
    6. Returns the final model with `.eval()` mode enabled.

    Args:
        args (argparse.Namespace): Configuration arguments containing:
            - model (str): One of {"vit", "dinov2", "clip_vit"}.
            - device (str): Device to use (e.g., "cuda:0").
            - cache_path (str): Path to save/load cached SAE models.
            - sae_ckpt_path (str): Path to pretrained SAE checkpoints.
            - epoch (int, default=50): Checkpoint epoch to load.
            - ablate_R (str, optional): R-ablation setting; one of {"never", "low", "high"}. Defaults to "never".
            - dino_head_ckpt_path (required if args.model == "dinov2"): Path to the pretrained linear head checkpoint for DINOv2.
            - clip_text_embed_path (required if args.model == "clip_vit"): Required only for clip_vit when used with textual supervision.
            - libragrad (bool, optional): If True, use LibraGrad for gradient computation. Defaults to False.
            - gamma_rule (Union[None, float], optional): If not None, apply gamma rule for gradient computation. Defaults to None.
            - verbose (bool, optional): If True, print additional information. Defaults to False.
            - mode (str, optional): The mode for the cascaded model. ("grad" or "act"). Defaults to "grad".

    Returns:
        nn.Module: A cascaded model (`Cascaded[Model]WithSAE`) integrating the vision transformer and SAE modules.
    """
    # Load Vision Transformer
    print("Loading Vision Transformer...")
    vision_model = load_default_vision_model(args)
    vision_model.eval()
    
    # Load SAEs
    print("Loading SAEs...")
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" in args.device else "cpu")
    config_SAEs = load_default_sae_configs(args)
    
    ablate_R = getattr(args, "ablate_R", "never")
    if ablate_R == "never":
        cache_file = Path(f"{args.cache_path}/{args.model}.pt")
    elif ablate_R == "low":
        config_SAEs = {i: {"K": v["K"], "R": 0.5} for i, v in config_SAEs.items()}  # NOTE: We propose to use 0.5 for low-R ablation
        cache_file = Path(f"{args.cache_path}/{args.model}_lowR.pt")
    elif ablate_R == "high":
        config_SAEs = {i: {"K": v["K"], "R": 4} for i, v in config_SAEs.items()}    # NOTE: We propose to use 4.0 for high-R ablation
        cache_file = Path(f"{args.cache_path}/{args.model}_highR.pt")
    else:
        raise ValueError(f"{args.ablate_R} is not supported.")
        
    sae_models = {}

    if cache_file.exists():
        print("Loading cached SAE data...")
        with torch.serialization.safe_globals([TopKSAE]):
            cache_data  = torch.load(cache_file, map_location=device)
            sae_models  = cache_data["sae_models"]
            config_SAEs = cache_data["config_SAEs"]
    else:
        for layernum in config_SAEs.keys():
            sae_cfg = load_default_sae_args(K=config_SAEs[layernum]['K'], R=config_SAEs[layernum]['R'], L=layernum)
            sae_cfg['device'] = args.device
            sae_model = TopKSAE(sae_cfg).to(device)
            checkpoint_path = f"{args.sae_ckpt_path}/{args.model}/K{sae_cfg['top_k']}_R{custom_format(sae_cfg['dict_size_R'])}/L{sae_cfg['model_layer']}/epoch{sae_cfg['epoch']}.pth"
            checkpoint = torch.load(checkpoint_path, map_location=device)
            sae_model.load_state_dict(checkpoint["model_state_dict"])
            sae_model.eval()
            sae_models[layernum] = sae_model

    # Save (Update) the models from wandb and the loaded models to the cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        torch.save({"sae_models": sae_models, "config_SAEs": config_SAEs}, f)
    
    # Load cascaded model
    print("Loading cascaded model...")
    if mode == "grad":
        cascaded_model = load_cascaded_model_for_grad(vision_model, sae_models, args)
    elif mode == "act":
        cascaded_model = load_cascaded_model_for_act(vision_model, sae_models, args)
    else:
        raise ValueError(f"Mode '{mode}' is not supported.")
    cascaded_model.eval()
    
    return cascaded_model