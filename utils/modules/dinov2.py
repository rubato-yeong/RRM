from types import SimpleNamespace
from typing import Optional, Union, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from utils.functions.functions import einsum
from utils.modules.modelgrad import ModelforGrad
from utils.modules.relevances import Relevances
from utils.modules.fullgrad import FullGradLayerNorm, FullGradGELU



class CascadedDinov2WithSAEforGrad(nn.Module, ModelforGrad):
    def __init__(
        self,
        dinov2_model: nn.Module,
        sae_models: dict,
        compute_graph: bool = False,
        relevance_norm: bool = False,
        softmax_correction: bool = False,
        relevance_debug: bool = False,
        libragrad: bool = False,
        gamma_rule: Union[float, None] = None, # deprecated
        verbose: bool = False,
    ):
        """
        Args:
            dinov2_model: Dinov2 model.
            sae_models: Dictionary of autoencoder models for each block.
            compute_graph: Flag to compute the graph for relevance propagation.
            relevance_norm: Not implemented.
            softmax_correction: Not implemented.
            relevance_debug: Flag to store intermediate relevance scores for debugging.
            libragrad: Flag to use libragrad for gradient computation.
                       (https://www.arxiv.org/abs/2411.16760)
            verbose: Flag to print debug information.
        """
        super().__init__()

        # Initialize the models
        self.dinov2 = dinov2_model
        self.sae_models = sae_models  # e.g., {0: sae0, 1: sae1, ...}
        
        # Compute graph flag
        self.compute_graph = compute_graph
        self.relevance_norm = relevance_norm
        self.softmax_correction = softmax_correction
        self.relevance_debug = relevance_debug
        self.libragrad = libragrad
        self.verbose = verbose
        self.relevances = Relevances(
            vit_model=dinov2_model,
            model="dinov2",
            relevance_debug=self.relevance_debug,
        )
        self.time = 0.0

        # Set vision transformer blocks
        self.vit_blocks = nn.ModuleList([
            Dinov2Block(
                block,
                self.compute_graph,
                self.libragrad,
            ) for block in self.dinov2.blocks
        ])
        self.layer_len = len(self.vit_blocks)

        # Post-block operations
        if self.libragrad:
            self.final_norm = FullGradLayerNorm(
                weight=self.dinov2.norm.weight,
                bias=self.dinov2.norm.bias,
                eps=self.dinov2.norm.eps,
            )
        else:
            self.final_norm = self.dinov2.norm


    def forward(
        self, 
        x, 
        mask_dict=None, 
        mean_feature_acts=None, 
        mean_error_acts=None,
        true_error_acts=None,
        mean_neuron_acts=None,
    ):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor image of shape [B, C, H, W]. Batch size should be 1.
            mask_dict: Dictionary of masks for each block SAE features.
        """
        assert x.shape[0] == 1, "CascadedDinov2WithSAE model supports only a single input."

        # Preprocess input
        x = self.dinov2.backbone.prepare_tokens_with_masks(x)  # [B, T+5, D]
        B, N, D = x.shape

        self.n_tokens = N+1+4 # equal to T
        mask = None

        # Pass through the transformer blocks
        for i, block in enumerate(self.vit_blocks):
            x = block(x)  # [B, T, D]
            
            if i != self.layer_len-1:
                if mask_dict is not None and mean_neuron_acts is not None:
                    mask_features = mask_dict[i]
                    if isinstance(mask_features, torch.Tensor):
                        mask_features = mask_features.long()  # Ensure it's long type
                    else: # Convert list to tensor (if it's a list)
                        mask_features = torch.tensor(mask_features, dtype=torch.long, device=self.cfg["device"])
                    x[:, :, mask_features] = mean_neuron_acts[i][:, mask_features]
            
            if i in self.sae_models:
                B, T, D = x.shape
                x_flat = x.reshape(1, B * T, D)  # [1, B*T, D]
                
                if mask_dict is None:
                    sae_out_dict = self.sae_models[i](x_flat, mask_features=mask, compute_graph=self.compute_graph)
                
                elif mean_feature_acts is not None and mean_error_acts is not None:
                    
                    sae_out_dict = self.sae_models[i](
                        x_flat,
                        mask_features=mask_dict[i],
                        compute_graph=self.compute_graph,
                        mean_feature_acts=mean_feature_acts[i],
                        mean_error_acts=mean_error_acts[i],
                        true_error_acts=true_error_acts[i] if true_error_acts is not None else None,
                    )
                    sae_out = sae_out_dict["sae_out_with_mask"].reshape(B, T, D)
                    sae_error = sae_out_dict["sae_error"].reshape(B, T, D)
                    x = sae_out + sae_error  # [B, T, D]
        
        # Classification head
        x = self.final_norm(x)
        self.cls_token = x[:, 0, :]  # [1, D]
        self.patch_tokens = x[:, 5:, :]  # [1, T-5, D]
        self.mean_patch_tokens = self.patch_tokens.mean(dim=1) # [1, D]; Skip 4 reg tokens
        self.head_input = torch.cat((self.cls_token, self.mean_patch_tokens), dim=1)  # [1, 2*D]
        logits = self.dinov2.head(self.head_input)  # [1, num_classes]
        
        return logits



class CascadedDinov2WithSAEforAct(nn.Module):
    def __init__(
        self,
        dinov2_model: nn.Module,
        sae_models: dict,
    ):
        """
        Args:
            dinov2_model: Dinov2 model.
            sae_models: Dictionary of autoencoder models for each block.
        """
        super().__init__()

        # Initialize the models
        self.dinov2 = dinov2_model
        self.sae_models = sae_models  # e.g., {0: sae0, 1: sae1, ...}

        # Set vision transformer blocks
        self.vit_blocks = nn.ModuleList([
            Dinov2Block(
                block,
                compute_graph=False,
                libragrad=False,
            ) for block in self.dinov2.blocks
        ])
        self.layer_len = len(self.vit_blocks)

        # Post-block operations
        self.final_norm = self.dinov2.norm

        # Monitoring
        self.feature_acts = {}
        self.error_acts = {}
        self.neuron_acts = {}


    def forward(self, x, mask_dict=None, type="mean"):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor image of shape [B, C, H, W]. Batch size should be 1.
            mask_dict: Dictionary of masks for each block SAE features.
            type: Type of activation to compute. (mean or median)
        """
        # assert x.shape[0] == 1, "CascadedDinov2WithSAE model supports only a single input."

        # Preprocess input
        x = self.dinov2.backbone.prepare_tokens_with_masks(x)  # [B, T+5, D]
        B, N, D = x.shape
        # cls_tokens = x[:, 0, :]    # [B, 1, D]
        # reg_tokens = x[:, 1:5, :]  # [B, 4, D]

        self.n_tokens = N+1+4 # equal to T
        mask = None

        # Pass through the transformer blocks
        for i, block in enumerate(self.vit_blocks):
            x = block(x)  # [B, T, D]

            if type == "median":
                neuron_acts = x.median(dim=0).values  # [T, D]
                if self.neuron_acts.get(i) is None: self.neuron_acts[i] = neuron_acts
            
            if i in self.sae_models:
                B, T, D = x.shape
                x_flat = x.reshape(1, B * T, D)  # [1, B*T, D]
                sae_out_dict = self.sae_models[i](x_flat, mask_features=None, compute_graph=False)
                
                if type == "mean":
                    feature_acts = sae_out_dict["feature_acts"].reshape(B, T, -1).sum(dim=0) # [T, F]
                    if self.feature_acts.get(i) is None: self.feature_acts[i] = feature_acts
                    else: self.feature_acts[i] += feature_acts
                    error_acts = sae_out_dict["sae_error"].reshape(B, T, D).sum(dim=0) # [T, D]
                    if self.error_acts.get(i) is None: self.error_acts[i] = error_acts
                    else: self.error_acts[i] += error_acts
                elif type == "median":
                    feature_acts = sae_out_dict["feature_acts"].reshape(B, T, -1).median(dim=0).values # [T, F]
                    if self.feature_acts.get(i) is None: self.feature_acts[i] = feature_acts
                    error_acts = sae_out_dict["sae_error"].reshape(B, T, D).median(dim=0).values # [T, D]
                    if self.error_acts.get(i) is None: self.error_acts[i] = error_acts
        
        # Classification head
        # self.before_head_before_ln = x
        x = self.final_norm(x)
        # self.before_head = x
        self.cls_token = x[:, 0, :]  # [1, D]
        self.patch_tokens = x[:, 5:, :]  # [1, T-5, D]
        self.mean_patch_tokens = self.patch_tokens.mean(dim=1) # [1, D]; Skip 4 reg tokens
        self.head_input = torch.cat((self.cls_token, self.mean_patch_tokens), dim=1)  # [1, 2*D]
        logits = self.dinov2.head(self.head_input)  # [1, num_classes]
        
        return logits



class Dinov2Block(nn.Module):
    def __init__(
        self, 
        original_block, 
        compute_graph: bool = False,
        libragrad: bool = False,
    ):
        super().__init__()
        self.compute_graph = compute_graph
        self.libragrad = libragrad
        self.intermediates = {}
        
        # Attention Block
        if self.libragrad:
            self.norm1 = FullGradLayerNorm(
                weight=original_block.norm1.weight,
                bias=original_block.norm1.bias,
                eps=original_block.norm1.eps,
            )
        else:
            self.norm1 = original_block.norm1
        self.attn = original_block.attn
        self.ls1 = original_block.ls1
        self.drop_path1 = original_block.drop_path1
        
        # MLP Block
        if self.libragrad:
            self.norm2 = FullGradLayerNorm(
                weight=original_block.norm2.weight,
                bias=original_block.norm2.bias,
                eps=original_block.norm2.eps,
            )
        else:
            self.norm2 = original_block.norm2
        self.mlp_fc1 = original_block.mlp.fc1
        if self.libragrad:
            self.mlp_act = FullGradGELU()
        else:
            self.mlp_act = original_block.mlp.act
        self.mlp_fc2 = original_block.mlp.fc2
        self.ls2 = original_block.ls2
        self.drop_path2 = original_block.drop_path2

        if self.compute_graph:
            dim = self.attn.qkv.in_features
            v_matrix = self.attn.qkv.weight[2*dim:3*dim, :].T  # [D, H*Dh] (head-wise concatenated)
            o_matrix = self.attn.proj.weight.T  # [H*Dh, D]

            self.intermediates["v_matrix"] = v_matrix  # [D, H*Dh]
            self.intermediates["o_matrix"] = o_matrix  # [H*Dh, D]
            # self.intermediates["o_bias"] = self.attn.proj.bias  # [D]

            self.intermediates["norm1_weight"] = self.norm1.weight  # [D]
            # self.intermediates["norm1_bias"] = self.norm1.bias      # [D]
            self.intermediates["norm2_weight"] = self.norm2.weight  # [D]
            # self.intermediates["norm2_bias"] = self.norm2.bias      # [D]
        
    def forward(self, x):
        if self.compute_graph:
            self.intermediates["residual_in"] = x  # [1, T, D]
        
        # 1. LayerNorm
        x_norm = self.norm1(x)
        
        if self.compute_graph:
            self.intermediates["attn_in"] = x_norm  # [1, T, D]
        
        # 2. Attention
        B, T, D_total = x_norm.shape
        num_heads = self.attn.num_heads
        head_dim = D_total // num_heads
        assert head_dim * num_heads == D_total, "Embedding dimension must be divisible by number of heads."
        
        qkv = self.attn.qkv(x_norm)  # [1, T, 3*D]

        q, k, v = qkv.chunk(3, dim=-1)  # [1, T, D]
        if self.compute_graph:
            self.intermediates["attn_v"] = v  # [1, Tk, D]
            
        q = q.reshape(B, T, num_heads, head_dim).permute(0, 2, 1, 3)  # [1, H, Tq, Dh]
        k = k.reshape(B, T, num_heads, head_dim).permute(0, 2, 1, 3)  # [1, H, Tk, Dh]
        v = v.reshape(B, T, num_heads, head_dim).permute(0, 2, 1, 3)  # [1, H, Tk, Dh]
        
        attn_scores = (q @ k.transpose(-2, -1)) * self.attn.scale  # [1, H, Tq, Tk]
        attn_weights = attn_scores.softmax(dim=-1)  # [1, H, Tq, Tk]
        if self.libragrad: attn_weights = attn_weights.detach() #! LibraGrad
        attn_out_heads = attn_weights @ v  # [1, H, Tq, Dh]
        
        attn_out = attn_out_heads.transpose(1, 2).reshape(B, T, D_total)  # [1, Tq, H, Dh] -> [1, Tq, D]
        attn_out = self.attn.proj(attn_out)  # [1, Tq, D]
        if self.compute_graph:
            self.intermediates["attn_map"] = attn_weights  # [1, H, Tq, Tk]
            self.intermediates["attn_av"] = attn_out_heads  # [1, H, Tq, Dh]
            self.intermediates["attn_o"] = attn_out  # [1, Tq, D]
        
        # 3. Residual Connection + DropPath
        attn_scaled = self.ls1(attn_out)  # omit if ls1 is Identity
        x = x + self.drop_path1(attn_scaled)
        if self.compute_graph:
            self.intermediates["residual_mid"] = x  # [1, T, D]
        
        # 4. LayerNorm
        x_norm2 = self.norm2(x)
        if self.compute_graph:
            self.intermediates["mlp_in"] = x_norm2  # [1, T, D]
        
        # 5. MLP
        mlp_hidden = self.mlp_fc1(x_norm2)
        mlp_act = self.mlp_act(mlp_hidden)
        mlp_out = self.mlp_fc2(mlp_act)
        if self.compute_graph:
            self.intermediates["mlp_mid"] = mlp_hidden  # [1, T, F]
            self.intermediates["mlp_out"] = mlp_out  # [1, T, D]
        
        # 6. Residual Connection + DropPath
        mlp_scaled = self.ls2(mlp_out)  # omit if ls2 is Identity
        x = x + self.drop_path2(mlp_scaled)
        if self.compute_graph:
            self.intermediates["residual_out"] = x  # [1, T, D]

        return x
    


class Dinov2Model(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, head_ckpt_path: str, device):
        super().__init__()
        self.backbone = backbone  # Preserve the entire backbone (including prepare_tokens_with_masks)
        self.blocks = backbone.blocks
        self.norm = backbone.norm
        self.embed_dim = backbone.embed_dim

        # classification head
        self.head = nn.Linear(self.embed_dim * 2, num_classes, bias=True)
        head_ckpt = torch.load(head_ckpt_path, map_location=device)
        self.head.load_state_dict(head_ckpt)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Preparation: cls_token, reg_token, pos_embed interpolation
        x = self.backbone.prepare_tokens_with_masks(x)

        # 2) Transformer blocks
        for blk in self.blocks:
            x = blk(x)
            
        # 3) Normalization
        x = self.norm(x)

        return {
            "cls_token": x[:, 0],                      # [B, D]
            "patch_tokens_mean": x[:, 1:].mean(dim=1)  # [B, D] (register token 포함 시 조정 가능)
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        concat_feat = torch.cat([feat["cls_token"], feat["patch_tokens_mean"]], dim=-1)  # [B, 2D]
        return self.head(concat_feat)
