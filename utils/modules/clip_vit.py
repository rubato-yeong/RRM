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
from utils.modules.fullgrad import FullGradLayerNorm, FullGradQuickGELU, FullGradNormalize



class CascadedClipViTWithSAEforGrad(nn.Module, ModelforGrad):
    def __init__(
        self,
        vit_model: nn.Module,
        sae_models: dict,
        text_features: torch.Tensor,
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
            vit_model: Vision Transformer model.
            sae_models: Dictionary of autoencoder models for each block.
            text_features: Text features (head) for CLIP model.
            compute_graph: Flag to compute the graph for relevance propagation.
            relevance_norm: Not implemented.
            softmax_correction: Not implemented.
                Citation: From Clustering to Cluster Explanations via Neural Networks.
                          https://arxiv.org/abs/1906.07633
            relevance_debug: Flag to store intermediate relevance scores for debugging.
            libragrad: Flag to use libragrad for gradient computation.
                       (https://www.arxiv.org/abs/2411.16760)
            verbose: Flag to print debug information.
        """
        super().__init__()

        # Initialize the models
        self.clip = vit_model.vision_model
        self.head = vit_model.visual_projection
        self.text_features = text_features
        self.sae_models = sae_models  # e.g., {0: sae0, 1: sae1, ...}
        
        # Compute graph flag
        self.compute_graph = compute_graph
        self.relevance_norm = relevance_norm
        self.softmax_correction = softmax_correction
        self.relevance_debug = relevance_debug
        self.libragrad = libragrad
        self.verbose = verbose
        self.relevances = Relevances(
            vit_model=self.clip.encoder.layers,
            model="clip_vit",
            relevance_debug=self.relevance_debug,
        )
        self.time = 0.0
        
        # Set vision transformer blocks
        self.vit_blocks = nn.ModuleList([
            ClipViTBlock(
                block, 
                self.compute_graph,
                self.libragrad,
            ) for block in self.clip.encoder.layers
        ])
        self.layer_len = len(self.vit_blocks)
        
        # Pre/Post-block operations
        if self.libragrad:
            self.pre_layernorm = FullGradLayerNorm(
                weight=self.clip.pre_layrnorm.weight,
                bias=self.clip.pre_layrnorm.bias,
                eps=self.clip.pre_layrnorm.eps,
            )
            self.post_layernorm = FullGradLayerNorm(
                weight=self.clip.post_layernorm.weight,
                bias=self.clip.post_layernorm.bias,
                eps=self.clip.post_layernorm.eps,
            )
            self.normalize =  FullGradNormalize()
        else:
            self.pre_layernorm = self.clip.pre_layrnorm
            self.post_layernorm = self.clip.post_layernorm
    

    def forward(
        self, 
        x, 
        mask_dict=None, 
        mean_feature_acts=None, 
        mean_error_acts=None,
        true_error_acts=None,
        mean_neuron_acts=None,
        mask_ratio=1.0,
    ):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor image of shape [B, C, H, W]. Batch size should be 1.
            mask_dict: Dictionary of masks for each block SAE features.
        """
        assert x.shape[0] == 1, "CascadedViTWithSAE model supports only a single input."

        # Preprocess input
        x = self.clip.embeddings(x)  # [B, N, D]
        x = self.pre_layernorm(x)  # CLIP applies pre-layernorm after embeddings
        B, N, D = x.shape
        self.n_tokens = N  # Remember the number of tokens (T)
        mask = None

        # Pass through the transformer blocks
        for i, block in enumerate(self.vit_blocks):
            x = block(x)  # (B, T, D)

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
                    # sae_out = sae_out_dict["sae_out"].reshape(B, T, D)
                    # sae_error = sae_out_dict["sae_error"].reshape(B, T, D)
                    # x = sae_out + sae_error # (disturb accurate computation)
                
                elif mean_feature_acts is not None and mean_error_acts is not None:
                    sae_out_dict = self.sae_models[i](
                        x_flat,
                        mask_features=mask_dict[i],
                        compute_graph=self.compute_graph,
                        mean_feature_acts=mean_feature_acts[i],
                        mean_error_acts=mean_error_acts[i],
                        true_error_acts=true_error_acts[i] if true_error_acts is not None else None,
                        mask_ratio=mask_ratio,
                    )
                    sae_out = sae_out_dict["sae_out_with_mask"].reshape(B, T, D)
                    sae_error = sae_out_dict["sae_error"].reshape(B, T, D)
                    x = sae_out + sae_error  # [B, T, D]

        # After encoder blocks
        x = self.post_layernorm(x)  # CLIP applies post-layernorm after all blocks
        self.cls_token_final = x[:, 0, :]  # [B, D]
        image_features = self.head(self.cls_token_final)  # [B, D_text]
        if self.libragrad:
            image_features = self.normalize(image_features, dim=-1)
        else:
            image_features = F.normalize(image_features, dim=-1)
        logits = image_features @ self.text_features.T
        
        return logits
    
    
    
    
class CascadedClipViTWithSAEforAct(nn.Module):
    def __init__(
        self,
        vit_model: nn.Module,
        sae_models: dict,
        text_features: torch.Tensor,
    ):
        """
        Args:
            vit_model: Vision Transformer model.
            sae_models: Dictionary of autoencoder models for each block.
            text_features: Text features (head) for CLIP model.
        """
        super().__init__()

        # Initialize the models
        self.clip = vit_model.vision_model
        self.head = vit_model.visual_projection
        self.text_features = text_features
        self.sae_models = sae_models  # e.g., {0: sae0, 1: sae1, ...}
        
        # Set vision transformer blocks
        self.vit_blocks = nn.ModuleList([
            ClipViTBlock(
                block, 
                compute_graph=False,
                libragrad=False,
            ) for block in self.clip.encoder.layers
        ])
        self.layer_len = len(self.vit_blocks)
        
        # Pre/Post-block operations
        self.pre_layernorm = self.clip.pre_layrnorm
        self.post_layernorm = self.clip.post_layernorm

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
        # assert x.shape[0] == 1, "CascadedViTWithSAE model supports only a single input."

        # Preprocess input
        x = self.clip.embeddings(x)  # [B, N, D]
        x = self.pre_layernorm(x)  # CLIP applies pre-layernorm after embeddings
        B, N, D = x.shape
        self.n_tokens = N  # Remember the number of tokens (T)
        mask = None

        # Pass through the transformer blocks
        for i, block in enumerate(self.vit_blocks):
            x = block(x)  # (B, T, D)

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

        # After encoder blocks
        x = self.post_layernorm(x)  # CLIP applies post-layernorm after all blocks
        self.cls_token_final = x[:, 0, :]  # [B, D]
        image_features = self.head(self.cls_token_final)  # [B, D_text]
        image_features = F.normalize(image_features, dim=-1)
        logits = image_features @ self.text_features.T
        
        return logits    
    
    
class ClipViTBlock(nn.Module):
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
                weight=original_block.layer_norm1.weight,
                bias=original_block.layer_norm1.bias,
                eps=original_block.layer_norm1.eps,
            )
        else:
            self.norm1 = original_block.layer_norm1
        self.attn = original_block.self_attn

        # MLP Block
        if self.libragrad:
            self.norm2 = FullGradLayerNorm(
                weight=original_block.layer_norm2.weight,
                bias=original_block.layer_norm2.bias,
                eps=original_block.layer_norm2.eps,
            )
        else:
            self.norm2 = original_block.layer_norm2
        self.mlp_fc1 = original_block.mlp.fc1
        if self.libragrad:
            self.mlp_act = FullGradQuickGELU()
        else:
            self.mlp_act = original_block.mlp.activation_fn
        self.mlp_fc2 = original_block.mlp.fc2
        
        if self.compute_graph:
            v_matrix = self.attn.v_proj.weight.T  # [D, H*Dh] (head-wise concatenated)
            o_matrix = self.attn.out_proj.weight.T  # [H*Dh, D]

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
        head_dim = self.attn.head_dim
        assert head_dim * num_heads == D_total, "Embedding dimension must be divisible by number of heads."

        q = self.attn.q_proj(x_norm)
        k = self.attn.k_proj(x_norm)
        v = self.attn.v_proj(x_norm)
        if self.compute_graph:
            self.intermediates["attn_v"] = v  # [1, Tk, D]

        q = q.reshape(B, T, num_heads, head_dim).permute(0, 2, 1, 3)  # [1, H, Tq, Dh]
        k = k.reshape(B, T, num_heads, head_dim).permute(0, 2, 1, 3)  # [1, H, Tk, Dh]
        v = v.reshape(B, T, num_heads, head_dim).permute(0, 2, 1, 3)  # [1, H, Tk, Dh]

        attn_scores = (q @ k.transpose(-2, -1)) * self.attn.scale  # [1, H, T, T]
        attn_weights = attn_scores.softmax(dim=-1)  # [1, H, Tq, Tk]
        if self.libragrad: attn_weights = attn_weights.detach() #! LibraGrad
        attn_out_heads = attn_weights @ v  # [1, H, T, Dh]

        attn_out = attn_out_heads.permute(0, 2, 1, 3).reshape(B, T, D_total)  # [1, Tq, H, Dh] -> [1, Tq, D]
        attn_out = self.attn.out_proj(attn_out)  # [1, Tq, D]
        if self.compute_graph:
            self.intermediates["attn_map"] = attn_weights  # [1, H, Tq, Tk]
            self.intermediates["attn_av"] = attn_out_heads  # [1, H, Tq, Dh]
            self.intermediates["attn_o"] = attn_out  # [1, Tq, D]
        
        x = x + attn_out  # Residual
        if self.compute_graph:
            self.intermediates["residual_mid"] = x  # [1, T, D]

        # 3. LayerNorm
        x_norm2 = self.norm2(x)
        if self.compute_graph:
            self.intermediates["mlp_in"] = x_norm2  # [1, T, D]

        # 4. MLP
        mlp_hidden = self.mlp_fc1(x_norm2)
        mlp_act = self.mlp_act(mlp_hidden)
        mlp_out = self.mlp_fc2(mlp_act)
        if self.compute_graph:
            self.intermediates["mlp_mid"] = mlp_hidden  # [1, T, F]
            self.intermediates["mlp_out"] = mlp_out  # [1, T, D]

        x = x + mlp_out  # Residual
        if self.compute_graph:
            self.intermediates["residual_out"] = x  # [1, T, D]

        return x
    
    
class ClipViTModel(nn.Module):
    def __init__(
        self,
        vit_model: nn.Module,
        text_features: torch.Tensor,
    ):
        """
        Args:
            vit_model: Vision Transformer model.
            sae_models: Dictionary of autoencoder models for each block.
            text_features: Text features (head) for CLIP model.
        """
        super().__init__()

        # Initialize the models
        self.clip = vit_model.vision_model
        head_weight = vit_model.visual_projection.weight.T @ text_features.T
        head_in_features = head_weight.shape[0]
        head_out_features = head_weight.shape[1]
        self.head = nn.Linear(head_in_features, head_out_features, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(head_weight.T)
        
        # Set vision transformer blocks
        self.vit_blocks = nn.ModuleList([
            ClipViTBlock(
                block, 
                compute_graph=False,
                libragrad=False,
            ) for block in self.clip.encoder.layers
        ])
        self.layer_len = len(self.vit_blocks)
        
        # Pre/Post-block operations
        self.pre_layernorm = self.clip.pre_layrnorm
        self.post_layernorm = self.clip.post_layernorm
    
    def forward(self, x, mask_dict=None, type="mean"):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor image of shape [B, C, H, W]. Batch size should be 1.
            mask_dict: Dictionary of masks for each block SAE features.
            type: Type of activation to compute. (mean or median)
        """

        # Preprocess input
        x = self.clip.embeddings(x)  # [B, N, D]
        x = self.pre_layernorm(x)  # CLIP applies pre-layernorm after embeddings
        B, N, D = x.shape
        self.n_tokens = N  # Remember the number of tokens (T)

        # Pass through the transformer blocks
        for i, block in enumerate(self.vit_blocks):
            x = block(x)  # (B, T, D)

        # After encoder blocks
        x = self.post_layernorm(x)  # CLIP applies post-layernorm after all blocks
        cls_token_final = x[:, 0, :]  # [B, D]
        logits = self.head(cls_token_final)  # [B, D_text]
        logits = F.normalize(logits, dim=-1)
        
        return logits    
    