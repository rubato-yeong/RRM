from types import SimpleNamespace
from typing import Optional, Union, List, Dict
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from utils.functions.functions import einsum
from utils.modules.modelgrad import ModelforGrad
from utils.modules.relevances import Relevances
from utils.modules.fullgrad import FullGradLayerNorm, FullGradGELU
from utils.modules.fullgrad import LinearGamma


class CascadedViTWithSAEforGrad(nn.Module, ModelforGrad):
    def __init__(
        self,
        vit_model: nn.Module,
        sae_models: dict,
        compute_graph: bool = False,
        relevance_norm: bool = False,
        softmax_correction: bool = False,
        relevance_debug: bool = False,
        libragrad: bool = False,
        gamma_rule: Union[float, None] = None,
        verbose: bool = False,
    ):
        """
        Args:
            vit_model: Vision Transformer model.
            sae_models: Dictionary of autoencoder models for each block.
            compute_graph: Flag to compute the graph for relevance propagation.
            relevance_norm: Not implemented.
            softmax_correction: Not implemented.
            relevance_debug: Flag to store intermediate relevance scores for debugging.
            libragrad: Flag to use libragrad for gradient computation.
                       (https://www.arxiv.org/abs/2411.16760)
            gamma_rule: Float value for gamma rule in relevance propagation. (MLP linear layers)
                        If None, do not apply gamma rule.
            verbose: Flag to print debug information.
        """
        super().__init__()
        if gamma_rule is not None:
            assert isinstance(gamma_rule, float), "gamma_rule must be a float."
            assert libragrad, "gamma_rule is only available when libragrad is True."

        # Initialize the models
        self.vit = vit_model
        self.sae_models = sae_models  # e.g., {0: sae0, 1: sae1, ...}
        
        # Compute graph flag
        self.compute_graph = compute_graph
        self.relevance_norm = relevance_norm
        self.softmax_correction = softmax_correction
        self.relevance_debug = relevance_debug
        self.libragrad = libragrad
        self.gamma = gamma_rule
        self.verbose = verbose
        self.relevances = Relevances(
            vit_model=vit_model,
            model="vit",
            relevance_debug=self.relevance_debug,
        )
        self.time = 0.0

        # Set vision transformer blocks
        self.vit_blocks = nn.ModuleList([
            ViTBlock(
                block, 
                self.compute_graph,
                self.libragrad,
                self.gamma,
            ) for block in self.vit.blocks
        ])
        self.layer_len = len(self.vit_blocks)

        # Post-block operations
        if self.gamma is not None:
            self.final_norm = FullGradLayerNorm(
                weight=self.vit.norm.weight,
                bias=self.vit.norm.bias,
                eps=self.vit.norm.eps,
            )
            self.final_head = LinearGamma(
                bias=self.vit.head.bias,
                weight=self.vit.head.weight,
                gamma=self.gamma,
            )
        elif self.libragrad:
            self.final_norm = FullGradLayerNorm(
                weight=self.vit.norm.weight,
                bias=self.vit.norm.bias,
                eps=self.vit.norm.eps,
            )
            self.final_head = self.vit.head
        else:
            self.final_norm = self.vit.norm
            self.final_head = self.vit.head
            
    
    def forward(
        self, 
        x, 
        mask_dict=None, 
        mean_feature_acts=None, 
        mean_error_acts=None,
        true_error_acts=None,
        mean_neuron_acts=None,
        fine_masking=False,
    ):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor image of shape [B, C, H, W]. Batch size should be 1.
            mask_dict: Dictionary of masks for each block SAE features.
        """
        assert x.shape[0] == 1, "the model supports only a single input."

        # Preprocess input
        x = self.vit.patch_embed(x)  # [B, N, D]
        B, N, D = x.shape
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D]
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        self.n_tokens = N+1 # equal to T
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
                        fine_masking=fine_masking,
                    )
                    sae_out = sae_out_dict["sae_out_with_mask"].reshape(B, T, D)
                    sae_error = sae_out_dict["sae_error"].reshape(B, T, D)
                    x = sae_out + sae_error  # [B, T, D]
                
        # Classification head
        # self.cls_token_before_ln = x[:, 0, :]  # [1, D]
        x = self.final_norm(x)  # [B, T, D]
        self.cls_token_final = x[:, 0, :]  # [1, D]
        logits = self.final_head(self.cls_token_final) # [1, num_classes]

        return logits
    



class CascadedViTWithSAEforViz(nn.Module):
    def __init__(
        self,
        vit_model: nn.Module,
        sae_model: Union[nn.Module, None] = None,
        layer_idx: Union[int, None] = None,
    ):
        """
        Args:
            vit_model: Vision Transformer model.
            sae_model: SAE model
            layer_idx: Index of the SAE layer to be used for visualization.
            (If None, logits from the last layer will be used.)
        """
        super().__init__()

        # Initialize the models
        self.vit = vit_model
        self.sae_model = sae_model

        # Set vision transformer blocks
        self.vit_blocks = nn.ModuleList([
            ViTBlock(block) for block in self.vit.blocks
        ])
        self.layer_len = len(self.vit_blocks)
        self.layer_idx = layer_idx
    

    def forward(self, x, feat_idx):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor image of shape [B, C, H, W]. Batch size should be 1.
            feat_idx: Index of the feature to be visualized.

        Returns:
            objective: Objective value for the given feature. # [B]
        """

        # Preprocess input
        x = self.vit.patch_embed(x)  # [B, N, D]
        B, N, D = x.shape
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D]
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        self.n_tokens = N+1 # equal to T
        mask = None
        
        # Pass through the transformer blocks
        for i, block in enumerate(self.vit_blocks):
            x = block(x)  # [B, T, D]
            
            if i == self.layer_idx:
                B, T, D = x.shape
                x_flat = x.reshape(1, B * T, D)  # [1, B*T, D]
                
                sae_out_dict = self.sae_model(x_flat, mask_features=None, compute_graph=False)
                objective = sae_out_dict["sae_out"].reshape(B, T, D)[:, :, feat_idx].mean(dim=1)

                return objective
        
        # Classification head
        self.cls_token_before_ln = x[:, 0, :]  # [1, D]
        x = self.vit.norm(x)  # [B, T, D]
        self.cls_token_final = x[:, 0, :]  # [1, D]
        logits = self.vit.head(self.cls_token_final) # [1, num_classes]
        objective = logits[:, feat_idx]

        return objective
    


class CascadedViTWithSAEforAct(nn.Module):
    def __init__(
        self,
        vit_model: nn.Module,
        sae_models: dict,
    ):
        """
        Args:
            vit_model: Vision Transformer model.
            sae_models: Dictionary of autoencoder models for each block.
        """
        super().__init__()

        # Initialize the models
        self.vit = vit_model
        self.sae_models = sae_models  # e.g., {0: sae0, 1: sae1, ...}

        # Set vision transformer blocks
        self.vit_blocks = nn.ModuleList([
            ViTBlock(
                block, 
                compute_graph=False,
                libragrad=False,
            ) for block in self.vit.blocks
        ])
        self.layer_len = len(self.vit_blocks)

        # Post-block operations
        self.final_norm = self.vit.norm

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
        # assert x.shape[0] == 1, "the model supports only a single input."

        # Preprocess input
        x = self.vit.patch_embed(x)  # [B, N, D]
        B, N, D = x.shape
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D]
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        self.n_tokens = N+1 # equal to T
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
        self.cls_token_before_ln = x[:, 0, :]  # [1, D]
        x = self.final_norm(x)  # [B, T, D]
        self.cls_token_final = x[:, 0, :]  # [1, D]
        logits = self.vit.head(self.cls_token_final) # [1, num_classes]

        return logits
    
    

class ViTBlock(nn.Module):
    def __init__(
        self, 
        original_block, 
        compute_graph: bool = False,
        libragrad: bool = False,
        gamma: Union[float, None] = None,
    ):
        super().__init__()
        self.compute_graph = compute_graph
        self.libragrad = libragrad
        self.gamma = gamma
        self.intermediates = {}
        
        # Attention Norm Block
        if self.libragrad:
            self.norm1 = FullGradLayerNorm(
                weight=original_block.norm1.weight,
                bias=original_block.norm1.bias,
                eps=original_block.norm1.eps,
            )
        else:
            self.norm1 = original_block.norm1
        
        # Attention Block
        self.attn = original_block.attn
        self.attn_qkv = original_block.attn.qkv
        self.attn_proj = original_block.attn.proj
        self.ls1 = original_block.ls1
        self.drop_path1 = original_block.drop_path1
        
        # MLP Norm Block
        if self.libragrad:
            self.norm2 = FullGradLayerNorm(
                weight=original_block.norm2.weight,
                bias=original_block.norm2.bias,
                eps=original_block.norm2.eps,
            )
        else:
            self.norm2 = original_block.norm2
            
        # MLP Block
        if self.gamma is not None:
            self.mlp_fc1 = LinearGamma(
                weight=original_block.mlp.fc1.weight,
                bias=original_block.mlp.fc1.bias,
                gamma=self.gamma,
            )
            self.mlp_fc2 = LinearGamma(
                weight=original_block.mlp.fc2.weight,
                bias=original_block.mlp.fc2.bias,
                gamma=self.gamma,
            )
        else:
            self.mlp_fc1 = original_block.mlp.fc1
            self.mlp_fc2 = original_block.mlp.fc2
        if self.libragrad:
            self.mlp_act = FullGradGELU()
        else:
            self.mlp_act = original_block.mlp.act
        self.ls2 = original_block.ls2
        self.drop_path2 = original_block.drop_path2

        if self.compute_graph:
            embed_dim = self.attn.head_dim * self.attn.num_heads
            v_matrix = self.attn.qkv.weight[2 * embed_dim : 3 * embed_dim, :].T  # [D, H*Dh] (head-wise concatenated)
            o_matrix = self.attn.proj.weight.T  # [H*Dh, D]
            # v_matrix_per_head = v_matrix.reshape(self.attn.num_heads, self.attn.head_dim, -1).permute(0, 2, 1)  # [H, D, Dh]
            # o_matrix_per_head = o_matrix.reshape(self.attn.num_heads, self.attn.head_dim, -1)  # [H, Dh, D]

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
        
        qkv = self.attn_qkv(x_norm)  # [1, T, 3*D]

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
        attn_out = self.attn_proj(attn_out)  # [1, Tq, D]
        if self.compute_graph:
            self.intermediates["attn_map"] = attn_weights  # [1, H, Tq, Tk]
            self.intermediates["attn_av"] = attn_out_heads  # [1, H, Tq, Dh]
            self.intermediates["attn_o"] = attn_out  # [1, Tq, D]
        
        # 3. Residual Connection + DropPath
        x = x + self.drop_path1(attn_out)
        if self.compute_graph:
            self.intermediates["residual_mid"] = x  # [1, T, D]
        x = self.ls1(x)  # omit if ls1 is Identity
        
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
        x = x + self.drop_path2(mlp_out)
        if self.compute_graph:
            self.intermediates["residual_out"] = x  # [1, T, D]
        x = self.ls2(x)  # omit if ls2 is Identity

        return x