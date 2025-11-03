from types import SimpleNamespace

import torch


class Relevances:
    """
    Relevances for each block in the ViT model.
    
    Attributes:
        block: List of SimpleNamespace objects, each containing the relevance information for a block.
            block_out: Relevance for the output of the block (after mlp).
            block_mid: Relevance for the intermediate output of the block (after attn, before mlp).
            block_in: Relevance for the input of the block (before attn).
            sae: Relevance for the SAE features (before block_in).
        head: None
    """
    def __init__(self, vit_model, model="vit", relevance_debug=False):
        
        try:
            if model in ["vit", "dinov2"]: num_blocks = len(vit_model.blocks)
            elif model in ["clip_vit"]: num_blocks = len(vit_model)
            else: raise ValueError("Model not supported.")
        except:
            raise AttributeError("The vision model must possess a 'blocks' attribute.")
        
        if not relevance_debug:
            self.block = [SimpleNamespace(
                sae=None,
                error=None,
                sae_grad=None,
                error_grad=None,
                edge=None,
                ) for _ in range(num_blocks)]
        else:
            self.block = [SimpleNamespace(
                block_out=None, 
                block_mid=None, 
                block_in=None, 
                sae=None,
                error=None,
                sae_grad=None,
                error_grad=None,
                edge=None,
                ) for _ in range(num_blocks)]
        self.head = None 


class NormalizedRelevances:
    def __init__(self, layer_num):
        self.layer_num = layer_num
        self.block = [SimpleNamespace(
            node=None, # [F+1]
            edge=None, # [F+1, F+1]
        ) for _ in range(layer_num)]


def get_normalized_relevance(relevances: Relevances, negative_relevance: bool = False, flip_relevance: bool = False):

    normalized_relevances = NormalizedRelevances(layer_num=len(relevances.block))

    for i, block in enumerate(relevances.block):

        # Node relevance
        if getattr(block, "sae") is not None:
            if getattr(block, "error") is not None:
                node_relevance = torch.cat([block.sae.sum(dim=0), block.error.sum().unsqueeze(0)], dim=0) # [F+1]
            else:
                node_relevance = block.sae.sum(dim=0) # [F]
            if flip_relevance:
                node_relevance = -node_relevance
            if negative_relevance:
                node_relevance = torch.abs(node_relevance)
            else:
                node_relevance = torch.clamp(node_relevance, min=0)
            norm = node_relevance.sum()
            node_relevance = node_relevance / norm
            normalized_relevances.block[i].node = node_relevance

        # Edge relevance   
        if getattr(block, "edge") is not None:
            edge_relevance = block.edge
            if flip_relevance:
                edge_relevance = -edge_relevance
            if negative_relevance:
                edge_relevance = torch.abs(edge_relevance) / norm
            else:
                edge_relevance = edge_relevance / norm
            normalized_relevances.block[i].edge = edge_relevance
    
    return normalized_relevances


def get_fine_normalized_relevance(relevances: Relevances, negative_relevance: bool = False):

    normalized_relevances = NormalizedRelevances(layer_num=len(relevances.block))

    for i, block in enumerate(relevances.block):

        # Node relevance
        if getattr(block, "sae") is not None:
            if getattr(block, "error") is not None:
                node_relevance = torch.cat([block.sae.flatten(), block.error.sum().unsqueeze(0)], dim=0) # [T*F+1]
            else:
                node_relevance = block.sae.flatten() # [T*F]
            if negative_relevance:
                node_relevance = torch.abs(node_relevance)
            else:
                node_relevance = torch.clamp(node_relevance, min=0)
            norm = node_relevance.sum()
            node_relevance = node_relevance / norm
            normalized_relevances.block[i].node = node_relevance

        # Edge relevance   
        if getattr(block, "edge") is not None:
            if negative_relevance:
                edge_relevance = torch.abs(block.edge) / norm
            else:
                edge_relevance = block.edge / norm
            normalized_relevances.block[i].edge = edge_relevance
    
    return normalized_relevances