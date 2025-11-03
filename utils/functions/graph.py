from typing import Union

import torch
import torch.nn as nn

from utils.functions.attribution import get_relevance
from utils.modules.relevances import NormalizedRelevances


def get_simplified_graph_by_node(
    normalized_relevances: NormalizedRelevances,
    topn: Union[None, float] = None,
    topk: Union[None, float] = None,
    threshold: Union[None, float] = None,
    max_node_num: Union[None, int] = None,
    return_values: bool = False,
):
    """
    Get simplified graph from normalized relevances (based on node relevance).
    Args:
        normalized_relevances (NormalizedRelevances): Normalized relevances for each layer.
        topn (float, optional): Percent of top features to select. (definite) Defaults to None.
        topk (float, optional): Percent of top features to select. (relative) Defaults to None.
        threshold (float, optional): Threshold for edge relevance. Defaults to None.
        max_node_num (int, optional): Maximum number of nodes to select. Defaults to None.
        return_values (bool, optional): Whether to return values. Defaults to False.
    Returns:
        node_indices (dict: {int: list[int]}): Dictionary of node indices for each layer.
        node_relevances (dict: {int: torch.Tensor}): Dictionary of node relevance for each layer. (if return_values is True)
        edge_relevances (dict: {int: torch.Tensor}): Dictionary of edge relevance for each layer. (if return_values is True)
    """

    assert topn is not None or topk is not None or threshold is not None, "Either topn, topk or threshold must be provided."
    if topn is not None:
        assert max_node_num is not None, "max_node_num must be provided when topn is used."

    node_indices = {}
    node_relevances = {}
    edge_relevances = {}

    if topn is not None or topk is not None: # select topk features per layer
        for i in range(1, normalized_relevances.layer_num):
            if topn is not None:
                indices_num = int(topn * max_node_num)
            elif topk is not None:
                indices_num = int(topk * normalized_relevances.block[i].node.shape[0])
            node_indices[i] = torch.argsort(normalized_relevances.block[i].node, descending=True)[:indices_num]
            if return_values:
                node_relevances[i] = normalized_relevances.block[i].node[node_indices[i]] \
                                    / torch.clamp(normalized_relevances.block[i].node, min=0).sum()
                edge_relevances[i] = normalized_relevances.block[i].edge[node_indices[i]] \
                                    / torch.clamp(normalized_relevances.block[i].node, min=0).sum()
    
    elif threshold is not None: # select features with (sum of node relevance) > threshold
        for i in range(1, normalized_relevances.layer_num):
            node_relevance_sum, node_idx = 0.0, 0
            sorted_node_indices = torch.argsort(normalized_relevances.block[i].node, descending=True)
            while node_relevance_sum < threshold:
                try:
                    node_relevance_sum += normalized_relevances.block[i].node[sorted_node_indices[node_idx]]
                    node_idx += 1
                except:
                    print(f"Layer {i} has no node relevance above threshold.")
                    break
            node_indices[i] = sorted_node_indices[:node_idx]
            if return_values:
                node_relevances[i] = normalized_relevances.block[i].node[node_indices[i]] \
                                    / torch.clamp(normalized_relevances.block[i].node, min=0).sum()
                edge_relevances[i] = normalized_relevances.block[i].edge[node_indices[i]] \
                                    / torch.clamp(normalized_relevances.block[i].node, min=0).sum()
    
    return (node_indices, node_relevances, edge_relevances) if return_values else node_indices


def get_simplified_graph_by_edge(
    normalized_relevances: NormalizedRelevances,
    topn: Union[None, float] = None,
    topk: Union[None, float] = None,
    threshold: Union[None, float] = None,
    max_node_num: Union[None, int] = None,
    return_values: bool = False,
):
    """
    Get simplified graph from normalized relevances (based on edge relevance).
    Args:
        normalized_relevances (NormalizedRelevances): Normalized relevances for each layer.
        topn (float, optional): Percent of top features to select. (definite) Defaults to None.
        topk (float, optional): Percent of top features to select. (relative) Defaults to None.
        threshold (float, optional): Threshold for edge relevance. Defaults to None.
        max_node_num (int, optional): Maximum number of nodes to select. Defaults to None.
        return_values (bool, optional): Whether to return values. Defaults to False.
    Returns:
        node_indices (dict: {int: list[int]}): Dictionary of node indices for each layer.
        node_relevances (dict: {int: torch.Tensor}): Dictionary of node relevance for each layer. (if return_values is True)
        edge_relevances (dict: {int: torch.Tensor}): Dictionary of edge relevance for each layer. (if return_values is True)
    """

    assert topn is not None or topk is not None or threshold is not None, "Either topn, topk or threshold must be provided."
    if topn is not None:
        assert max_node_num is not None, "max_node_num must be provided when topn is used."

    node_indices = {}
    node_relevances = {}
    edge_relevances = {}

    if topn is not None or topk is not None: # select topk features per layer
        for i in range(normalized_relevances.layer_num-1, 0, -1):
            if i == normalized_relevances.layer_num-1:
                if topn is not None:
                    indices_num = int(topn * max_node_num)
                elif topk is not None:
                    indices_num = int(topk * normalized_relevances.block[i].node.shape[0])
                node_indices[i] = torch.argsort(normalized_relevances.block[i].node, descending=True)[:indices_num]
                if return_values:
                    node_relevances[i] = normalized_relevances.block[i].node[node_indices[i]] \
                                        / torch.clamp(normalized_relevances.block[i].node, min=0).sum()
                    edge_relevances[i] = node_relevances[i].unsqueeze(-1)     
            else:
                if topn is not None:
                    indices_num = int(topn * max_node_num)
                elif topk is not None:
                    indices_num = int(topk * normalized_relevances.block[i].node.shape[0])
                normalized_edge = normalized_relevances.block[i].edge[:, node_indices[i+1]] \
                    / torch.clamp(normalized_relevances.block[i].edge[:, node_indices[i+1]].sum(dim=-1), min=0).sum()  
                node_indices[i] = torch.argsort(normalized_edge.sum(dim=-1), descending=True)[:indices_num]
                if return_values:
                    node_relevances[i] = normalized_edge[node_indices[i]].sum(dim=-1)
                    edge_relevances[i] = normalized_edge[node_indices[i]]
        
    elif threshold is not None: # select features with (sum of node relevance) > threshold
        for i in range(normalized_relevances.layer_num-1, 0, -1):
            if i == normalized_relevances.layer_num-1:
                node_relevance_sum, node_idx = 0.0, 0
                sorted_node_indices = torch.argsort(normalized_relevances.block[i].node, descending=True)
                while node_relevance_sum < threshold:
                    node_relevance_sum += normalized_relevances.block[i].node[sorted_node_indices[node_idx]]
                    node_idx += 1
                node_indices[i] = sorted_node_indices[:node_idx]
                if return_values:
                    node_relevances[i] = normalized_relevances.block[i].node[node_indices[i]] \
                                        / torch.clamp(normalized_relevances.block[i].node, min=0).sum()
                    edge_relevances[i] = node_relevances[i].unsqueeze(-1)
            else:
                node_relevance_sum, node_idx = 0.0, 0
                normalized_edge = normalized_relevances.block[i].edge[:, node_indices[i+1]] \
                    / torch.clamp(normalized_relevances.block[i].edge[:, node_indices[i+1]].sum(dim=-1), min=0).sum() 
                sorted_node_indices = torch.argsort(normalized_edge.sum(dim=-1), descending=True)
                while node_relevance_sum < threshold:
                    try:
                        node_relevance_sum += normalized_edge.sum(dim=-1)[sorted_node_indices[node_idx]]
                        node_idx += 1
                    except:
                        print(f"Layer {i} has no node relevance above threshold.")
                        break
                node_indices[i] = sorted_node_indices[:node_idx]
                if return_values:
                    node_relevances[i] = normalized_edge[node_indices[i]].sum(dim=-1)
                    edge_relevances[i] = normalized_edge[node_indices[i]]
                    
    return (node_indices, node_relevances, edge_relevances) if return_values else node_indices