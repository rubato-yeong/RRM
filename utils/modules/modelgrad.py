import time
from typing import Optional, Union, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.functions.functions import einsum


class ModelforGrad():
    
    def get_true_error_acts(self):
        """
        Get true error activations for the nodes in the graph.
        
        Returns:
            true_error_acts: Dict of true error activations for the nodes in the graph.
        """
        
        true_error_acts = {}
        for i in range(0, self.layer_len-1):
            if i not in self.sae_models.keys(): 
                continue
            
            # Get true error activations
            true_error_acts[i] = (self.vit_blocks[i+1].intermediates["residual_in"].squeeze() - self.sae_models[i].intermediates['x_hat']).detach()
        
        return true_error_acts
    
    
    def clean_grad(self):
        """
        Clean gradients for the nodes in the graph.
        """
        
        self.zero_grad()
        for i in range(self.layer_len-1, 0, -1):
            self.vit_blocks[i].intermediates["residual_in"].grad = None
        self.time = 0.0
        
    
    def get_grad(self, label, logits, adjust_logit=False):
        """
        Get gradients for the nodes in the graph.
        
        Args:
            label: Target label for the classification task.
            logits: Output logits from the model.
            adjust_logit: Flag to adjust the logits for the target label.
                For more flexible usage, it can be a list of logits to adjust.
        """
        
        # Retain gradients
        for i in range(self.layer_len-1, 0, -1):
            self.vit_blocks[i].intermediates["residual_in"].retain_grad()
            
        # Compute gradients
        try:
            logit = logits[:, label] - logits[:, adjust_logit].mean()
        except:
            if adjust_logit is True: logit = logits[:, label] - logits.mean()
            else: logit = logits[:, label]
        logit.backward()
        
        # Collect gradients
        grads = {}
        for i in range(self.layer_len-1, 0, -1):
            # Skip if no SAE model is present
            if i-1 not in self.sae_models.keys(): 
                return
            
            ##### Preparation #####
            W_dec = self.sae_models[i-1].W_dec  # [D*R, D]
            x_std = self.sae_models[i-1].intermediates['x_std']  # [T, 1]
            
            ##### Attribution #####
            grad = self.vit_blocks[i].intermediates["residual_in"].grad.squeeze().detach() # [T, D]
            attr = x_std * grad  # [T, D]
            attr = attr @ W_dec.T  # [T, D*R]
            grads[i] = [attr, grad]  # sae, error gradients

        return grads
    
    
    def compute_node_attributions(
            self,
            grads: dict,
            mean_feature_acts: dict = None,
            mean_error_acts: dict = None,
        ):
        """
        Compute attributions for the nodes in the graph.
        
        Args:
            grads: Gradients for the nodes (sae and error).
            mean_feature_acts: Mean feature activation (dictionary) for each layer.
                               If not provided, it assumes all activations are 0.
            mean_error_acts: Mean error activation (dictionary) for each layer.
                             If not provided, it assumes all activations are 0.
        """
        
        # Compute Attribution
        if self.verbose:
            print("Computing Node Attributions...", end=" ")
            start = time.time()
        for i in range(self.layer_len-1, 0, -1):
            self._compute_node_attr(
                layer_num=i, 
                grad=grads[i],
                mean_feature_act=mean_feature_acts[i-1] if mean_feature_acts is not None else None,
                mean_error_act=mean_error_acts[i-1] if mean_error_acts is not None else None,
            )
        if self.verbose:
            end = time.time()
            print(f"Node Attribution Time: {end-start:.4f}s")
            self.time += end-start
            
    
    @torch.no_grad()
    def _compute_node_attr(
        self, 
        layer_num: int,
        grad: tuple,
        mean_feature_act: torch.Tensor = None,
        mean_error_act: torch.Tensor = None,
    ):
        """
        Compute the attribution for the nodes in the graph.
        
        Args:
            layer_num: Layer number for which to compute the attribution.
            grad: Gradients for the nodes (sae and error).
            mean_feature_act: Mean feature activation for the layer. (Optional)
            mean_error_act: Mean error activation for the layer. (Optional)
        """
        
        # Skip if no SAE model is present
        if layer_num-1 not in self.sae_models.keys(): 
            return
        
        ##### Preparation #####
        z = self.sae_models[layer_num-1].intermediates['z']  # [T, D*R]
        error = self.vit_blocks[layer_num].intermediates['residual_in'].squeeze() \
                - self.sae_models[layer_num-1].intermediates['x_hat']  # [T, D]
        sae_grad, error_grad = grad  # [T, D*R], [T, D]

        if mean_feature_act is not None:
            self.relevances.block[layer_num].sae = (sae_grad * (z - mean_feature_act)).detach()  # [T, D*R]
        else:
            self.relevances.block[layer_num].sae = (sae_grad * z).detach()  # [T, D*R]
        if mean_error_act is not None:
            self.relevances.block[layer_num].error = (error_grad * (error - mean_error_act)).detach()  # [T, D]
        else:
            self.relevances.block[layer_num].error = (error_grad * error).detach()  # [T, D]
        
        return


    def compute_edge_attributions(
            self, 
            grads: dict,
            mean_feature_acts: dict = None,
            mean_error_acts: dict = None,
        ):
        """
        Compute edge attributions for the nodes in the graph.
        
        Args:
            grads: Gradients for the nodes (sae and error).
            mean_feature_acts: Mean feature activation (dictionary) for each layer.
                               If not provided, it assumes all activations are 0.
            mean_error_acts: Mean error activation (dictionary) for each layer.
                             If not provided, it assumes all activations are 0.
        """

        # Check if node attributions are already computed
        relevance_flag = True
        for i in range(self.layer_len-1, 0, -1):
            if self.relevances.block[i].sae is None or self.relevances.block[i].error is None:
                relevance_flag = False
                break
        if not relevance_flag:
            raise ValueError("Node attributions must be computed before edge attributions.")

        # Compute Edge Attribution
        for i in range(self.layer_len-1, 0, -1): # (i+1 -> i)
            if self.verbose:
                print(f"Computing Edge Attributions for Layer {i} to {i+1}...", end=" ")
                start = time.time()
            self._compute_edge_attr(
                layer_num=(i, i+1),
                grad=grads[i+1] if i < self.layer_len-1 else None,
                mean_feature_act=mean_feature_acts[i-1] if mean_feature_acts is not None else None,
                mean_error_act=mean_error_acts[i-1] if mean_error_acts is not None else None,
            )
            if self.verbose:
                end = time.time()
                print(f"Edge Attribution Time: {end-start:.4f}s")
                self.time += end-start
        if self.verbose:
            print(f"Total Time: {self.time:.4f}s")
    
    
    @torch.no_grad()
    def _compute_edge_attr(
        self, 
        layer_num: tuple,
        grad: Union[tuple, None],
        mean_feature_act: torch.Tensor = None,
        mean_error_act: torch.Tensor = None,
    ):
        """
        Compute the attribution for the edges in the graph.
        
        Args:
            layer_num: Layer numbers (tuple) for which to compute the attribution.
            grad: Gradients for the nodes (sae and error). For the last layer, it is None.
            mean_feature_act: Mean feature activation for the layer. (Optional)
            mean_error_act: Mean error activation for the layer. (Optional)
        """

        # In the last layer, the edge is already computed as node
        if layer_num == (self.layer_len-1, self.layer_len):
            self.relevances.block[layer_num[0]].edge = torch.cat((
                self.relevances.block[layer_num[0]].sae.sum(dim=0),  # [Fi]
                self.relevances.block[layer_num[0]].error.sum().unsqueeze(0),  # [1]
            ), dim=0).unsqueeze(1)  # [Fi + 1, 1]
            return
        
        # Skip if no SAE model is present
        if layer_num[0]-1 not in self.sae_models.keys() or layer_num[1]-1 not in self.sae_models.keys():
            return
        
        ##### Preparation #####
        z = self.sae_models[layer_num[0]-1].intermediates['z']  # [T, D*R]
        W_dec = self.sae_models[layer_num[0]-1].W_dec  # [D*R, D]
        x_std = self.sae_models[layer_num[0]-1].intermediates['x_std']  # [T, 1]
        x = self.vit_blocks[layer_num[0]].intermediates['residual_in']  # [1, T, D]
        error = x.squeeze() - self.sae_models[layer_num[0]-1].intermediates['x_hat']  # [T, D]
        sae_grad, _ = grad  # [T, D*R]
        
        ##### Layer-wise Model Function #####
        def _layerwise_model_func(x):
            x = self.vit_blocks[layer_num[0]](x) # [1, T, D] #! Can be more generalized (num_0 to num_1)
            x = x[0, :, :] # [T, D]
            x_mean, x_std = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True).detach()
            x = (x - x_mean) / (x_std + 1e-5)
            x_cent = x - self.sae_models[layer_num[1]-1].b_dec
            z_before_acts = x_cent @ self.sae_models[layer_num[1]-1].W_enc # [B*T, F]
            acts = F.relu(z_before_acts)  # [B*T, F]
            acts_topk = torch.topk(acts, self.sae_models[layer_num[1]-1].cfg["top_k"], dim=-1)  # (Indices, Values)
            acts_topk = torch.zeros_like(acts).scatter(-1, acts_topk.indices, acts_topk.values)  # [B*T, F] = [T, D*R]
            acts_topk = acts_topk * sae_grad # [T, D*R] (Gradient Trick)
            acts_topk_sum = acts_topk.sum(dim=0)  # [F] (Corruption)
            return acts_topk_sum

        ##### Jacobian Computation #####

        # Fi <- Fo, Ei <- Fo
        edge = torch.func.jacrev(_layerwise_model_func, chunk_size=None)(x).squeeze() # [Fo, 1, T, D] -> [Fo, T, D]
        if mean_error_act is not None:
            R_err = einsum('Fo T D, T D -> Fo', [edge, error - mean_error_act])  # [Fo]
        else:
            R_err = einsum('Fo T D, T D -> Fo', [edge, error])  # [Fo]
        edge = edge @ W_dec.T  # [Fo, T, Fi]
        edge = edge * x_std.unsqueeze(0)  # [Fo, T, Fi]
        if mean_feature_act is not None:
            edge = einsum('Fo T Fi, T Fi -> Fi Fo', [edge, z - mean_feature_act])
        else:
            edge = einsum('Fo T Fi, T Fi -> Fi Fo', [edge, z])  # [Fi, Fo]
        edge = torch.cat((edge, R_err.unsqueeze(0)), dim=0)  # [Fi + 1, Fo]

        # Fi <- Eo, Ei <- Eo
        if layer_num[0] != self.layer_len-1:
            node = torch.cat((
                self.relevances.block[layer_num[0]].sae.sum(dim=0),  # [Fi]
                self.relevances.block[layer_num[0]].error.sum().unsqueeze(0),  # [1]
            ), dim=0) # [Fi + 1]
            node = node - edge.sum(dim=1)
            edge = torch.cat((edge, node.unsqueeze(1)), dim=1) # [Fi + 1, Fo + 1]
        
        self.relevances.block[layer_num[0]].edge = edge.detach()  # [Fi + 1, Fo + 1]

        return


    @torch.no_grad()
    def compute_node_and_edge_with_leaf_error_attributions(
            self,
            grads: dict,
            mean_feature_acts: dict = None,
            mean_error_acts: dict = None,
            hierarchical_attribution: bool = False,
            topn: Union[None, float] = None,
            topk: Union[None, float] = None,
            threshold: Union[None, float] = None,
            max_node_num: Union[None, int] = None,
        ):
        """
        Compute node and edge attributions for the nodes in the graph.
        Warning: Do not support IG (Integrated Gradients).
        
        Args:
            grads: Gradients for the nodes (sae and error).
            mean_feature_acts: Mean feature activation (dictionary) for each layer.
                               If not provided, it assumes all activations are 0.
            mean_error_acts: Mean error activation (dictionary) for each layer.
                             If not provided, it assumes all activations are 0.
            hierarchical_attribution: Flag to compute hierarchical attributions.
            topn, topk, threshold, max_node_num: Parameters for hierarchical attribution.
        """

        for i in range(self.layer_len-1, 0, -1):
            
            if self.verbose:
                print(f"Computing Node and Edge Attributions for Layer {i} to {i+1}...", end=" ")
                start = time.time()
            
            if i == self.layer_len-1:
                self._compute_node_attr(
                    layer_num=i, 
                    grad=grads[i],
                    mean_feature_act=mean_feature_acts[i-1] if mean_feature_acts is not None else None,
                    mean_error_act=mean_error_acts[i-1] if mean_error_acts is not None else None,
                )
                self.relevances.block[i].edge = torch.cat((
                    self.relevances.block[i].sae.sum(dim=0),  # [Fi]
                    self.relevances.block[i].error.sum().unsqueeze(0),  # [1]
                ), dim=0).unsqueeze(1)  # [Fi + 1, 1]
                
            else:
                ##### Preparation #####
                z = self.sae_models[i-1].intermediates['z']  # [T, D*R]
                W_dec = self.sae_models[i-1].W_dec  # [D*R, D]
                x_std = self.sae_models[i-1].intermediates['x_std']  # [T, 1]
                x = self.vit_blocks[i].intermediates['residual_in']  # [1, T, D]
                error = x.squeeze() - self.sae_models[i-1].intermediates['x_hat']  # [T, D]
                sae_grad, _ = grads[i+1]  # [T, D*R]
                
                ##### Layer-wise Model Function #####
                def _layerwise_model_func(x):
                    x = self.vit_blocks[i](x) # [1, T, D] #! Can be more generalized (num_0 to num_1)
                    x = x[0, :, :] # [T, D]
                    x_mean, x_std = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True).detach()
                    x = (x - x_mean) / (x_std + 1e-5)
                    x_cent = x - self.sae_models[i].b_dec
                    z_before_acts = x_cent @ self.sae_models[i].W_enc # [B*T, F]
                    acts = F.relu(z_before_acts)  # [B*T, F]
                    acts_topk = torch.topk(acts, self.sae_models[i].cfg["top_k"], dim=-1)  # (Indices, Values)
                    acts_topk = torch.zeros_like(acts).scatter(-1, acts_topk.indices, acts_topk.values)  # [B*T, F] = [T, D*R]
                    acts_topk = acts_topk * sae_grad # [T, D*R] (Gradient Trick)
                    acts_topk_sum = acts_topk.sum(dim=0)  # [F] (Corruption)
                    return acts_topk_sum

                ##### Jacobian Computation #####
                
                # Past node hierarchy
                if hierarchical_attribution:
                    past_nodes = torch.cat((
                        self.relevances.block[i+1].sae.sum(dim=0),  # [Fo]
                        self.relevances.block[i+1].error.sum().unsqueeze(0),  # [1]
                    ), dim=0)  # [Fo + 1]
                    past_nodes = torch.clamp(past_nodes, min=0.0)  # [Fo + 1]
                    past_nodes = past_nodes / past_nodes.sum()  # [Fo + 1]
                    sorted_past_nodes = torch.argsort(past_nodes, descending=True)
                    if topn is not None:
                        indices_num = int(topn * max_node_num)
                        node_indices = sorted_past_nodes[:indices_num]
                    elif topk is not None:
                        indices_num = int(topk * (past_nodes.shape[0] + 1))
                        node_indices = sorted_past_nodes[:indices_num]
                    elif threshold is not None:
                        node_relevance_sum, node_idx = 0.0, 0
                        while node_relevance_sum < threshold:
                            try:
                                node_relevance_sum += past_nodes[sorted_past_nodes[node_idx]]
                                node_idx += 1
                            except:
                                print(f"Layer {i+1} has no node relevance above threshold.")
                                break
                        node_indices = sorted_past_nodes[:node_idx]
                    # Remove the last index (error)
                    node_indices = node_indices[node_indices != past_nodes.shape[0] - 1]
                         
                # Fi <- Fo, Ei <- Fo
                edge = torch.func.jacrev(_layerwise_model_func, chunk_size=None)(x).squeeze() # [Fo, 1, T, D] -> [Fo, T, D]
                
                if hierarchical_attribution:
                    edge[~torch.isin(torch.arange(edge.shape[0], device=edge.device), node_indices)] = 0.0  # [Fo, T, D]
                    
                if mean_error_acts[i-1] is not None:
                    R_err = edge * (error - mean_error_acts[i-1])  # [Fo, T, D]
                else:
                    R_err = edge * error  # [Fo, T, D]
                edge = edge @ W_dec.T  # [Fo, T, Fi]
                edge = edge * x_std.unsqueeze(0)  # [Fo, T, Fi]

                grads[i] = [edge.sum(dim=0).detach(), None]  # [T, Fi]  #! Updated

                if mean_feature_acts[i-1] is not None:
                    edge = edge * (z - mean_feature_acts[i-1])  # [Fo, T, Fi]
                else:
                    edge = edge * z  # [Fo, T, Fi]

                self.relevances.block[i].sae = edge.sum(dim=0).detach()  # [T, Fi]
                self.relevances.block[i].error = R_err.sum(dim=0).detach()  # [T, D]

                edge = edge.sum(dim=1).permute(1, 0)  # [Fi, Fo]
                edge = torch.cat((edge, R_err.sum(dim=(1, 2)).unsqueeze(0)), dim=0)  # [Fi + 1, Fo]
                dummy = torch.zeros_like(edge[:, 0]).unsqueeze(1).detach()  # [Fi + 1, 1]
                edge = torch.cat((edge, dummy), dim=1)  # [Fi + 1, Fo + 1]
                
                self.relevances.block[i].edge = edge.detach()  # [Fi + 1, Fo + 1]

            if self.verbose:
                end = time.time()
                print(f"Node and Edge Attribution Time: {end-start:.4f}s")
                self.time += end-start
        
        if self.verbose:
            print(f"Total Time: {self.time:.4f}s")
        
        return
    
    
    def get_neuron_grad(self, label, logits, adjust_logit=False):
        """
        Get gradients for the neurons in the graph.
        
        Args:
            label: Target label for the classification task.
            logits: Output logits from the model.
            adjust_logit: Flag to adjust the logits for the target label.
        """
        
        # Retain gradients
        for i in range(self.layer_len-1, 0, -1):
            self.vit_blocks[i].intermediates["residual_in"].retain_grad()
            
        # Compute gradients
        if adjust_logit: logit = logits[:, label] - logits.mean()
        else: logit = logits[:, label]
        logit.backward()
        
        # Collect gradients
        grads = {}
        for i in range(self.layer_len-1, 0, -1):
            grads[i] = self.vit_blocks[i].intermediates["residual_in"].grad.squeeze().detach() # [T, D]
        
        return grads

    
    def compute_neuron_node_attributions(
            self,
            grads: dict,
            mean_neuron_acts: dict = None,
        ):
        """
        Compute attributions for the neurons in the graph.
        
        Args:
            grads: Gradients for the neurons.
            mean_neuron_acts: Mean neuron activation (dictionary) for each layer.
                               If not provided, it assumes all activations are 0.
        """
        
        # Compute Attribution
        if self.verbose:
            print("Computing Node Attributions...", end=" ")
            start = time.time()
        for i in range(self.layer_len-1, 0, -1):
            self._compute_neuron_node_attr(
                layer_num=i, 
                grad=grads[i],
                mean_neuron_act=mean_neuron_acts[i-1] if mean_neuron_acts is not None else None,
            )
        if self.verbose:
            end = time.time()
            print(f"Node Attribution Time: {end-start:.4f}s")
            self.time += end-start
            
            
    @torch.no_grad()
    def _compute_neuron_node_attr(
        self, 
        layer_num: int,
        grad: torch.Tensor,
        mean_neuron_act: torch.Tensor = None,
    ):
        """
        Compute the attribution for the neurons in the graph.
        
        Args:
            layer_num: Layer number for which to compute the attribution.
            grad: Gradients for the neurons.
            mean_neuron_act: Mean neuron activation for the layer. (Optional)
        """
        
        ##### Preparation #####
        x = self.vit_blocks[layer_num].intermediates["residual_in"].squeeze()  # [T, D]
        
        if mean_neuron_act is not None:
            self.relevances.block[layer_num].sae = (grad * (x - mean_neuron_act)).detach()  # [T, D]
        else:
            self.relevances.block[layer_num].sae = (grad * x).detach()
        
        return


    def compute_neuron_edge_attributions(
            self, 
            grads: dict,
            mean_neuron_acts: dict = None,
        ):
        """
        Compute edge attributions for the neurons in the graph.
        
        Args:
            grads: Gradients for the neurons.
            mean_neuron_acts: Mean neuron activation (dictionary) for each layer.
                               If not provided, it assumes all activations are 0.
        """

        # Check if node attributions are already computed
        relevance_flag = True
        for i in range(self.layer_len-1, 0, -1):
            if self.relevances.block[i].sae is None:
                relevance_flag = False
                break
        if not relevance_flag:
            raise ValueError("Node attributions must be computed before edge attributions.")

        # Compute Edge Attribution
        for i in range(self.layer_len-1, 0, -1):  # (i+1 -> i)
            if self.verbose:
                print(f"Computing Edge Attributions for Layer {i} to {i+1}...", end=" ")
                start = time.time()
            self._compute_neuron_edge_attr(
                layer_num=(i, i+1),
                grad=grads[i+1] if i < self.layer_len-1 else None,
                mean_neuron_act=mean_neuron_acts[i-1] if mean_neuron_acts is not None else None,
            )
            if self.verbose:
                end = time.time()
                print(f"Edge Attribution Time: {end-start:.4f}s")
                self.time += end-start
        if self.verbose:
            print(f"Total Time: {self.time:.4f}s")
    
    
    @torch.no_grad()
    def _compute_neuron_edge_attr(
        self, 
        layer_num: tuple,
        grad: Union[torch.Tensor, None],
        mean_neuron_act: torch.Tensor = None,
    ):
        """
        Compute the attribution for the edges in the graph.
        
        Args:
            layer_num: Layer numbers (tuple) for which to compute the attribution.
            grad: Gradients for the neurons. For the last layer, it is None.
            mean_neuron_act: Mean neuron activation for the layer. (Optional)
        """

        # In the last layer, the edge is already computed as node
        if layer_num == (self.layer_len-1, self.layer_len):
            self.relevances.block[layer_num[0]].edge = self.relevances.block[layer_num[0]].sae.sum(dim=0).unsqueeze(1)  # [D]
            return
        
        ##### Preparation #####
        x = self.vit_blocks[layer_num[0]].intermediates["residual_in"]
        
        ##### Layer-wise Model Function #####
        def _layerwise_model_func(x):
            x = self.vit_blocks[layer_num[0]](x)  # [1, T, D]
            x = x[0, :, :]  # [T, D]
            x = x * grad  # [T, D] (Gradient Trick)
            x_sum = x.sum(dim=0)  # [D] (Corruption)
            return x_sum
        
        ##### Jacobian Computation #####
        edge = torch.func.jacrev(_layerwise_model_func, chunk_size=None)(x).squeeze()  # [Do, 1, T, Di] -> [Do, T, Di]
        if mean_neuron_act is not None:
            edge = einsum('Do T Di, T Di -> Di Do', [edge, x.squeeze() - mean_neuron_act])  
        else:
            edge = einsum('Do T Di, T Di -> Di Do', [edge, x.squeeze()])  # [Di, Do]
            
        self.relevances.block[layer_num[0]].edge = edge.detach()  # [Di, Do]
        
        return
    
    
    @torch.no_grad()
    def _special_compute_fine_node(
            self,
            grads: dict,
            mean_feature_acts: dict = None,
            mean_error_acts: dict = None,
            topn: Union[None, float] = None,
            topk: Union[None, float] = None,
            threshold: Union[None, float] = None,
            max_node_num: Union[None, int] = None,
        ):
        """
        Compute fine node attributions for the nodes in the graph.
        Warning: Do not support IG (Integrated Gradients).

        Args:
            grads: Gradients for the nodes (sae and error).
            mean_feature_acts: Mean feature activation (dictionary) for each layer.
                               If not provided, it assumes all activations are 0.
            mean_error_acts: Mean error activation (dictionary) for each layer.
                             If not provided, it assumes all activations are 0.
            topn, topk, threshold, max_node_num: Parameters for hierarchical attribution.
        """
        
        for i in range(self.layer_len-1, 0, -1):
            
            if self.verbose:
                print(f"Computing Fine Node Attributions for Layer {i} to {i+1}...", end=" ")
                start = time.time()
            
            if i == self.layer_len-1:
                self._compute_node_attr(
                    layer_num=i, 
                    grad=grads[i],
                    mean_feature_act=mean_feature_acts[i-1] if mean_feature_acts is not None else None,
                    mean_error_act=mean_error_acts[i-1] if mean_error_acts is not None else None,
                )
            
            else:
                ##### Preparation #####
                z = self.sae_models[i-1].intermediates['z']  # [T, D*R]
                W_dec = self.sae_models[i-1].W_dec  # [D*R, D]
                x_std = self.sae_models[i-1].intermediates['x_std']  # [T, 1]
                x = self.vit_blocks[i].intermediates['residual_in']  # [1, T, D]
                error = x.squeeze() - self.sae_models[i-1].intermediates['x_hat']  # [T, D]
                sae_grad, _ = grads[i+1]  # [T, D*R]
                
                ##### Layer-wise Model Function #####
                def _layerwise_model_func(x, node_indices):
                    x = self.vit_blocks[i](x) # [1, T, D] #! Can be more generalized (num_0 to num_1)
                    x = x[0, :, :] # [T, D]
                    x_mean, x_std = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True).detach()
                    x = (x - x_mean) / (x_std + 1e-5)
                    x_cent = x - self.sae_models[i].b_dec
                    z_before_acts = x_cent @ self.sae_models[i].W_enc # [B*T, F]
                    acts = F.relu(z_before_acts)  # [B*T, F]
                    acts_topk = torch.topk(acts, self.sae_models[i].cfg["top_k"], dim=-1)  # (Indices, Values)
                    acts_topk = torch.zeros_like(acts).scatter(-1, acts_topk.indices, acts_topk.values)  # [B*T, F] = [T, D*R]
                    acts_topk = acts_topk * sae_grad # [T, D*R] (Gradient Trick)
                    rows, cols = node_indices // acts_topk.shape[1], node_indices % acts_topk.shape[1]
                    acts_topk[rows, cols] = 0.0  # [T, D*R]
                    acts_topk_sum = acts_topk.sum()  # [1] (Corruption)
                    return acts_topk_sum

                ##### Jacobian Computation #####
                
                # Past node hierarchy
                past_nodes = torch.cat((
                    self.relevances.block[i+1].sae.flatten(),  # [T*Fo]
                    self.relevances.block[i+1].error.sum().unsqueeze(0),  # [1]
                ), dim=0)  # [T*Fo + 1]
                past_nodes = torch.clamp(past_nodes, min=0.0)  # [T*Fo + 1]
                past_nodes = past_nodes / past_nodes.sum()  # [T*Fo + 1]
                sorted_past_nodes = torch.argsort(past_nodes, descending=True)
                if topn is not None:
                    indices_num = int(topn * max_node_num)
                    node_indices = sorted_past_nodes[:indices_num]
                elif topk is not None:
                    indices_num = int(topk * (past_nodes.shape[0] + 1))
                    node_indices = sorted_past_nodes[:indices_num]
                elif threshold is not None:
                    node_relevance_sum, node_idx = 0.0, 0
                    while node_relevance_sum < threshold:
                        try:
                            node_relevance_sum += past_nodes[sorted_past_nodes[node_idx]]
                            node_idx += 1
                        except:
                            print(f"Layer {i+1} has no node relevance above threshold.")
                            break
                    node_indices = sorted_past_nodes[:node_idx]
                # Remove the last index (error)
                node_indices = node_indices[node_indices != past_nodes.shape[0] - 1]

                full_node_indices = torch.arange(past_nodes.shape[0] - 1).to(node_indices.device)
                mask = ~torch.isin(full_node_indices, node_indices)
                         
                # Fi <- Fo, Ei <- Fo
                edge = torch.func.jacrev(_layerwise_model_func, chunk_size=None)(x, full_node_indices[mask]).squeeze() # [1, 1, T, D] -> [T, D]

                if mean_error_acts[i-1] is not None:
                    R_err = edge * (error - mean_error_acts[i-1])  # [T, D]
                else:
                    R_err = edge * error  #  [T, D]
                edge = edge @ W_dec.T  # [T, Fi]
                edge = edge * x_std  # [T, Fi]

                grads[i] = [edge.detach(), None]  # [T, Fi]  #! Updated

                if mean_feature_acts[i-1] is not None:
                    edge = edge * (z - mean_feature_acts[i-1])  # [T, Fi]
                else:
                    edge = edge * z  # [T, Fi]

                self.relevances.block[i].sae = edge.detach()  # [T, Fi]
                self.relevances.block[i].error = R_err.detach()  # [T, D]

            if self.verbose:
                end = time.time()
                print(f"Node and Edge Attribution Time: {end-start:.4f}s")
                self.time += end-start
        
        if self.verbose:
            print(f"Total Time: {self.time:.4f}s")
        
        return