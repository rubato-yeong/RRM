import torch
import torch.nn as nn


def get_relevance(
    model: nn.Module,
    image: torch.Tensor, 
    label: torch.Tensor,
    method: str = 'grad', 
    steps: int = 5,
    target: str = 'feature',
    leaf_error: bool = True,
    mean_feature_acts: dict = None,
    mean_error_acts: dict = None,
    mean_neuron_acts: dict = None,
    get_logit: bool = False,
    adjust_logit: bool = False,
    compute_edge: bool = False,
):
    """
    Get the relevance score for the input image using the specified method.
    
    Args:
        model (Cascaded[Model]WithSAE(forGrad)): The model to be used for attribution.
        image (torch.Tensor): The input image tensor. Do not add batch dimension.
        label (torch.Tensor): The label tensor.
        method (str): The attribution method to use.
        target (str): The target for attribution. Options are 'feature' or 'neuron'.
        mean_feature_acts: Mean feature activation (dictionary) for each layer.
                            If not provided, it assumes all activations are 0.
        mean_error_acts: Mean error activation (dictionary) for each layer.
                            If not provided, it assumes all activations are 0.
        mean_neuron_acts: Mean neuron activation (dictionary) for each layer.
                            If not provided, it assumes all activations are 0.
        get_logit (bool): Whether to return the logit value.
        adjust_logit (bool): Whether to adjust the logit value.
        compute_edge (bool): Whether to compute edge relevance.
        
    Returns:
        Relevances: The relevance score for the input image.
    """
    
    if target == 'neuron':
        if method == 'grad':
            return attr_neuron_grad(model, image, label, mean_neuron_acts, get_logit, adjust_logit, compute_edge)
        else:
            raise ValueError("Invalid attribution method for neuron. Choose 'grad'.")
    
    elif target == 'feature':
        if leaf_error:
            if method == 'grad':
                return attr_grad_leaf_error(model, image, label, mean_feature_acts, mean_error_acts, get_logit, adjust_logit)
            else:
                raise ValueError("Invalid attribution method for leaf error. Choose 'grad'.")
        else:
            if method == 'grad':
                return attr_grad(model, image, label, mean_feature_acts, mean_error_acts, get_logit, adjust_logit, compute_edge)
            else:
                raise ValueError("Invalid attribution method. Choose from 'grad' or 'ig'.")
    
    else:
        raise ValueError("Invalid target. Choose 'feature' or 'neuron'.")


def attr_grad_leaf_error(model, image, label, mean_feature_acts, mean_error_acts, get_logit, adjust_logit):

    model.eval()
    
    logits = model(image.unsqueeze(0))
    
    grads = model.get_grad(label, logits, adjust_logit)

    model.compute_node_and_edge_with_leaf_error_attributions(
        grads=grads,
        mean_feature_acts=mean_feature_acts,
        mean_error_acts=mean_error_acts,
    )

    model.clean_grad()

    if get_logit:
        return model.relevances, logits
    else:
        return model.relevances
    

def attr_grad(model, image, label, mean_feature_acts, mean_error_acts, get_logit, adjust_logit, compute_edge):
    """
    Perform gradient-based attribution and return the relevance score.
    
    Args:
        model (Cascaded[Model]WithSAEforGrad): The model to be used for gradient-based attribution.
        image (torch.Tensor): The input image tensor. Do not add batch dimension.
        label (torch.Tensor): The label tensor.
        mean_feature_acts: Mean feature activation (dictionary) for each layer.
                            If not provided, it assumes all activations are 0.
        mean_error_acts: Mean error activation (dictionary) for each layer.
                            If not provided, it assumes all activations are 0.
        get_logit (bool): Whether to return the logit value.
        adjust_logit (bool): Whether to adjust the logit value.
        compute_edge (bool): Whether to compute edge relevance.
        
    Returns:
        Relevances: The relevance score for the input image.
    """
    
    model.eval()
    
    logits = model(image.unsqueeze(0))
    
    grads = model.get_grad(label, logits, adjust_logit)
    model.compute_node_attributions(
        grads=grads,
        mean_feature_acts=mean_feature_acts,
        mean_error_acts=mean_error_acts,
        )
    if compute_edge:
        model.compute_edge_attributions(
            grads=grads,
            mean_feature_acts=mean_feature_acts,
            mean_error_acts=mean_error_acts,
            )
    
    model.clean_grad()

    if get_logit:
        return model.relevances, logits
    else:
        return model.relevances


def attr_neuron_grad(model, image, label, mean_neuron_acts, get_logit, adjust_logit, compute_edge):
    """
    Perform neuron gradient-based attribution and return the relevance score.
    
    Args:
        model (Cascaded[Model]WithSAEforGrad): The model to be used for neuron gradient-based attribution.
        image (torch.Tensor): The input image tensor. Do not add batch dimension.
        label (torch.Tensor): The label tensor.
        mean_neuron_acts: Mean neuron activation (dictionary) for each layer.
                            If not provided, it assumes all activations are 0.
        get_logit (bool): Whether to return the logit value.
        adjust_logit (bool): Whether to adjust the logit value.
        compute_edge (bool): Whether to compute edge relevance.
        
    Returns:
        Relevances: The relevance score for the input image.
    """
    
    model.eval()
    
    logits = model(image.unsqueeze(0))
    
    grads = model.get_neuron_grad(label, logits, adjust_logit)
    model.compute_neuron_node_attributions(
        grads=grads,
        mean_neuron_acts=mean_neuron_acts,
        )
    if compute_edge:
        model.compute_neuron_edge_attributions(
            grads=grads,
            mean_neuron_acts=mean_neuron_acts,
            )
    
    model.clean_grad()

    if get_logit:
        return model.relevances, logits
    else:
        return model.relevances