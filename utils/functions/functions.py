import random
import numpy as np
import torch
import string
import re


def set_seed(seed: int):
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch (CPU & GPU).

    Args:
        seed (int): The seed value to ensure deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def custom_format(value):
    """
    Add '.0' to integer values for consistent formatting.
    
    Args:
        value (int or float): The numeric value to format.
    """
    if isinstance(value, int):
        return f"{value}.0"
    else:
        return str(value)


def einsum(equation, operands):
    """
    A function that mimics einops.einsum.
    Internally, it replaces multi-character tokens in the equation with single characters and calls torch.einsum.
    
    Args:
        equation: e.g., 't_k d_i, h d_i d_o -> h t_k d_o'
        operands: List of tensors to be used in the einsum operation
    """
    # Extract tokens using regular expressions (including numbers and underscores)
    tokens = re.findall(r"[a-zA-Z0-9_]+", equation)
    
    # Single-character tokens are used as is
    used = {token for token in tokens if len(token) == 1}
    # Available characters (default: ascii_letters excluding already used ones)
    allowed = [c for c in string.ascii_letters if c not in used]
    
    mapping = {}
    for token in tokens:
        if len(token) > 1 and token not in mapping:
            # Assign a character from allowed for new tokens
            mapping[token] = allowed.pop(0)
    
    # Replacement function: replace multi-character tokens with mapped characters
    def replace_token(match):
        token = match.group(0)
        return mapping.get(token, token)
    
    # Replace tokens in the entire equation (applies to all a-zA-Z0-9_)
    new_equation = re.sub(r"[a-zA-Z0-9_]+", replace_token, equation)
    # Example: 't_k d_i, h d_i d_o -> h t_k d_o' is transformed to 'a b, h b c -> h a c'
    
    return torch.einsum(new_equation, *operands)


def collate_fn_clip(batch, processor):
    images, labels = zip(*batch)
    pixel_values = torch.stack([
        processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        for image in images
    ])
    labels = torch.tensor(labels)
    return pixel_values, labels


def unnormalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).to(tensor.device)
    std = torch.tensor([0.5, 0.5, 0.5]).to(tensor.device)
    tensor = tensor * std[:, None, None] + mean[:, None, None]
    return tensor.clamp(0, 1)


def clip_unnormalize(tensor):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(tensor.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(tensor.device)
    tensor = tensor * std[:, None, None] + mean[:, None, None]
    return tensor.clamp(0, 1)