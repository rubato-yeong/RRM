import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class FullGradLayerNorm(nn.Module):
    """
    Layer Normalization with FullGrad-completeness support.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
    ):
        
        super().__init__()

        # Initialize parameters
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def forward(self, x: torch.Tensor):

        # Compute mean and variance
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps
        std = torch.sqrt(var).detach()

        # Normalize
        x = (x - mean) / std
        x = x * self.weight + self.bias

        return x


class FullGradGELU(nn.Module):
    """
    GELU activation function with FullGrad-completeness support.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        
        # Compute NonLinearGate
        gate = (0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))).detach()
        x = x * gate

        return x


class FullGradQuickGELU(nn.Module):
    """
    QuickGELU activation function with FullGrad-completeness support.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        
        # Compute NonLinearGate
        gate = torch.sigmoid(1.702 * x).detach()
        x = x * gate

        return x


class FullGradNormalize(nn.Module):
    """
    Normalization layer with FullGrad-completeness support.
    """

    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor, dim=-1):
        
        return x / torch.clamp(torch.norm(x, dim=dim, keepdim=True), min=1e-12).detach()


class LinearGammaFunction(Function):
    @staticmethod
    def forward(input, weight, bias, gamma):
        """
        Forward pass for a linear layer.
        Args:
            ctx: Context object to save tensors for backward pass.
            input: Input tensor of shape (*, in_features).
            weight: Weight tensor of shape (out_features, in_features).
            bias: Bias tensor of shape (out_features,).
            gamma: gamma coefficient for the gamma-rule LRP.
        Returns:
            output: Output tensor of shape (*, out_features).
        """
        output = input @ weight.t() + bias
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Setup context for backward pass.
        Args:
            inputs:
                - input: Input tensor of shape (*, in_features).
                - weight: Weight tensor of shape (out_features, in_features).
                - bias: Bias tensor of shape (out_features,).
                - gamma: gamma coefficient for the gamma-rule LRP.
            output: Output tensor of shape (*, out_features).
        """
        input, weight, bias, gamma = inputs
        ctx.save_for_backward(output, input, weight, bias)
        ctx.gamma = gamma

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for a linear layer.
        Args:
            ctx: Context object containing saved tensors.
            grad_output: Gradient of the loss w.r.t. the output, of shape (*, out_features).
        Returns:
            grad_input: Gradient of the loss w.r.t. the input, of shape (*, in_features).
            grad_weight: Gradient of the loss w.r.t. the weight, of shape (out_features, in_features).
            grad_bias: Gradient of the loss w.r.t. the bias, of shape (out_features,).
        """
        output, input, weight, bias = ctx.saved_tensors
        gamma = ctx.gamma
        
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:  # Gradient w.r.t. input
            
            # Indicator matrix
            pos_indicator = (output > 0).float().unsqueeze(-2)  # [B, T, 1, D2]
            neg_indicator = (output < 0).float().unsqueeze(-2)  # [B, T, 1, D2]

            # Output matrix
            matrix_out = input.unsqueeze(-1) * weight.t()  # [B, T, D1, D2] (y_jk)

            # Prepare positive and negative matrices
            pos_matrix_out = torch.clamp(matrix_out, min=0)  # [B, T, D1, D2] (y_jk+)
            neg_matrix_out = torch.clamp(matrix_out, max=0)  # [B, T, D1, D2] (y_jk-)
            pos_bias_out = torch.clamp(bias, min=0)  # [D2] (b_k+)
            neg_bias_out = torch.clamp(bias, max=0)  # [D2] (b_k-)

            # Jacobian matrix [B, T, D1, D2]
            grad_input = pos_indicator * (matrix_out + gamma * pos_matrix_out) / (output.unsqueeze(-2) + gamma * (torch.sum(pos_matrix_out, dim=-2, keepdim=True) + pos_bias_out)) \
                    + neg_indicator * (matrix_out + gamma * neg_matrix_out) / (output.unsqueeze(-2) + gamma * (torch.sum(neg_matrix_out, dim=-2, keepdim=True) + neg_bias_out))
            grad_input = grad_input * output.unsqueeze(-2)
            grad_input = torch.einsum('...j,...ij->...i', grad_output, grad_input)  # [B, T, D1]
            grad_input = grad_input / torch.where(input == 0, 1, input)
            
        if ctx.needs_input_grad[1]:  # Gradient w.r.t. weight
            grad_weight = torch.einsum('...j,...i->ji', grad_output, input)
        if ctx.needs_input_grad[2]:
            grad_bias = torch.sum(grad_output, dim=0)

        return grad_input, grad_weight, grad_bias, None


class LinearGamma(nn.Module):
    def __init__(self, weight, bias, gamma=0.05):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.gamma = gamma

    def forward(self, input):
        return LinearGammaFunction.apply(input, self.weight, self.bias, self.gamma)


class LinearAbsorbingBiasFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, eps):
        """
        Forward pass for a linear layer.
        Args:
            ctx: Context object to save tensors for backward pass.
            input: Input tensor of shape (*, in_features).
            weight: Weight tensor of shape (out_features, in_features).
            bias: Bias tensor of shape (out_features,).
            eps: Small value to prevent division by zero.
        Returns:
            output: Output tensor of shape (*, out_features).
        """
        ctx.save_for_backward(input, weight, bias)
        ctx.eps = eps
        output = input @ weight.t() + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for a linear layer.
        Args:
            ctx: Context object containing saved tensors.
            grad_output: Gradient of the loss w.r.t. the output, of shape (*, out_features).
        Returns:
            grad_input: Gradient of the loss w.r.t. the input, of shape (*, in_features).
            grad_weight: Gradient of the loss w.r.t. the weight, of shape (out_features, in_features).
        """
        input, weight, bias = ctx.saved_tensors
        eps = ctx.eps
        
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:  # Gradient w.r.t. input
            denominator = input.shape[-1] * input.unsqueeze(-1).detach()
            denominator = torch.where(denominator == 0, eps, denominator)
            bias_weight = bias / denominator
            grad_input = torch.einsum('...j,...ij->...i', grad_output, (weight.t() + bias_weight))
        if ctx.needs_input_grad[1]:  # Gradient w.r.t. weight
            grad_weight = torch.einsum('...j,...i->ji', grad_output, input)

        return grad_input, grad_weight, grad_bias, None


class LinearAbsorbingBias(nn.Module):
    def __init__(self, weight, bias, backprop_eps=1e-5):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.eps = backprop_eps

    def forward(self, input):
        return LinearAbsorbingBiasFunction.apply(input, self.weight, self.bias, self.eps)


class ElementWiseLinearAbsorbingBiasFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, eps):
        """
        Forward pass for an element-wise linear layer.
        Args:
            ctx: Context object to save tensors for backward pass.
            input: Input tensor of shape (*, features).
            weight: Weight tensor of shape (features,).
            bias: Bias tensor of shape (features,).
            eps: Small value to prevent division by zero.
        Returns:
            output: Output tensor of shape (*, features).
        """
        ctx.save_for_backward(input, weight, bias)
        ctx.eps = eps
        output = input * weight + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for an element-wise linear layer.
        Args:
            ctx: Context object containing saved tensors.
            grad_output: Gradient of the loss w.r.t. the output, of shape (*, features).
        Returns:
            grad_input: Gradient of the loss w.r.t. the input, of shape (*, features).
            grad_weight: Gradient of the loss w.r.t. the weight, of shape (features,).
        """
        input, weight, bias = ctx.saved_tensors
        eps = ctx.eps
        
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:  # Gradient w.r.t. input
            denominator = input.detach()
            denominator = torch.where(denominator == 0, eps, denominator)
            bias_weight = bias / denominator
            grad_input = grad_output * (weight + bias_weight)
        if ctx.needs_input_grad[1]:  # Gradient w.r.t. weight
            grad_weight = torch.sum(grad_output * input, dim=0)

        return grad_input, grad_weight, grad_bias, None


class LayerNormAbsorbingBias(nn.Module):
    def __init__(self, weight, bias, eps=1e-5, backprop_eps=1e-5):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.eps = eps
        self.backprop_eps = backprop_eps

    def forward(self, input):
        
        # Compute mean and variance
        mean = torch.mean(input, dim=-1, keepdim=True)
        var = torch.var(input, dim=-1, keepdim=True, unbiased=False) + self.eps
        std = torch.sqrt(var).detach()

        # Normalize
        input = (input - mean) / std
        return ElementWiseLinearAbsorbingBiasFunction.apply(input, self.weight, self.bias, self.backprop_eps)
