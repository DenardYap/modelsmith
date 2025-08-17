# Credit: Kyegomez on Github (Zeta)
# Taken directly from Zeta's codebase: https://github.com/kyegomez/zeta/blob/8b87de565413cde216ffd256ea46729211f9afde/zeta/nn/quant/ste.py#L27

import torch
import torch.nn as nn


class STEFunc(torch.autograd.Function):
    """
    Straight Through Estimator

    This function is used to bypass the non differentiable operations


    Args:
        input (torch.Tensor): the input tensor

    Returns:
        torch.Tensor: the output tensor

    Usage:
    >>> x = torch.randn(2, 3, requires_grad=True)
    >>> y = STEFunc.apply(x)
    >>> y.backward(torch.ones_like(x))
    >>> x.grad


    """

    @staticmethod
    def forward(ctx, input):
        """
        Forward pass of the STE function where we clamp the input between -1 and 1 and then apply the sign function

        Args:
            ctx (torch.autograd.Function): the context object
            input (torch.Tensor): the input tensor



        """
        return torch.sign(input)
    

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the STE function where we bypass the non differentiable operations

        """
        # Bypass the non differterable operations
        return grad_output

    # This is another method of doing STE, which is in the original paper 
    # @staticmethod
    # def forward(ctx, input):
    #     ctx.save_for_backward(input)
    #     return torch.sign(input)
    # @staticmethod
    # def backward(ctx, grad_output):
    #     input = ctx.saved_tensors[0]
    #     grad_input = grad_output.clone()
    #     grad_input[input.abs() > 1] = 0
    #     return grad_input
