import torch
from torch.autograd import Function

class STEFunc(Function):
	@staticmethod
	def forward(ctx, input):
		return torch.sign(input)

	@staticmethod
	def backward(ctx, grad_output):
		return grad_output.clone() 