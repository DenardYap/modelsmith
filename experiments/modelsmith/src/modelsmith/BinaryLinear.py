import torch
import torch.nn as nn
import torch.nn.functional as F
from .ste import STEFunc
try:
	from . import _C as ms
except Exception:
	ms = None

class BinLinear(nn.Linear):
	"""
	Custom linear layer with bit quantization.

	Both the weights and the activations are quantized to 1 bit (-1, 1).
	"""

	def __init__(self, in_features, out_features, bias=True):
		super().__init__(in_features, out_features, bias)
		self.alpha = nn.Parameter(self.weight.abs().mean())
		
	def forward(self, input):
		weight_bin = STEFunc.apply(self.weight)
		input = input - input.mean(dim=1, keepdim=True)
		beta = input.abs().mean()
		input_bin = STEFunc.apply(input)
		# Compute binary matmul without bias first
		if self.training or ms is None or not hasattr(ms, "binary_matmul_tiled"):
			mat = F.linear(input_bin, weight_bin, bias=None)
		else:
			mat = ms.binary_matmul_tiled(input_bin, weight_bin.T).to(input.dtype)
		# Scale and then add bias (do not scale bias)
		output = mat * self.alpha * beta
		if self.bias is not None:
			output = output + self.bias
		return output 