import torch
import torch.nn as nn
# import modelsmith  # your compiled C++ extension
import torch.nn.functional as F
from ste import STEFunc
import modelsmith as ms

class BinLinear(nn.Linear):
	"""
	Custom linear layer with bit quantization.

	Both the weights and the activations are quantized to 1 bit (-1, 1).
	"""


	def __init__(self, in_features, out_features, bias=True):
		super().__init__(in_features, out_features, bias)
		# TODO: Add other intiialization method like kaiming if needed 
		# self.register_buffer('alpha', torch.tensor(1.0))

		# Original paper shows that alpha = (1/n ||W||_ℓ1), but later 
		# implementations (TODO: cite) showed that it's better for alpha to be trainable 
		self.alpha = nn.Parameter(self.weight.abs().mean())
		
	def forward(self, input):
		"""
		TODO:
		1. Once we STEFunc.apply(self.weight) 
		2. Figure out how to implement custom matrix multiplication and 
		   update weight ourselves, but for now F.linear is fine
		3. How do we update the weights using real-valued of the weights and activations
		"""
		
		weight_bin = STEFunc.apply(self.weight)

		# normalize input to have 0 mean 
		input = input - input.mean(dim=1, keepdim=True)  # Zero-mean
		beta = input.abs().mean() 
		input_bin = STEFunc.apply(input)

		# Compute matmul without bias first
		if (not hasattr(ms, "binary_matmul_tiled")) or self.training:
			mat = F.linear(input_bin, weight_bin, bias=None)
		else:
			mat = ms.binary_matmul_tiled(input_bin, weight_bin.T).to(input.dtype)

		# Scale and then add bias (bias should not be scaled)
		output = mat * self.alpha * beta
		if self.bias is not None:
			output = output + self.bias

		return output


class BinLinearOld(nn.Linear):
	"""
	Custom linear layer with bit quantization.

	Both the weights and the activations are quantized to 1 bit (-1, 1).
	"""


	def __init__(self, in_features, out_features, bias=True):
		super().__init__(in_features, out_features, bias)
		# TODO: Add other intiialization method like kaiming if needed 
		# self.register_buffer('alpha', torch.tensor(1.0))

		# Original paper shows that alpha = (1/n ||W||_ℓ1), but later 
		# implementations (TODO: cite) showed that it's better for alpha to be trainable 
		self.alpha = nn.Parameter(self.weight.abs().mean())
		
	def forward(self, input):
		"""
		TODO:
		1. Once we STEFunc.apply(self.weight) 
		2. Figure out how to implement custom matrix multiplication and 
		   update weight ourselves, but for now F.linear is fine
		3. How do we update the weights using real-valued of the weights and activations
		"""
		weight_bin = STEFunc.apply(self.weight)

		# normalize input to have 0 mean 
		input = input - input.mean(dim=1, keepdim=True)  # Zero-mean

		beta = input.abs().mean() 

		input_bin = STEFunc.apply(input)

		mat = F.linear(input_bin, weight_bin, bias=None)
		output = mat * self.alpha * beta
		if self.bias is not None:
			output = output + self.bias

		return output

if __name__ == "__main__":
	# Make a sequential of BinLinear layers
	ms_Linear = BinLinear(768, 10, True)
	# ms_Linear = nn.Sequential(
	#     BinLinear(768, 512),
	#     BinLinear(512, 512),
	#     BinLinear(512, 512),
	#     BinLinear(512, 512),
	#     BinLinear(512, 512),
	#     BinLinear(512, 512),
	#     BinLinear(512, 10),
	# )
	torch_Linear = nn.Linear(768, 10, True)
	# linear = nn.Linear(768, 10, True)
	# torch_Linear = nn.Sequential(
	#     nn.Linear(768, 512),
	#     nn.Linear(512, 512),
	#     nn.Linear(512, 512),
	#     nn.Linear(512, 512),
	#     nn.Linear(512, 512),
	#     nn.Linear(512, 512),
	#     nn.Linear(512, 10),
	# )
	M = 4
	K = 768
	x = torch.randint(0, 2, (M, K)) * 2 - 1
	x = x.to(torch.float32)

	import time 
	
	ms_start = time.time()
	ms_Linear(x)
	ms_end = time.time()

	torch_start = time.time()
	torch_Linear(x)
	torch_end = time.time()
	print("Torch Linear time: ", torch_end - torch_start)
	print("Modelsmith Linear time: ", ms_end - ms_start)
	print("ModelSmith Linear is ", "{:.2f}".format((torch_end - torch_start) / (ms_end - ms_start) ), "x times faster than PyTorch Linear")