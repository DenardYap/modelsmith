import sys
import os
import torch

# Add paths to find either the built extension or the package
ROOT = "/Users/bernardyap/Desktop/ModelSmith/experiments/modelsmith/src"
BUILD_LIB = os.path.join(ROOT, "build", "lib.macosx-15.4-arm64-cpython-311")
BUILD_LIB_313 = os.path.join(ROOT, "build", "lib.macosx-15.0-arm64-cpython-313")
if BUILD_LIB_313 not in sys.path:
	sys.path.insert(0, BUILD_LIB_313)
if BUILD_LIB not in sys.path:
	sys.path.insert(0, BUILD_LIB)
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)
import modelsmith

# Resolve function handle, prefer tiled
if hasattr(modelsmith, "binary_matmul_tiled"):
	binary_fn = modelsmith.binary_matmul_tiled
elif hasattr(modelsmith, "_C") and hasattr(modelsmith._C, "binary_matmul_tiled"):
	binary_fn = modelsmith._C.binary_matmul_tiled
else:
	raise AttributeError("modelsmith.binary_matmul_tiled not found (tried top-level and modelsmith._C)")

# Test a variety of matrix sizes, including non-multiples of 8
sizes = [
	(1, 1, 1),
	(2, 3, 4),
	(5, 7, 9),
	(8, 8, 8),
	(9, 9, 9),
	(16, 15, 17),
	(31, 33, 35),
	(64, 127, 32),
	(128, 129, 64),
]

all_ok = True
for (M, K, N) in sizes:
	torch.manual_seed(0)
	A = (torch.randint(0, 2, (M, K), dtype=torch.int8) * 2 - 1).to(torch.float32)
	B = (torch.randint(0, 2, (K, N), dtype=torch.int8) * 2 - 1).to(torch.float32)
	C_ms = binary_fn(A, B)
	C_ref = (A @ B).to(torch.int32)
	ok = torch.equal(C_ms, C_ref)
	max_diff = (C_ms - C_ref).abs().max().item()
	print(f"M={M}, K={K}, N={N}: ok={ok}, max_diff={max_diff}")
	all_ok = all_ok and ok

print(f"ALL_OK={all_ok}") 