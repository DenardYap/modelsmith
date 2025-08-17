import torch
import modelsmith

# Set random seed for reproducibility
torch.manual_seed(42)

# Create random binary tensors (-1 or 1)
M, K, N = 128, 256, 64  # Example dimensions
A = torch.randint(0, 2, (M, K), dtype=torch.float32) * 2 - 1  # Convert 0,1 to -1,1
B = torch.randint(0, 2, (K, N), dtype=torch.float32) * 2 - 1

print("Input shapes:")
print(f"A: {A.shape}")
print(f"B: {B.shape}")

# Test regular binary matmul
result1 = modelsmith.binary_matmul(A, B)
print("\nRegular binary matmul output shape:", result1.shape)
print("Output sample (top-left 3x3):")
print(result1[:3, :3])

# Test tiled binary matmul
result2 = modelsmith.binary_matmul_tiled(A, B)
print("\nTiled binary matmul output shape:", result2.shape)
print("Output sample (top-left 3x3):")
print(result2[:3, :3])

# Verify results match
max_diff = torch.max(torch.abs(result1 - result2))
print("\nMaximum difference between regular and tiled results:", max_diff.item())
print("Results match:", torch.allclose(result1, result2)) 