import torch
import modelsmith
import time
import numpy as np

def benchmark_matmul(M, K, N, num_runs=10):
    # Create random binary tensors (-1 or 1)
    A = torch.randint(0, 2, (M, K), dtype=torch.float32) * 2 - 1
    B = torch.randint(0, 2, (K, N), dtype=torch.float32) * 2 - 1
    
    # Warm-up run
    _ = modelsmith.binary_matmul(A, B)
    _ = modelsmith.binary_matmul_tiled(A, B)
    _ = torch.matmul(A, B)
    
    times = {'binary': [], 'tiled': [], 'torch': []}
    
    for _ in range(num_runs):
        # Regular binary matmul
        start = time.perf_counter()
        _ = modelsmith.binary_matmul(A, B)
        end = time.perf_counter()
        times['binary'].append(end - start)
        
        # Tiled binary matmul
        start = time.perf_counter()
        _ = modelsmith.binary_matmul_tiled(A, B)
        end = time.perf_counter()
        times['tiled'].append(end - start)
        
        # PyTorch matmul
        start = time.perf_counter()
        _ = torch.matmul(A, B)
        end = time.perf_counter()
        times['torch'].append(end - start)
    
    # Calculate mean and std
    results = {}
    for key in times:
        mean = np.mean(times[key]) * 1000  # Convert to milliseconds
        std = np.std(times[key]) * 1000
        results[key] = (mean, std)
    
    return results

# Test different matrix sizes
sizes = [
    (128, 128, 128),    # Small square matrices
    (256, 256, 256),    # Medium square matrices
    (512, 512, 512),    # Large square matrices
    (1024, 1024, 1024), # Very large square matrices
    (128, 512, 128),    # Tall-skinny * fat-skinny
    (512, 128, 512),    # Fat-skinny * tall-skinny
    (1024, 1024, 1),    # Fat-skinny * tall-skinny
]

print("Benchmarking matrix multiplication methods...")
print("\nFormat: mean_time_ms ± std_ms")
print("\nMatrix Sizes (M×K * K×N = M×N) | Regular Binary | Tiled Binary | PyTorch")
print("-" * 75)

for M, K, N in sizes:
    results = benchmark_matmul(M, K, N)
    size_str = f"{M}×{K} * {K}×{N} = {M}×{N}"
    binary_str = f"{results['binary'][0]:.2f} ± {results['binary'][1]:.2f}"
    tiled_str = f"{results['tiled'][0]:.2f} ± {results['tiled'][1]:.2f}"
    torch_str = f"{results['torch'][0]:.2f} ± {results['torch'][1]:.2f}"
    
    print(f"{size_str:<28} | {binary_str:<13} | {tiled_str:<12} | {torch_str}")

# Calculate and print speedups
print("\nSpeedups relative to PyTorch:")
print("\nMatrix Sizes (M×K * K×N) | Regular Binary | Tiled Binary")
print("-" * 60)

for M, K, N in sizes:
    results = benchmark_matmul(M, K, N)
    size_str = f"{M}×{K} * {K}×{N}"
    binary_speedup = results['torch'][0] / results['binary'][0]
    tiled_speedup = results['torch'][0] / results['tiled'][0]
    
    print(f"{size_str:<24} | {binary_speedup:>8.2f}x     | {tiled_speedup:>8.2f}x") 