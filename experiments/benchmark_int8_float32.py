import numpy as np
import time

# Define matrix size
size = (5120, 5120)

# Generate float32 matrices A and B
A = np.random.rand(*size).astype(np.float32)
B = np.random.rand(*size).astype(np.float32)

# Generate int8 matrices C and D
C = np.random.randint(-128, 127, size=size, dtype=np.int8)
D = np.random.randint(-128, 127, size=size, dtype=np.int8)

# Benchmark float32 addition
start = time.time()
float_add = A + B
float_add_time = (time.time() - start) * 1000

# Benchmark float32 multiplication
start = time.time()
float_mul = A * B
float_mul_time = (time.time() - start) * 1000

# Benchmark int8 addition
start = time.time()
int_add = C + D
int_add_time = (time.time() - start) * 1000

# Benchmark int8 multiplication
start = time.time()
int_mul = C * D
int_mul_time = (time.time() - start) * 1000

# Print results
print(f"float32 addition time: {float_add_time:.6f} ms")
print(f"float32 multiplication time: {float_mul_time:.6f} ms")
print(f"int8 addition time: {int_add_time:.6f} ms")
print(f"int8 multiplication time: {int_mul_time:.6f} ms")
