#include <torch/extension.h>
#include <vector>

// TODO: maybe use aligned_alloc for maximal performance 
// Wrapper exposed to Python
torch::Tensor binary_matmul(torch::Tensor, torch::Tensor, int);