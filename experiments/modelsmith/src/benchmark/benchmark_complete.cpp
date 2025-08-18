#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <torch/torch.h>

// Declare the wrapper implemented in binary_matmul.cpp
torch::Tensor binary_matmul_tiled_with_params(torch::Tensor A_bin,
                                              torch::Tensor B_bin);

static std::vector<int8_t> generate_binary_pm1(int rows, int cols, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, 1);
    std::vector<int8_t> mat(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i * cols + j] = static_cast<int8_t>(dist(rng) ? 1 : -1);
        }
    }
    return mat;
}

static double time_ms(std::function<void()> fn, int iters) {
    using clock = std::chrono::high_resolution_clock;
    // Warmup
    for (int i = 0; i < std::max(1, iters / 5); ++i) fn();
    auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto t1 = clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " M N K\n";
        return 1;
    }
    const int M = std::atoi(argv[1]);
    const int N = std::atoi(argv[2]);
    const int K = std::atoi(argv[3]);

    std::mt19937 rng(0);

    // Generate data in {-1,+1}
    std::vector<int8_t> A_pm1 = generate_binary_pm1(M, K, rng);
    std::vector<int8_t> B_pm1 = generate_binary_pm1(K, N, rng);

    // Prepare tensors for PyTorch
    // Baseline matmul expects float32
    std::vector<float> A_f(M * K), B_f(K * N);
    for (int i = 0; i < M * K; ++i) A_f[i] = static_cast<float>(A_pm1[i]);
    for (int i = 0; i < K * N; ++i) B_f[i] = static_cast<float>(B_pm1[i]);

    torch::Tensor A_f_t = torch::from_blob(A_f.data(), {M, K}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    torch::Tensor B_f_t = torch::from_blob(B_f.data(), {K, N}, torch::TensorOptions().dtype(torch::kFloat32)).clone();

    // For the binary wrapper, we pass float32 tensors with values in {-1,+1}
    // The wrapper will convert to int8 internally as needed.

    // Time PyTorch mm
    auto torch_call = [&]() {
        auto C = torch::mm(A_f_t, B_f_t);
        (void)C;
    };
    double torch_ms = time_ms(torch_call, 20);

    // Time binary_matmul_tiled_with_params (does its own packing + tiled kernel)
    auto wrapper_call = [&]() {
        auto C_bin = binary_matmul_tiled_with_params(A_f_t, B_f_t);
        (void)C_bin;
    };
    double wrapper_ms = time_ms(wrapper_call, 20);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "PyTorch mm: " << torch_ms << " ms (avg over 20)\n";
    std::cout << "binary_matmul_tiled_with_params: " << wrapper_ms << " ms (avg over 20)\n";

    return 0;
}


