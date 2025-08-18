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

// Kernel declaration from gemm.cpp
void neon_binary_gemm_tiled_with_params(const uint8_t* A_packed,
                                       const uint8_t* B_packed,
                                       int32_t* C,
                                       int M, int K, int N,
                                       int tile_m, int tile_n, int tile_k);

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

// Pack A (MxK) by rows into bytes; +1->1, -1->0, MSB-first
static std::vector<uint8_t> pack_rows_bits_msb_first(const std::vector<int8_t>& A, int M, int K) {
    const int packed_K = (K + 7) / 8;
    std::vector<uint8_t> packed(M * packed_K, 0);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            const int byte_index = k / 8;
            const int bit_in_byte = 7 - (k % 8);
            const uint8_t bit = (A[i * K + k] > 0) ? 1u : 0u;
            packed[i * packed_K + byte_index] |= (bit << bit_in_byte);
        }
    }
    return packed;
}

// Pack B (KxN) by columns into bytes; +1->1, -1->0, MSB-first
static std::vector<uint8_t> pack_cols_bits_msb_first(const std::vector<int8_t>& B, int K, int N) {
    const int packed_K = (K + 7) / 8;
    std::vector<uint8_t> packed(N * packed_K, 0);
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) {
            const int byte_index = k / 8;
            const int bit_in_byte = 7 - (k % 8);
            const uint8_t bit = (B[k * N + j] > 0) ? 1u : 0u;
            packed[j * packed_K + byte_index] |= (bit << bit_in_byte);
        }
    }
    return packed;
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
    std::vector<float> A_f(M * K), B_f(K * N);
    for (int i = 0; i < M * K; ++i) A_f[i] = static_cast<float>(A_pm1[i]);
    for (int i = 0; i < K * N; ++i) B_f[i] = static_cast<float>(B_pm1[i]);

    torch::Tensor A_t = torch::from_blob(A_f.data(), {M, K}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    torch::Tensor B_t = torch::from_blob(B_f.data(), {K, N}, torch::TensorOptions().dtype(torch::kFloat32)).clone();

    // Time PyTorch matmul
    auto torch_call = [&]() {
        auto C = torch::mm(A_t, B_t);
        (void)C;
    };
    double torch_ms = time_ms(torch_call, 20);

    // Prepare packed inputs once
    std::vector<uint8_t> A_packed = pack_rows_bits_msb_first(A_pm1, M, K);
    std::vector<uint8_t> B_packed = pack_cols_bits_msb_first(B_pm1, K, N);
    std::vector<int32_t> C_kernel(M * N);

    // Candidate tile sizes
    std::vector<int> tiles = {1, 2, 4, 8, 16, 64, 256};

    // Measure all combinations and keep top-3 fastest
    struct Result { double ms; int tm, tn, tk; };
    std::vector<Result> results;
    results.reserve(tiles.size() * tiles.size() * tiles.size());

    for (int tm : tiles) {
        for (int tn : tiles) {
            for (int tk : tiles) {
                auto kernel_call = [&]() {
                    neon_binary_gemm_tiled_with_params(A_packed.data(), B_packed.data(), C_kernel.data(),
                                                       M, K, N, tm, tn, tk);
                };
                double ms = time_ms(kernel_call, 20);
                results.push_back({ms, tm, tn, tk});
            }
        }
    }

    std::sort(results.begin(), results.end(), [](const Result& a, const Result& b){ return a.ms < b.ms; });

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "PyTorch mm: " << torch_ms << " ms (avg over 20)\n";
    std::cout << "Top 3 kernel tile combinations (avg over 20):\n";
    for (int i = 0; i < 3 && i < (int)results.size(); ++i) {
        const auto& r = results[i];
        std::cout << "  tiles (tm, tn, tk) = (" << r.tm << ", " << r.tn << ", " << r.tk << ") : "
                  << r.ms << " ms\n";
    }

    return 0;
}


