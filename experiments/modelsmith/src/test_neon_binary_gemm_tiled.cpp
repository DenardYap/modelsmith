#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

// Declarations from gemm.cpp
void neon_binary_gemm_tiled_with_params(const uint8_t* A_packed,
                                       const uint8_t* B_packed,
                                       int32_t* C,
                                       int M, int K, int N,
                                       int tile_m, int tile_n, int tile_k);

// Helper: generate {-1, +1} matrix stored as int8_t
static std::vector<int8_t> generate_binary_matrix(int rows, int cols, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, 1);
    std::vector<int8_t> mat(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int bit = dist(rng);
            mat[i * cols + j] = static_cast<int8_t>(bit ? 1 : -1);
        }
    }
    return mat;
}

// Pack a matrix of shape (rows x K) where values are in {-1,+1} into bytes.
// We map +1 -> bit 1, -1 -> bit 0. Bits are packed MSB-first per byte.
static std::vector<uint8_t> pack_rows_bits_msb_first(const std::vector<int8_t>& mat,
                                                     int rows, int K) {
    const int packed_K = (K + 7) / 8;
    std::vector<uint8_t> packed(rows * packed_K, 0);
    for (int r = 0; r < rows; ++r) {
        for (int k = 0; k < K; ++k) {
            const int byte_index = k / 8;
            const int bit_in_byte = 7 - (k % 8); // MSB-first
            const int8_t v = mat[r * K + k];
            const uint8_t bit = (v > 0) ? 1u : 0u;
            packed[r * packed_K + byte_index] |= (bit << bit_in_byte);
        }
    }
    return packed;
}

// For B of shape (K x N), we need B_packed as (N x packed_K), i.e., pack by columns.
static std::vector<uint8_t> pack_cols_bits_msb_first(const std::vector<int8_t>& mat,
                                                     int K, int N) {
    const int packed_K = (K + 7) / 8;
    std::vector<uint8_t> packed(N * packed_K, 0);
    for (int c = 0; c < N; ++c) {
        for (int k = 0; k < K; ++k) {
            const int byte_index = k / 8;
            const int bit_in_byte = 7 - (k % 8); // MSB-first
            const int8_t v = mat[k * N + c]; // column access
            const uint8_t bit = (v > 0) ? 1u : 0u;
            packed[c * packed_K + byte_index] |= (bit << bit_in_byte);
        }
    }
    return packed;
}

// Reference GEMM for {-1,+1} matrices: C = A * B, A(MxK), B(KxN), C(MxN)
static std::vector<int32_t> reference_gemm_pm1(const std::vector<int8_t>& A,
                                               const std::vector<int8_t>& B,
                                               int M, int K, int N) {
    std::vector<int32_t> C(M * N, 0);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(A[i * K + k]) * static_cast<int32_t>(B[k * N + j]);
            }
            C[i * N + j] = acc;
        }
    }
    return C;
}

static void run_case(int M, int K, int N, int tile_m, int tile_n, int tile_k, std::mt19937& rng) {
    std::cout << "Case M=" << M << " K=" << K << " N=" << N << std::endl;
    // Generate A (MxK) and B (KxN) in {-1,+1}
    std::vector<int8_t> A = generate_binary_matrix(M, K, rng);
    std::vector<int8_t> B = generate_binary_matrix(K, N, rng);

    // Pack
    const int packed_K = (K + 7) / 8;
    std::vector<uint8_t> A_packed = pack_rows_bits_msb_first(A, M, K);
    std::vector<uint8_t> B_packed = pack_cols_bits_msb_first(B, K, N);

    // Compute using kernel
    std::vector<int32_t> C_kernel(M * N, 0);
    neon_binary_gemm_tiled_with_params(A_packed.data(), B_packed.data(), C_kernel.data(),
                                       M, K, N, tile_m, tile_n, tile_k);

    // Reference
    const std::vector<int32_t> C_ref = reference_gemm_pm1(A, B, M, K, N);

    // Compare
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t a = C_kernel[i * N + j];
            int32_t b = C_ref[i * N + j];
            if (a != b) {
                std::cerr << "Mismatch at (" << i << "," << j << ") got " << a
                          << " expected " << b << std::endl;
                std::exit(1);
            }
        }
    }
    std::cout << "  PASSED" << std::endl;
}

int main() {
    std::mt19937 rng(0);

    // Test a variety of shapes, including K not multiple of 8
    const int tile_m = 16;
    const int tile_n = 128;
    const int tile_k = 128;

    struct S { int M, K, N; };
    std::vector<S> cases = {
        {1, 1, 1},
        {2, 5, 3},
        {3, 9, 4},
        {4, 15, 8},
        {4, 32, 8},
        {8, 63, 16},
        {8, 64, 16},
        {8, 65, 16},
        {8, 127, 16},
        {8, 128, 16},
        {8, 129, 16},
        {16, 256, 32}
    };

    for (const auto& c : cases) {
        run_case(c.M, c.K, c.N, tile_m, tile_n, tile_k, rng);
    }

    std::cout << "All tests passed." << std::endl;
    return 0;
}


