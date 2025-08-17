#include <iostream>
#include <torch/torch.h>
#include <chrono>
#include <vector>
#include <functional>
#include <omp.h>
#include <algorithm>
#include <cstring>
#include <immintrin.h>  // For bit manipulation intrinsics

// ARM NEON intrinsics
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// Apple's Accelerate framework
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

// Cache-friendly block sizes (tuned for Apple Silicon)
#define MC 384    // L3 cache block for M
#define NC 256    // L3 cache block for N  
#define KC 256    // L3 cache block for K
#define MR 4      // Micro-panel height
#define NR 8      // Micro-panel width

// Binary GEMM specific constants
#define BITS_PER_INT 32

/**
 * Pack binary matrix A (+1/-1 values) into bit-packed format
 * Each 32-bit integer stores 32 binary values
 */
void pack_binary_matrix_A(uint32_t* A_packed, const float* A, int M, int K) {
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k += BITS_PER_INT) {
            uint32_t packed_bits = 0;
            for (int b = 0; b < BITS_PER_INT && k + b < K; b++) {
                // Convert +1 to 1, -1 to 0
                if (A[i * K + k + b] > 0) {
                    packed_bits |= (1U << b);
                }
            }
            A_packed[i * ((K + BITS_PER_INT - 1) / BITS_PER_INT) + k / BITS_PER_INT] = packed_bits;
        }
    }
}

/**
 * Pack binary matrix B (+1/-1 values) into bit-packed format
 */
void pack_binary_matrix_B(uint32_t* B_packed, const float* B, int K, int N) {
    int K_packed = (K + BITS_PER_INT - 1) / BITS_PER_INT;
    
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k += BITS_PER_INT) {
            uint32_t packed_bits = 0;
            for (int b = 0; b < BITS_PER_INT && k + b < K; b++) {
                // Convert +1 to 1, -1 to 0
                if (B[(k + b) * N + j] > 0) {
                    packed_bits |= (1U << b);
                }
            }
            B_packed[j * K_packed + k / BITS_PER_INT] = packed_bits;
        }
    }
}

/**
 * Verify binary GEMM correctness by comparing with float computation
 */
bool verify_binary_gemm(torch::Tensor A_binary, torch::Tensor B_binary, 
                        torch::Tensor C_binary, torch::Tensor C_float) {
    float max_diff = 0.0f;
    auto diff = torch::abs(C_binary - C_float);
    max_diff = torch::max(diff).item<float>();
    
    std::cout << "Binary GEMM max difference: " << max_diff << std::endl;
    return max_diff < 1e-5f;
}

/**
 * My binary cblas_sgemm using XNOR + POPCOUNT
 * Mimics cblas_sgemm interface but for binary matrices
 */
torch::Tensor binary_cblas_sgemm(torch::Tensor A_binary, torch::Tensor B_binary) {
    int M = A_binary.size(0);
    int K = A_binary.size(1);
    int N = B_binary.size(1);
    
    torch::Tensor C = torch::zeros({M, N}, torch::kFloat32);
    
    float* A_data = A_binary.data_ptr<float>();
    float* B_data = B_binary.data_ptr<float>();
    float* C_data = C.data_ptr<float>();
    
    // Pack matrices into binary format
    int K_packed = (K + BITS_PER_INT - 1) / BITS_PER_INT;
    std::vector<uint32_t> A_packed(M * K_packed);
    std::vector<uint32_t> B_packed(N * K_packed);
    
    pack_binary_matrix_A(A_packed.data(), A_data, M, K);
    pack_binary_matrix_B(B_packed.data(), B_data, K, N);
    
    // Binary matrix multiplication with XNOR + POPCOUNT
    #pragma omp parallel for schedule(static) num_threads(8)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int popcount_sum = 0;
            
            // XNOR + POPCOUNT across all packed integers
            for (int k = 0; k < K_packed; k++) {
                uint32_t a_bits = A_packed[i * K_packed + k];
                uint32_t b_bits = B_packed[j * K_packed + k];
                
                // XNOR operation: ~(a XOR b) 
                uint32_t xnor_result = ~(a_bits ^ b_bits);
                
                // Count number of 1s (matches)
                popcount_sum += __builtin_popcount(xnor_result);
            }
            
            // Convert popcount to binary GEMM result
            // For binary values {+1, -1}, the dot product is: 2*matches - K
            C_data[i * N + j] = 2.0f * popcount_sum - K;
        }
    }
    
    return C;
}

/**
 * Convert float matrix to binary (+1/-1) representation
 */
torch::Tensor float_to_binary(torch::Tensor input) {
    return torch::where(input >= 0, 1.0f, -1.0f);
}

/**
 * Pack matrix A into row-major panels for micro-kernel
 */
void pack_matrix_A(float* A_packed, const float* A, int M, int K, int lda) {
    for (int i = 0; i < M; i += MR) {
        for (int k = 0; k < K; k++) {
            for (int ii = 0; ii < MR && i + ii < M; ii++) {
                A_packed[(i/MR)*KC*MR + k*MR + ii] = A[(i + ii)*lda + k];
            }
            // Pad with zeros if needed
            for (int ii = M - i; ii < MR; ii++) {
                if (i + ii >= M) {
                    A_packed[(i/MR)*KC*MR + k*MR + ii] = 0.0f;
                }
            }
        }
    }
}

/**
 * Pack matrix B into column-major panels for micro-kernel
 */
void pack_matrix_B(float* B_packed, const float* B, int K, int N, int ldb) {
    for (int j = 0; j < N; j += NR) {
        for (int k = 0; k < K; k++) {
            for (int jj = 0; jj < NR && j + jj < N; jj++) {
                B_packed[(j/NR)*KC*NR + k*NR + jj] = B[k*ldb + j + jj];
            }
            // Pad with zeros if needed
            for (int jj = N - j; jj < NR; jj++) {
                if (j + jj >= N) {
                    B_packed[(j/NR)*KC*NR + k*NR + jj] = 0.0f;
                }
            }
        }
    }
}

/**
 * High-performance micro-kernel: C += A * B
 * Computes MR x NR block using packed data
 */
void micro_kernel(const float* A_packed, const float* B_packed, 
                  float* C, int ldc, int k_iter) {
    
#ifdef __ARM_NEON
    // Use NEON for optimal vectorization
    float32x4_t c[MR][NR/4];
    float32x4_t a_vec, b_vec;
    
    // Initialize accumulators
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < NR/4; j++) {
            c[i][j] = vld1q_f32(&C[i*ldc + j*4]);
        }
    }
    
    // Main computation loop
    for (int k = 0; k < k_iter; k++) {
        // Load B vector
        for (int j = 0; j < NR/4; j++) {
            b_vec = vld1q_f32(&B_packed[k*NR + j*4]);
            
            // Multiply with each A element
            for (int i = 0; i < MR; i++) {
                a_vec = vdupq_n_f32(A_packed[k*MR + i]);
                c[i][j] = vmlaq_f32(c[i][j], a_vec, b_vec);
            }
        }
    }
    
    // Store results back
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < NR/4; j++) {
            vst1q_f32(&C[i*ldc + j*4], c[i][j]);
        }
    }
    
#else
    // Fallback scalar implementation with aggressive unrolling
    for (int k = 0; k < k_iter; k++) {
        for (int i = 0; i < MR; i++) {
            float a_ik = A_packed[k*MR + i];
            for (int j = 0; j < NR; j++) {
                C[i*ldc + j] += a_ik * B_packed[k*NR + j];
            }
        }
    }
#endif
}

/**
 * Macro-kernel: handles MC x NC block
 */
void macro_kernel(const float* A, const float* B, float* C,
                  int M, int N, int K, int lda, int ldb, int ldc) {
    
    // Allocate packing buffers
    std::vector<float> A_packed(MC * KC);
    std::vector<float> B_packed(KC * NC);
    
    // Pack A matrix
    pack_matrix_A(A_packed.data(), A, M, K, lda);
    
    for (int j = 0; j < N; j += NR) {
        int n_iter = std::min(NR, N - j);
        
        // Pack B panel
        pack_matrix_B(B_packed.data() + (j/NR)*KC*NR, B, K, n_iter, ldb);
        
        for (int i = 0; i < M; i += MR) {
            int m_iter = std::min(MR, M - i);
            
            // Call micro-kernel
            micro_kernel(A_packed.data() + (i/MR)*KC*MR,
                        B_packed.data() + (j/NR)*KC*NR,
                        &C[i*ldc + j], ldc, K);
        }
    }
}

/**
 * My recreation of cblas_sgemm with key optimizations
 */
torch::Tensor my_cblas_sgemm(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    torch::Tensor C = torch::zeros({M, N}, torch::kFloat32);
    
    float* A_data = A.data_ptr<float>();
    float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();
    
    // Multi-level blocking (L3 cache level)
    #pragma omp parallel for schedule(static) num_threads(8)
    for (int ii = 0; ii < M; ii += MC) {
        for (int jj = 0; jj < N; jj += NC) {
            for (int kk = 0; kk < K; kk += KC) {
                
                int m_iter = std::min(MC, M - ii);
                int n_iter = std::min(NC, N - jj);
                int k_iter = std::min(KC, K - kk);
                
                // Call macro-kernel for this block
                macro_kernel(&A_data[ii*K + kk], &B_data[kk*N + jj], 
                           &C_data[ii*N + jj], m_iter, n_iter, k_iter,
                           K, N, N);
            }
        }
    }
    
    return C;
}

/**
 * Apple Accelerate BLAS - should match PyTorch performance
 */
torch::Tensor accelerate_blas_gemm(torch::Tensor A, torch::Tensor B) {
#ifdef __APPLE__
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    torch::Tensor C = torch::zeros({M, N}, torch::kFloat32);
    
    float* A_data = A.data_ptr<float>();
    float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();
    
    // Use single precision BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f,
                A_data, K,
                B_data, N,
                0.0f,
                C_data, N);
    
    return C;
#else
    // Fallback
    return torch::matmul(A, B);
#endif
}

/**
 * Benchmark function to measure performance
 */
double benchmark_function(std::function<torch::Tensor()> func, int num_trials = 30) {
    // Extended warm up
    for (int i = 0; i < 10; i++) {
        func();
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_trials; i++) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count() / num_trials;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <M> <K> <N>" << std::endl;
        return 1;
    }
    
    int M = std::stoi(argv[1]);
    int K = std::stoi(argv[2]);
    int N = std::stoi(argv[3]);

    // Create random input matrices
    auto A = torch::randn({M, K}, torch::kFloat32);
    auto B = torch::randn({K, N}, torch::kFloat32);
    
    // Create binary versions
    auto A_binary = float_to_binary(A);
    auto B_binary = float_to_binary(B);
    
    int num_trials = 30;
    
    // Benchmark torch::matmul
    double torch_time = benchmark_function([&]() { return torch::matmul(A, B); }, num_trials);
    
    // Benchmark binary torch::matmul
    double torch_binary_time = benchmark_function([&]() { return torch::matmul(A_binary, B_binary); }, num_trials);
    
    #ifdef __APPLE__
    // Test Apple's Accelerate BLAS
    double accelerate_time = benchmark_function([&]() { return accelerate_blas_gemm(A, B); }, num_trials);
    std::cout << "accelerate " << accelerate_time << std::endl;
    #endif
    
    // Test my recreation of cblas_sgemm
    double my_cblas_time = benchmark_function([&]() { return my_cblas_sgemm(A, B); }, num_trials);
    std::cout << "my_cblas " << my_cblas_time << std::endl;
    
    // Test binary cblas_sgemm with XNOR+POPCOUNT
    double binary_cblas_time = benchmark_function([&]() { return binary_cblas_sgemm(A_binary, B_binary); }, num_trials);
    std::cout << "binary_cblas " << binary_cblas_time << std::endl;
    
    // Verify correctness of binary GEMM
    auto C_torch_binary = torch::matmul(A_binary, B_binary);
    auto C_custom_binary = binary_cblas_sgemm(A_binary, B_binary);
    bool is_correct = verify_binary_gemm(A_binary, B_binary, C_custom_binary, C_torch_binary);
    std::cout << "Binary GEMM correctness: " << (is_correct ? "PASSED" : "FAILED") << std::endl;
    
    std::cout << "torch_float " << torch_time << std::endl;
    std::cout << "torch_binary " << torch_binary_time << std::endl;
    
    return 0;
}
