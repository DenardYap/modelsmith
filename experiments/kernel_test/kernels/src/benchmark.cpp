#include <iostream>
#include <torch/torch.h>
#include <chrono>
/**
 * Immediate to do:
 * 1. Align memory
 * 2. Eliminate need to transpose, and loop reordering
 * 3. 
 * 4. 
 */

/**
 * Summary of things to try:
 * 
 * 1. Tiling only vs Tranpose only 
 * 2. Figure out a way to eliminate or improve casting + bitpacking 
 * 3. Figure out why neon_gemm is so slow for bigger matrices
 * 4. Pack matrix / align to memory if it's the first time 
 * 5. Implement tiling
 * 6. Unrolling
 * 7. In row-major programming languages like C/C++, having k as the innest loop is bad.
 */

// Forward declarations of packing and kernel entry
torch::Tensor pack_binary_matrix_SIMD128(torch::Tensor mat);
void neon_binary_gemm_tiled_with_params(const uint8_t* A_packed,
                                        const uint8_t* B_packed,
                                        int32_t* C,
                                        int M, int K, int N,
                                        int tile_m, int tile_n, int tile_k);

torch::Tensor binary_matmul_tiled_with_params(torch::Tensor A_bin, torch::Tensor B_bin);

int main(int argc, char* argv[]) {
    int M = std::stoi(argv[1]);
    int K = std::stoi(argv[2]);
    int N = std::stoi(argv[3]);

    std::cout << "M: " << M << ", K: " << K << ", N: " << N << std::endl;
    auto A = (torch::randint(0, 2, {M, K}, torch::kFloat32) * 2 - 1).to(torch::kFloat32);
    auto B = (torch::randint(0, 2, {K, N}, torch::kFloat32) * 2 - 1).to(torch::kFloat32);
    
    // Pre-pack once to isolate kernel performance
    auto A_i8 = A.to(torch::kInt8).contiguous();
    auto B_i8 = B.to(torch::kInt8).contiguous();
    auto B_t_i8 = B_i8.transpose(0, 1).contiguous();
    auto A_packed = pack_binary_matrix_SIMD128(A_i8);
    auto B_packed = pack_binary_matrix_SIMD128(B_t_i8);

    // Dry run for 5 loops
    for (int i = 0; i < 5; i++){
        auto out = torch::zeros({M, N}, torch::kInt32);
        neon_binary_gemm_tiled_with_params(A_packed.data_ptr<uint8_t>(),
                                           B_packed.data_ptr<uint8_t>(),
                                           out.data_ptr<int32_t>(),
                                           M, K, N,
                                           16, 16, 128);
        torch::matmul(A, B);
    }
    
    double time_torch = 0;

    int TILE_Ms[] = {1, 2, 4, 16, 64, 128, 256};
    int TILE_Ns[] = {1, 2, 4, 16, 64, 128, 256};
    int TILE_Ks[] = {1, 2, 4, 16, 64, 128, 256};
    int avg_trial = 10;
    float min_time = 1000000;
    int min_tile_m = 0;
    int min_tile_n = 0;
    int min_tile_k = 0;
    // First benchmark torch::matmul for reference
    for (int i = 0; i < avg_trial; i++){
        auto torch_start = std::chrono::high_resolution_clock::now();
        torch::matmul(A, B);
        auto torch_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> torch_duration = torch_end - torch_start;
        time_torch += torch_duration.count();
    }

    std::cout << "========== TILE SIZE VARIANTS (Kernel only, pre-packed) ==========" << std::endl;
    
    // Triple nested loop to test all tile size combinations
    for (int m_idx = 0; m_idx < 7; m_idx++) {
        for (int n_idx = 0; n_idx < 7; n_idx++) {
            for (int k_idx = 0; k_idx < 7; k_idx++) {
                int tile_m = TILE_Ms[m_idx];
                int tile_n = TILE_Ns[n_idx];    
                int tile_k = TILE_Ks[k_idx];
                
                double variant_time = 0;

                // Average over avg_trial runs
                for (int trial = 0; trial < avg_trial; trial++) {
                    auto out = torch::zeros({M, N}, torch::kInt32);
                    auto start = std::chrono::high_resolution_clock::now();
                    auto start2 = std::chrono::high_resolution_clock::now();
                    auto start3 = std::chrono::high_resolution_clock::now();
                    A_i8 = A.to(torch::kInt8).contiguous();
                    B_i8 = B.to(torch::kInt8).contiguous();
                    B_t_i8 = B_i8.transpose(0, 1).contiguous();
                    auto end2 = std::chrono::high_resolution_clock::now();

                    A_packed = pack_binary_matrix_SIMD128(A_i8);
                    B_packed = pack_binary_matrix_SIMD128(B_t_i8);
                    auto end3 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> duration2 = end2 - start2;
                    std::chrono::duration<double, std::milli> duration3 = end3 - start3;
                    std::cout << "Conversion time 2: " << duration2.count() << " ms" << std::endl;
                    std::cout << "Packing time 3: " << duration3.count() << " ms" << std::endl;
                    neon_binary_gemm_tiled_with_params(A_packed.data_ptr<uint8_t>(),
                                                       B_packed.data_ptr<uint8_t>(),
                                                       out.data_ptr<int32_t>(),
                                                       M, K, N,
                                                       tile_m, tile_n, tile_k);
                    auto end = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> duration = end - start;
                    variant_time += duration.count();
                }
                
                double avg_time = variant_time / avg_trial;
                if (avg_time < min_time) {
                    min_time = avg_time;
                    min_tile_m = tile_m;
                    min_tile_n = tile_n;
                    min_tile_k = tile_k;
                }
                std::cout << "TILE_M=" << tile_m << ", TILE_N=" << tile_n << ", TILE_K=" << tile_k 
                          << " | M=" << M << ", K=" << K << ", N=" << N 
                          << " | Time: " << avg_time << " ms" << std::endl;
            }
        }
    }
    std::cout << "torch::matmul: " << time_torch / avg_trial << " ms" << std::endl;
    std::cout << "Best tile size: TILE_M=" << min_tile_m << ", TILE_N=" << min_tile_n << ", TILE_K=" << min_tile_k << " | Time: " << min_time << " ms" << std::endl;

}

/**
 * 
 * Bitpack and casting are the bottleneck for binary_matmul128 when the input is small (e.g. 1024 x 1024 vs 1024 x 1)
 * Neon SIMD is the bottleneck when the input is large (e.g. 1024 x 1024 vs 1024 x 1024)
 * 
 * Solutions:
 * 
 * 1. Have the matrix weights convert to -1 and 1 after training, in this way we can eliminate the casting 
 * Q: Can we bitpack the weight after inference? 
 * A: Prob yes, then during the forward loop, we need a flag (e.g. self.training = False) to know if we are 
 *    in inference mode, if so, we can call neon_binary_gemm to perform optimized bitpacked-aware binary 
 *    matrix multiplication
 * 
 * e.g. during training W = [0.3, -0.6, 9.3,
 *                           1.2, 3.3, -1.2,
 *                           0, -8, 22]
 * After training convert W = W_bin = [1, -1, 1,
 *                                     1, 1, -1,
 *                                     1, -1, 1] 
 * 
 * This works well for like transformers (e.g. QKV are all learned matrices), but for linear layer, the input x 
 * is not already casted + bitpacked, so we would still need to do it in the forward loop
 * 
 * 
 * Then convert W_bin to W_bitpacked = [1010000, 11000000, 10100000]
 * 
 * Look into AWQ
 */ 