#include <iostream>
#include <torch/torch.h>
#include <cassert>
#include <chrono>
#include <vector>

// Forward declarations
torch::Tensor pack_binary_matrix_SIMD128(torch::Tensor mat);
void neon_binary_gemm_tiled_with_params(const uint8_t* A_packed,
                      const uint8_t* B_packed,
                      int32_t* C,
                                       int M, int K, int N,
                                       int tile_m, int tile_n, int tile_k);

void test_tiled_gemm_correctness(int M, int K, int N, 
                                int tile_m, int tile_n, int tile_k,
                                const std::string& test_name = "normal") {
    
    std::cout << "Testing " << test_name << ": " << M << "x" << K << " * " << K << "x" << N 
              << " with tiles(" << tile_m << "," << tile_n << "," << tile_k << ")" << std::endl;

    // Set seed for reproducible results
    torch::manual_seed(42 + M + K + N);

    // Generate random binary matrices (-1, 1)
    auto A = (torch::randint(0, 2, {M, K}, torch::kFloat32) * 2 - 1).to(torch::kFloat32);
    auto B = (torch::randint(0, 2, {K, N}, torch::kFloat32) * 2 - 1).to(torch::kFloat32);

    // Convert to int8 and pack
    auto A_int8 = A.to(torch::kInt8).contiguous();
    auto B_int8 = B.to(torch::kInt8).contiguous();
    auto B_t_int8 = B_int8.transpose(0, 1).contiguous();
    
    auto A_packed = pack_binary_matrix_SIMD128(A_int8);
    auto B_packed = pack_binary_matrix_SIMD128(B_t_int8);

    // Run our tiled kernel
    auto start = std::chrono::high_resolution_clock::now();
    auto C_tiled = torch::zeros({M, N}, torch::kInt32);
    
    neon_binary_gemm_tiled_with_params(
        A_packed.data_ptr<uint8_t>(),
        B_packed.data_ptr<uint8_t>(),
        C_tiled.data_ptr<int32_t>(),
        M, K, N,
        tile_m, tile_n, tile_k
    );
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_tiled = std::chrono::duration<double, std::milli>(end - start);

    // Run PyTorch reference
    start = std::chrono::high_resolution_clock::now();
    auto C_ref = torch::matmul(A, B);
    end = std::chrono::high_resolution_clock::now();
    auto elapsed_torch = std::chrono::duration<double, std::milli>(end - start);

    // Check correctness
    auto C_tiled_float = C_tiled.to(torch::kFloat32);
    auto C_ref_float = C_ref.to(torch::kFloat32);

    // Verify shapes match
    if (C_tiled.sizes() != C_ref.sizes()) {
        std::cerr << "❌ Shape mismatch! Tiled: " << C_tiled.sizes() 
                  << ", Reference: " << C_ref.sizes() << std::endl;
        throw std::runtime_error("Shape mismatch");
    }

    // Verify values match
    if (!torch::allclose(C_tiled_float, C_ref_float, 1e-6, 1e-6)) {
        std::cerr << "❌ Value mismatch!" << std::endl;
        
        // Find first difference for debugging
        auto diff = torch::abs(C_tiled_float - C_ref_float);
        auto max_diff = torch::max(diff);
        auto indices = torch::where(diff == max_diff);
        
        if (indices[0].numel() > 0) {
            int i = indices[0][0].item<int>();
            int j = indices[1][0].item<int>();
            std::cerr << "Max difference at (" << i << "," << j << "): "
                      << "Tiled=" << C_tiled_float[i][j].item<float>()
                      << ", Reference=" << C_ref_float[i][j].item<float>()
                      << ", Diff=" << max_diff.item<float>() << std::endl;
        }
        throw std::runtime_error("Value mismatch");
    }

    std::cout << "✅ PASSED - Tiled: " << elapsed_tiled.count() 
              << "ms, PyTorch: " << elapsed_torch.count() << "ms";
    
    if (elapsed_tiled.count() < elapsed_torch.count()) {
        std::cout << " (🚀 " << (elapsed_torch.count() / elapsed_tiled.count()) << "x faster!)";
    }
    std::cout << std::endl << std::endl;
}

void test_edge_cases() {
    std::cout << "=== Testing Edge Cases ===" << std::endl;
    
    // Test various edge cases with different tile sizes
    std::vector<std::tuple<int, int, int, int, int, int, std::string>> test_cases = {
        // M, K, N, tile_m, tile_n, tile_k, name
        {1, 1, 1, 1, 1, 1, "1x1x1"},
        {1, 128, 1, 1, 1, 128, "skinny_vector"},
        {128, 1, 128, 16, 16, 1, "tall_vector"},
        {1, 1024, 1024, 1, 64, 256, "1x1024x1024"},
        {7, 13, 5, 4, 2, 8, "odd_dimensions"},
        {15, 17, 19, 8, 4, 16, "prime_like_dimensions"},
        {128, 129, 130, 32, 32, 64, "near_power_of_2"},
        {0, 0, 0, 1, 1, 1, "zero_dimensions"},
    };

    for (auto& test_case : test_cases) {
        auto [M, K, N, tm, tn, tk, name] = test_case;
        
        // Skip zero dimension tests as they would cause issues
        if (M == 0 || K == 0 || N == 0) continue;
        
        try {
            test_tiled_gemm_correctness(M, K, N, tm, tn, tk, name);
        } catch (const std::exception& e) {
            std::cerr << "❌ FAILED test " << name << ": " << e.what() << std::endl;
            throw;
        }
    }
}

void test_various_tile_sizes() {
    std::cout << "=== Testing Various Tile Sizes ===" << std::endl;
    
    // Test same matrix with different tile configurations
    int M = 256, K = 512, N = 128;
    
    std::vector<std::tuple<int, int, int, std::string>> tile_configs = {
        {1, 1, 1, "minimal_tiles"},
        {2, 2, 2, "small_tiles"},
        {4, 4, 8, "small_tiles_2"},
        {16, 16, 64, "medium_tiles"},
        {32, 32, 128, "large_tiles"},
        {64, 64, 256, "very_large_tiles"},
        {M, N, K, "single_tile"},  // Entire matrix as one tile
        {1, N, K, "row_tiles"},    // Process one row at a time
        {M, 1, K, "col_tiles"},    // Process one col at a time
    };

    for (auto& [tm, tn, tk, name] : tile_configs) {
        try {
            test_tiled_gemm_correctness(M, K, N, tm, tn, tk, name);
        } catch (const std::exception& e) {
            std::cerr << "❌ FAILED tile config " << name << ": " << e.what() << std::endl;
            throw;
        }
    }
}

void test_common_nn_shapes() {
    std::cout << "=== Testing Common Neural Network Shapes ===" << std::endl;
    
    std::vector<std::tuple<int, int, int, int, int, int, std::string>> nn_cases = {
        // Common transformer/LLM shapes with good tile sizes
        {1024, 768, 1, 64, 1, 256, "transformer_projection"},
        {2048, 1024, 4096, 128, 256, 128, "large_mlp"},
        {512, 512, 512, 64, 64, 128, "square_medium"},
        {4096, 4096, 1, 256, 1, 256, "large_to_scalar"},
        {1, 4096, 4096, 1, 128, 256, "batch_1_large"},
        {768, 3072, 768, 64, 256, 128, "transformer_ffn"},
        {1024, 1024, 1024, 128, 128, 256, "large_square"},
    };

    for (auto& test_case : nn_cases) {
        auto [M, K, N, tm, tn, tk, name] = test_case;
        try {
            test_tiled_gemm_correctness(M, K, N, tm, tn, tk, name);
        } catch (const std::exception& e) {
            std::cerr << "❌ FAILED NN shape " << name << ": " << e.what() << std::endl;
            throw;
        }
    }
}

void test_special_patterns() {
    std::cout << "=== Testing Special Value Patterns ===" << std::endl;
    
    // Test with all ones
    torch::manual_seed(1337);
    int M = 64, K = 128, N = 32;
    int tm = 16, tn = 8, tk = 64;
    
    auto A_ones = torch::ones({M, K}, torch::kFloat32);
    auto B_ones = torch::ones({K, N}, torch::kFloat32);
    
    auto A_int8 = A_ones.to(torch::kInt8).contiguous();
    auto B_int8 = B_ones.to(torch::kInt8).contiguous();
    auto B_t_int8 = B_int8.transpose(0, 1).contiguous();
    
    auto A_packed = pack_binary_matrix_SIMD128(A_int8);
    auto B_packed = pack_binary_matrix_SIMD128(B_t_int8);
    
    auto C_tiled = torch::zeros({M, N}, torch::kInt32);
    neon_binary_gemm_tiled_with_params(
        A_packed.data_ptr<uint8_t>(),
        B_packed.data_ptr<uint8_t>(),
        C_tiled.data_ptr<int32_t>(),
        M, K, N, tm, tn, tk
    );
    
    auto C_ref = torch::matmul(A_ones, B_ones);
    auto C_tiled_float = C_tiled.to(torch::kFloat32);
    
    if (!torch::allclose(C_tiled_float, C_ref, 1e-6, 1e-6)) {
        throw std::runtime_error("All-ones test failed");
    }
    std::cout << "✅ All-ones pattern test passed" << std::endl;
    
    // Test with all negative ones
    auto A_neg = -torch::ones({M, K}, torch::kFloat32);
    auto B_neg = -torch::ones({K, N}, torch::kFloat32);
    
    A_int8 = A_neg.to(torch::kInt8).contiguous();
    B_int8 = B_neg.to(torch::kInt8).contiguous();
    B_t_int8 = B_int8.transpose(0, 1).contiguous();
    
    A_packed = pack_binary_matrix_SIMD128(A_int8);
    B_packed = pack_binary_matrix_SIMD128(B_t_int8);
    
    C_tiled = torch::zeros({M, N}, torch::kInt32);
    neon_binary_gemm_tiled_with_params(
        A_packed.data_ptr<uint8_t>(),
        B_packed.data_ptr<uint8_t>(),
        C_tiled.data_ptr<int32_t>(),
        M, K, N, tm, tn, tk
    );
    
    C_ref = torch::matmul(A_neg, B_neg);
    C_tiled_float = C_tiled.to(torch::kFloat32);
    
    if (!torch::allclose(C_tiled_float, C_ref, 1e-6, 1e-6)) {
        throw std::runtime_error("All-negative-ones test failed");
    }
    std::cout << "✅ All-negative-ones pattern test passed" << std::endl << std::endl;
}

int main() {
    std::cout << "🧪 Testing neon_binary_gemm_tiled_with_params" << std::endl;
    std::cout << "=============================================" << std::endl << std::endl;

    try {
        test_edge_cases();
        test_various_tile_sizes();  
        test_common_nn_shapes();
        test_special_patterns();
        
        std::cout << "🎉 ALL TESTS PASSED! 🎉" << std::endl;
        std::cout << "The tiled binary GEMM implementation is working correctly." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "💥 TEST SUITE FAILED: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}