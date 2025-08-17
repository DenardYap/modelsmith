#include <iostream>
#include "../src/kernel_helper.h"
#include "../src/gemm.h"
#include <torch/torch.h>
#include <iostream>
#include <cassert>
#include <bitset>
#include <chrono>
#include <fstream>
#include <string>

// Forward declaration
torch::Tensor pack_binary_matrix(torch::Tensor mat);
torch::Tensor pack_binary_matrix_SIMD128(torch::Tensor mat);
void neon_binary_gemm_improved(const uint8_t* A_packed,
                      const uint8_t* B_packed,
                      int32_t* C,
                      int M, int K, int N);

// Write a function to write kth row of a 2d tensor into a file 
void write_row_to_file(const torch::Tensor& tensor, int row, const std::string& filename) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        file << "[ ";
        for (int col = 0; col < tensor.size(1); ++col) {
            file << tensor[row][col].item<int>() << ",";
        }
        file << "]";

        file << "\n";
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

// Write a function to write kth col of a 2d tensor into a file 
void write_col_to_file(const torch::Tensor& tensor, int col, const std::string& filename) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        file << "[ ";
        for (int row = 0; row < tensor.size(0); ++row) {
            file << tensor[row][col].item<int>() << ", ";
        }
        file << "]";
        file << "\n";
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

void naive_gemm(const uint8_t* A, 
                const uint8_t* B, 
                float* C, 
                int M, int K, int N);

// Ensure 16-byte alignment for NEON
torch::Tensor make_aligned(const torch::Tensor& t) {
    void* ptr = t.data_ptr<uint8_t>();
    if (reinterpret_cast<uintptr_t>(ptr) % 16 == 0) {
        std::cout << "Tensor is already 16-byte aligned!" << std::endl;
        return t;
    } else {
        std::cout << "Tensor is not 16-byte aligned! Creating aligned copy." << std::endl;

        // Allocate a new tensor with same shape and dtype, explicitly requesting contiguous layout
        auto aligned = torch::empty_like(t, torch::MemoryFormat::Contiguous);
        aligned.copy_(t);

        // Sanity check
        assert(reinterpret_cast<uintptr_t>(aligned.data_ptr<uint8_t>()) % 16 == 0 && "Tensor is not 16-byte aligned!");
        return aligned;
    }
}


void test_gemm_shape_and_values(int M, int K, int N, std::string mode = "normal") {
    torch::manual_seed(42);

    // set a random seed so it's always the same matrix 

    // Generate random -1/1 matrices

    torch::Tensor A, B;
    if (mode == "normal") {
        A = (torch::randint(0, 2, {M, K}, torch::kFloat32) * 2 - 1).to(torch::kFloat32);
        B = (torch::randint(0, 2, {K, N}, torch::kFloat32) * 2 - 1).to(torch::kFloat32);
    } else if (mode == "mostly_negative_ones") {
        // Define probabilities for 0 and 2
        auto probabilities_A = torch::tensor({0.8, 0.2}, torch::kFloat32);
        auto indices_A = torch::multinomial(probabilities_A, M * K, true).reshape({M, K});
        A = (indices_A * 2 - 1).to(torch::kFloat32);
        auto probabilities_B = torch::tensor({0.75, 0.25}, torch::kFloat32);
        auto indices_B = torch::multinomial(probabilities_B, K * N, true).reshape({K, N});
        B = (indices_B * 2 - 1).to(torch::kFloat32);
    } else if (mode == "almost_negative_ones") {
       
        // Define probabilities for 0 and 2
        auto probabilities_A = torch::tensor({0.99, 0.01}, torch::kFloat32);
        auto indices_A = torch::multinomial(probabilities_A, M * K, true).reshape({M, K});
        A = (indices_A * 2 - 1).to(torch::kFloat32);
        auto probabilities_B = torch::tensor({0.98, 0.02}, torch::kFloat32);
        auto indices_B = torch::multinomial(probabilities_B, K * N, true).reshape({K, N});
        B = (indices_B * 2 - 1).to(torch::kFloat32);
    } else if (mode == "mostly_ones") {
        // Define probabilities for 0 and 2
        auto probabilities_A = torch::tensor({0.21, 0.79}, torch::kFloat32);
        auto indices_A = torch::multinomial(probabilities_A, M * K, true).reshape({M, K});
        A = (indices_A * 2 - 1).to(torch::kFloat32);

        auto probabilities_B = torch::tensor({0.23, 0.77}, torch::kFloat32);
        auto indices_B = torch::multinomial(probabilities_B, K * N, true).reshape({K, N});
        B = (indices_B * 2 - 1).to(torch::kFloat32);
    } else if (mode == "almost_ones") {
       
        // Define probabilities for 0 and 2
        auto probabilities_A = torch::tensor({0.0001, 0.9999}, torch::kFloat32);
        auto indices_A = torch::multinomial(probabilities_A, M * K, true).reshape({M, K});
        A = (indices_A * 2 - 1).to(torch::kFloat32);

        auto probabilities_B = torch::tensor({0.03, 0.97}, torch::kFloat32);
        auto indices_B = torch::multinomial(probabilities_B, K * N, true).reshape({K, N});
        B = (indices_B * 2 - 1).to(torch::kFloat32);
    } else if (mode == "all_ones"){
        A = torch::ones({M, K}, torch::kFloat32);
        B = torch::ones({K, N}, torch::kFloat32);
    } else if (mode == "all_negative_ones"){
        A = -1 * torch::ones({M, K}, torch::kFloat32);
        B = -1 * torch::ones({K, N}, torch::kFloat32);
    } else {
        A = (torch::randint(0, 2, {M, K}, torch::kFloat32) * 2 - 1).to(torch::kFloat32);
        B = (torch::randint(0, 2, {K, N}, torch::kFloat32) * 2 - 1).to(torch::kFloat32);
    }
    // std::cout << A << std::endl;
    // std::cout << B << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    

    auto A_bin_int8 = A.to(torch::kInt8).contiguous();
    auto B_bin_int8 = B.to(torch::kInt8).contiguous();
    auto B_trans = B_bin_int8.transpose(0, 1).contiguous();
    auto A_packed = pack_binary_matrix(A_bin_int8);
    auto B_packed = pack_binary_matrix(B_trans);

    // A_packed = make_aligned(A_packed);
    // B_packed = make_aligned(B_packed);

    auto C_neon = torch::zeros({M, N}, torch::kInt32);

    neon_binary_gemm_improved(
        A_packed.data_ptr<uint8_t>(),
        B_packed.data_ptr<uint8_t>(),
        C_neon.data_ptr<int32_t>(),
        M, K, N
    );


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_neon = end - start;


    start = std::chrono::high_resolution_clock::now();
    torch::Tensor C_ref = torch::matmul(A, B);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_torch = end - start;

    // Print the first row of A and first col of B

    // Print the value, row, and col of the first difference between C_ref and C_neon
    // for (int i = 0; i < C_ref.size(0); ++i) {
    //     for (int j = 0; j < C_ref.size(1); ++j) {
    //         if (C_ref[i][j].item<int>() != C_neon[i][j].item<int>()) {
    //             std::cout << "Difference at row " << i << ", col " << j
    //                     << ": C_ref=" << C_ref[i][j].item<int>()
    //                     << ", C_neon=" << C_neon[i][j].item<int>() << std::endl;
    //         }
    //     }
    // }
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Neon Binary Gemm Elapsed Time: " << elapsed_neon.count() << " ms\n";
    std::cout << "Torch Gemm Elapsed Time: " << elapsed_torch.count() << " ms\n";

    // Check shape
    assert(C_neon.sizes() == C_ref.sizes());

    // Check values (allowing for int/float comparison)
    auto C_neon_float = C_neon.to(torch::kFloat32);
    auto C_ref_float = C_ref.to(torch::kFloat32);
    assert(torch::allclose(C_neon_float, C_ref_float, 1e-8, 1e-8));

    std::cout << "🟢 Passed GEMM test: " << M << "x" << K << " * " << K << "x" << N << " | Mode: " << mode << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}

/**
 * Test cases
 * Normal test cases
 * 4x4 * 4x2 
 * 2x2 * 2x4
 * 3x9 x 9x2
 * 3x10 x 10x2
 * 3x11 x 11x4
 * Edge cases 
 * 1x1 x 1x1
 * 1x1 x 1x1000
 * 1000x1  x 1x1000
 * 1x1000 x 1000x1
 * Common matrix sizes 
 * 4096x4096 x 4096x4096 
 * 1024x1024 x 1024x1024 
 * 1024x1024 x 1024x1 
 * 1024x768 x 768x1 
 * 768x768 x 768x1 
 * 512x512 x 512x1 
 * Stress tests
 * 10000 x 10000
 * 20000 x 20000
 * 50000 x 50000
 * 
 * Values test cases
 * All -1s
 * All 1s
 * Mostly -1s
 * Mostly 1s
 */
int main() {
    // Testing small shape
    test_gemm_shape_and_values(1, 1, 1);
    test_gemm_shape_and_values(1, 1, 2);
    test_gemm_shape_and_values(2, 1, 1);
    test_gemm_shape_and_values(1, 2, 1);
    test_gemm_shape_and_values(2, 3, 5);
    test_gemm_shape_and_values(2, 2, 4);
    test_gemm_shape_and_values(2, 11, 4);
    test_gemm_shape_and_values(2, 4, 11);
    test_gemm_shape_and_values(2, 5, 7);
    test_gemm_shape_and_values(4, 7, 4);
    test_gemm_shape_and_values(3, 9, 2);
    test_gemm_shape_and_values(3, 10, 3);
    test_gemm_shape_and_values(3, 10, 2);
    test_gemm_shape_and_values(3, 11, 4);
    test_gemm_shape_and_values(4, 7, 2);
    test_gemm_shape_and_values(4, 7, 11);
    test_gemm_shape_and_values(11, 5, 8);
    test_gemm_shape_and_values(8, 8, 8);
    test_gemm_shape_and_values(0, 0, 0);
    test_gemm_shape_and_values(8, 8, 0);
    test_gemm_shape_and_values(0, 8, 8);
    test_gemm_shape_and_values(8, 0, 0);
    test_gemm_shape_and_values(0, 8, 0);
    test_gemm_shape_and_values(0, 0, 8);
    

    // Testing shapes > 128 
    test_gemm_shape_and_values(121, 121, 121);
    test_gemm_shape_and_values(122, 122, 122);
    test_gemm_shape_and_values(123, 123, 123);
    test_gemm_shape_and_values(124, 124, 124);
    test_gemm_shape_and_values(127, 127, 127);
    test_gemm_shape_and_values(128, 128, 1);
    test_gemm_shape_and_values(128, 128, 128);
    test_gemm_shape_and_values(144, 144, 144);
    test_gemm_shape_and_values(248, 248, 248);
    test_gemm_shape_and_values(249, 249, 249);
    test_gemm_shape_and_values(254, 254, 254);
    test_gemm_shape_and_values(256, 256, 256);
    test_gemm_shape_and_values(376, 376, 376);
    test_gemm_shape_and_values(377, 377, 377);
    test_gemm_shape_and_values(129, 129, 1);

    test_gemm_shape_and_values(156, 156, 156);
    test_gemm_shape_and_values(1, 256, 256);
    test_gemm_shape_and_values(256, 128, 512);
    test_gemm_shape_and_values(128, 512, 0);
    test_gemm_shape_and_values(128, 512, 1);
    test_gemm_shape_and_values(256, 256, 1);
    test_gemm_shape_and_values(372, 372, 372);
    test_gemm_shape_and_values(384, 384, 384);
    test_gemm_shape_and_values(512, 512, 512);
    test_gemm_shape_and_values(641, 641, 641);
    test_gemm_shape_and_values(656, 656, 656);
    test_gemm_shape_and_values(672, 672, 672);
    test_gemm_shape_and_values(688, 688, 688);
    test_gemm_shape_and_values(704, 704, 704);
    test_gemm_shape_and_values(722, 722, 722);
    test_gemm_shape_and_values(740, 740, 740);
    test_gemm_shape_and_values(512, 512, 1);
    test_gemm_shape_and_values(642, 642, 642);
    test_gemm_shape_and_values(768, 768, 1);
    test_gemm_shape_and_values(768, 1, 768);
    test_gemm_shape_and_values(896, 896, 896);
    test_gemm_shape_and_values(1, 768, 768);
    test_gemm_shape_and_values(1, 1, 768);
    test_gemm_shape_and_values(768, 1, 1);
    test_gemm_shape_and_values(1, 768, 1);

    test_gemm_shape_and_values(768, 768, 768);
    test_gemm_shape_and_values(512, 1024, 1);
    test_gemm_shape_and_values(768, 1024, 1);
    test_gemm_shape_and_values(512, 768, 1024);
    test_gemm_shape_and_values(512, 768, 1);
    test_gemm_shape_and_values(0, 512, 0);
    test_gemm_shape_and_values(768, 512, 1);
    test_gemm_shape_and_values(768, 768, 1);
    test_gemm_shape_and_values(900, 900, 900);
    test_gemm_shape_and_values(900, 0, 900);
    test_gemm_shape_and_values(900, 333, 222);
    test_gemm_shape_and_values(1024, 1024, 1024);
    test_gemm_shape_and_values(1024, 1024, 1);
    test_gemm_shape_and_values(1024, 768, 1);
    test_gemm_shape_and_values(124, 712, 381);
    test_gemm_shape_and_values(2048, 2048, 2048);
    test_gemm_shape_and_values(4096, 4096, 4096);

    test_gemm_shape_and_values(1024, 32, 1024, "all_ones");
    test_gemm_shape_and_values(1024, 567, 1024, "all_negative_ones");
    test_gemm_shape_and_values(1024, 1024, 1024, "almost_negative_ones");
    test_gemm_shape_and_values(1024, 1024, 1024, "almost_ones");
    test_gemm_shape_and_values(1024, 1024, 1024, "mostly_negative_ones");
    test_gemm_shape_and_values(1024, 7182, 1024, "mostly_ones");

    test_gemm_shape_and_values(1024, 768, 1024, "all_ones");
    test_gemm_shape_and_values(828, 1024, 1024, "all_negative_ones");
    test_gemm_shape_and_values(1024, 22, 768, "almost_negative_ones");
    test_gemm_shape_and_values(1024, 1024, 1024, "almost_ones");
    test_gemm_shape_and_values(768, 1024, 1024, "mostly_negative_ones");
    test_gemm_shape_and_values(1024, 1024, 1024, "mostly_ones");

    test_gemm_shape_and_values(1024, 1024, 1024, "all_ones");
    test_gemm_shape_and_values(1024, 1024, 391, "all_negative_ones");
    test_gemm_shape_and_values(1024, 768, 1024, "almost_negative_ones");
    test_gemm_shape_and_values(2283, 1024, 1024, "almost_ones");
    test_gemm_shape_and_values(512, 128, 1, "mostly_negative_ones");
    test_gemm_shape_and_values(768, 1024, 1024, "mostly_ones");

    test_gemm_shape_and_values(1024, 1024, 1024, "all_ones");
    test_gemm_shape_and_values(1, 1024, 768, "all_negative_ones");
    test_gemm_shape_and_values(1024, 1024, 1024, "almost_negative_ones");
    test_gemm_shape_and_values(1024, 719, 1024, "almost_ones");
    test_gemm_shape_and_values(111, 1024, 123, "mostly_negative_ones");
    test_gemm_shape_and_values(1024, 1024, 1, "mostly_ones");
    
    // TODO: test if it's faster than pytorch
    // TODO: figure out why small shapes are slower 
    // TODO: finish other test cases

    return 0;
}