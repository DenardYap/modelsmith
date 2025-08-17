#include <torch/extension.h>
#include <vector>

torch::Tensor pack_binary_matrix_SIMD128(torch::Tensor mat);
void neon_binary_gemm(const uint8_t* A_packed,
                      const uint8_t* B_bin,
                      int32_t* C,
                      int M, int K, int N);

void neon_binary_gemm_tiled_with_params(const uint8_t* A_packed,
                                       const uint8_t* B_bin,
                                       int32_t* C,
                                       int M, int K, int N,
                                       int tile_m, int tile_n, int tile_k);


torch::Tensor binary_matmul(torch::Tensor A_bin,
                                    torch::Tensor B_bin) {

    // Step 1 : Check the input tensors, any type is accepted as of now (float, double, int, etc.)
    TORCH_CHECK(A_bin.size(1) == B_bin.size(0), "Expected second dimension of A_bin to match first dimension of B_bin");   
    int K = A_bin.size(1);
    auto end = std::chrono::high_resolution_clock::now();

    // Making sure only -1 and 1 are present in the matrices A_bin and B_bin
    auto A_bin_int8 = A_bin.to(torch::kInt8).contiguous();
    auto B_bin_int8 = B_bin.to(torch::kInt8).contiguous();
    
    int M = A_bin.size(0);
    int N = B_bin.size(1);  

    // Step 2: Transpose B_matrix because in neon_binary_gemm we need B to be in column major order
    torch::Tensor B_trans = B_bin_int8.transpose(0, 1).contiguous();

    // Step 3 : Pack the binary matrices
    torch::Tensor A_packed = pack_binary_matrix_SIMD128(A_bin_int8);
    torch::Tensor B_packed = pack_binary_matrix_SIMD128(B_trans);

    // TODO: make sure the type is lower or something
    auto output =  torch::zeros({M, N}, torch::kInt32);

    // Step 4 : Call the neon_binary_gemm function
    neon_binary_gemm(A_packed.data_ptr<uint8_t>(),
                     B_packed.data_ptr<uint8_t>(),
                     output.data_ptr<int32_t>(),
                     M, K, N);


    return output;
}

torch::Tensor binary_linear_matmul(torch::Tensor A_bin,
                                    torch::Tensor B_bin) {

    // Step 1 : Check the input tensors, any type is accepted as of now (float, double, int, etc.)
    TORCH_CHECK(A_bin.size(1) == B_bin.size(0), "Expected second dimension of A_bin to match first dimension of B_bin");   
    int K = A_bin.size(1);
    auto end = std::chrono::high_resolution_clock::now();

    // Making sure only -1 and 1 are present in the matrices A_bin and B_bin
    auto A_bin_int8 = A_bin.to(torch::kInt8).contiguous();
    auto B_bin_int8 = B_bin.to(torch::kInt8).contiguous();
    
    int M = A_bin.size(0);
    int N = B_bin.size(1);  

    // Step 2: Transpose B_matrix because in neon_binary_gemm we need B to be in column major order
    torch::Tensor B_trans = B_bin.transpose(0, 1).contiguous();

    // Step 3 : Pack the binary matrices
    torch::Tensor A_packed = pack_binary_matrix_SIMD128(A_bin);
    torch::Tensor B_packed = pack_binary_matrix_SIMD128(B_trans);

    // TODO: make sure the type is lower or something
    auto output =  torch::zeros({M, N}, torch::kInt32);

    // Step 4 : Call the neon_binary_gemm function
    neon_binary_gemm(A_packed.data_ptr<uint8_t>(),
                     B_packed.data_ptr<uint8_t>(),
                     output.data_ptr<int32_t>(),
                     M, K, N);


    return output;
}


torch::Tensor binary_matmul_tiled_with_params(torch::Tensor A_bin,
                                             torch::Tensor B_bin) {

    // Step 1 : Check the input tensors
    TORCH_CHECK(A_bin.size(1) == B_bin.size(0), "Expected second dimension of A_bin to match first dimension of B_bin");   
    int K = A_bin.size(1);

    // Making sure only -1 and 1 are present in the matrices A_bin and B_bin
    auto A_bin_int8 = A_bin.to(torch::kInt8).contiguous();
    auto B_bin_int8 = B_bin.to(torch::kInt8).contiguous();
    
    int M = A_bin.size(0);
    int N = B_bin.size(1);  

    // Step 2: Transpose B_matrix because in neon_binary_gemm we need B to be in column major order
    torch::Tensor B_trans = B_bin_int8.transpose(0, 1).contiguous();

    // Step 3 : Pack the binary matrices
    torch::Tensor A_packed = pack_binary_matrix_SIMD128(A_bin_int8);
    torch::Tensor B_packed = pack_binary_matrix_SIMD128(B_trans);

    // TODO: make sure the type is lower or something
    auto output =  torch::zeros({M, N}, torch::kInt32);
    
    // Step 4 : Call the neon_binary_gemm_tiled_with_params function
    neon_binary_gemm_tiled_with_params(A_packed.data_ptr<uint8_t>(),
                                     B_packed.data_ptr<uint8_t>(),
                                     output.data_ptr<int32_t>(),
                                     M, K, N,
                                     16, 128, 128);

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("binary_matmul", &binary_matmul, "Binary Matrix Multiplication");
    m.def("binary_matmul_tiled", &binary_matmul_tiled_with_params, "Tiled Binary Matrix Multiplication");
}

