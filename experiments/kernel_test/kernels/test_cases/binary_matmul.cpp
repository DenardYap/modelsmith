#include <torch/extension.h>
#include <vector>
#define ENABLE_INPUT_CHECKS 1

torch::Tensor pack_binary_matrix(torch::Tensor mat);
torch::Tensor pack_binary_matrix_SIMD64(torch::Tensor mat);
torch::Tensor pack_binary_matrix_SIMD128(torch::Tensor mat);
torch::Tensor pack_binary_matrix_SIMD128_float_input(torch::Tensor mat);
void neon_binary_gemm(const uint8_t* A_packed,
                      const uint8_t* B_bin,
                      int32_t* C,
                      int M, int K, int N);
// void neon_binary_gemm2(const uint8_t* A_packed,
//                       const uint8_t* B_bin,
//                       int32_t* C,
//                       int M, int K, int N);
void neon_binary_gemm_improved(const uint8_t* A_packed,
                      const uint8_t* B_bin,
                      int32_t* C,
                      int M, int K, int N);
void neon_binary_gemm_tiled(const uint8_t* A_packed,
                      const uint8_t* B_bin,
                      int32_t* C,
                      int M, int K, int N);
void neon_binary_gemm_tiled_unrolled(const uint8_t* A_packed,
                      const uint8_t* B_bin,
                      int32_t* C,
                      int M, int K, int N);
void neon_binary_gemm_tiled_old(const uint8_t* A_packed,
                      const uint8_t* B_bin,
                      int32_t* C,
                      int M, int K, int N);
void neon_binary_gemm_tiled_with_params(const uint8_t* A_packed,
                                       const uint8_t* B_packed,
                                       int32_t* C,
                                       int M, int K, int N,
                                       int tile_m, int tile_n, int tile_k);
                  

                      
// TODO: maybe use aligned_alloc for maximal performance 
// Wrapper exposed to Python
torch::Tensor binary_matmul(torch::Tensor A_bin,
                                    torch::Tensor B_bin) {

    std::cout << "===== binary_matmul ======" << std::endl;
    auto total_start = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    // Step 1 : Check the input tensors, any type is accepted as of now (float, double, int, etc.)
    TORCH_CHECK(A_bin.size(1) == B_bin.size(0), "Expected second dimension of A_bin to match first dimension of B_bin");   
    int K = A_bin.size(1);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> check_time1 = end - start;

    start = std::chrono::high_resolution_clock::now();
    // Making sure only -1 and 1 are present in the matrices A_bin and B_bin
    auto A_bin_int8 = A_bin.to(torch::kInt8);
    auto B_bin_int8 = B_bin.to(torch::kInt8);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> check_time2 = end - start;
 
        // Torch check the max and min val of A_bin_int8 make sure they are -1 and 1
    
    // #if ENABLE_INPUT_CHECKS
    //     start = std::chrono::high_resolution_clock::now();
    //     for (int i = 0; i < A_bin_int8.numel(); ++i) {
    //         int8_t val = A_bin_int8.data_ptr<int8_t>()[i];
    //         TORCH_CHECK(val == -1 || val == 1, "A_bin contains values other than -1 or 1");
    //     }
    //     end = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double, std::milli> check_time3 = end - start;
    //     start = std::chrono::high_resolution_clock::now();
    //     for (int i = 0; i < B_bin_int8.numel(); ++i) {
    //         int8_t val = B_bin_int8.data_ptr<int8_t>()[i];
    //         TORCH_CHECK(val == -1 || val == 1, "B_bin contains values other than -1 or 1");
    //     }
    //     end = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double, std::milli> check_time4 = end - start;
    // #endif 

    int M = A_bin.size(0);
    int N = B_bin.size(1);  

    // Step 2: Transpose B_matrix because in neon_binary_gemm we need B to be in column major order

    // Step 3 : Pack the binary matrices

    start = std::chrono::high_resolution_clock::now();
    torch::Tensor B_trans = B_bin_int8.transpose(0, 1);
    torch::Tensor A_packed = pack_binary_matrix(A_bin_int8);
    torch::Tensor B_packed = pack_binary_matrix(B_trans);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> pack_time = end - start;

    // TODO: make sure the type is lower or something
    auto output =  torch::zeros({M, N}, torch::kInt32);


    start = std::chrono::high_resolution_clock::now();
    // std::cout << "Neon res:" << std::endl;
    // Step 4 : Call the neon_binary_gemm function
    neon_binary_gemm(A_packed.data_ptr<uint8_t>(),
                     B_packed.data_ptr<uint8_t>(),
                     output.data_ptr<int32_t>(),
                     M, K, N);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gemm_time = end - start;
    std::cout << "Gemm Elapsed Time: " << gemm_time.count() << " ms\n";

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_time = total_end - total_start;
    // std::cout << "Total Elapsed Time: " << total_time.count() << " ms\n";

    double check_time1_fraction = (check_time1.count() / total_time.count()) * 100.0;
    double check_time2_fraction = (check_time2.count() / total_time.count()) * 100.0;
    double pack_fraction = (pack_time.count() / total_time.count()) * 100.0;
    double gemm_fraction = (gemm_time.count() / total_time.count()) * 100.0;

    std::cout << "check_time1_fraction: " << check_time1_fraction << "%\n";
    std::cout << "check_time2_fraction: " << check_time2_fraction << "%\n";
    std::cout << "bit packing: " << pack_fraction << "%\n";
    std::cout << "neon_binary_gemm: " << gemm_fraction << "%\n";
    std::cout << "===== binary_matmul ======" << std::endl;
    return output;
}


// torch::Tensor binary_matmul(torch::Tensor A_packed,
//                                     torch::Tensor B_packed) {


//     // The same thign as binary_matmul but here A_packed and B_packed is expected to already been casted + bitpacked
//     TORCH_CHECK(A_packed.size(1) == B_packed.size(0), "Expected second dimension of A_packed to match first dimension of B_packed");   

//     int M = A_packed.size(0);
//     int N = B_packed.size(1);  
//     auto output =  torch::zeros({M, N}, torch::kInt32);


//     neon_binary_gemm(A_packed.data_ptr<uint8_t>(),
//                      B_packed.data_ptr<uint8_t>(),
//                      output.data_ptr<int32_t>(),
//                      M, K, N);

//     return output;
// }

torch::Tensor binary_matmul64(torch::Tensor A_bin,
                                    torch::Tensor B_bin) {

    auto total_start = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    // Step 1 : Check the input tensors, any type is accepted as of now (float, double, int, etc.)
    TORCH_CHECK(A_bin.size(1) == B_bin.size(0), "Expected second dimension of A_bin to match first dimension of B_bin");   
    int K = A_bin.size(1);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> check_time1 = end - start;

    start = std::chrono::high_resolution_clock::now();
    // Making sure only -1 and 1 are present in the matrices A_bin and B_bin
    auto A_bin_int8 = A_bin.to(torch::kInt8).contiguous();
    auto B_bin_int8 = B_bin.to(torch::kInt8).contiguous();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> check_time2 = end - start;
 
        // Torch check the max and min val of A_bin_int8 make sure they are -1 and 1
    
    #if ENABLE_INPUT_CHECKS
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < A_bin_int8.numel(); ++i) {
            int8_t val = A_bin_int8.data_ptr<int8_t>()[i];
            TORCH_CHECK(val == -1 || val == 1, "A_bin contains values other than -1 or 1");
        }
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> check_time3 = end - start;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < B_bin_int8.numel(); ++i) {
            int8_t val = B_bin_int8.data_ptr<int8_t>()[i];
            TORCH_CHECK(val == -1 || val == 1, "B_bin contains values other than -1 or 1");
        }
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> check_time4 = end - start;
    #endif 

    int M = A_bin.size(0);
    int N = B_bin.size(1);  

    // Step 2: Transpose B_matrix because in neon_binary_gemm we need B to be in column major order
    torch::Tensor B_trans = B_bin_int8.transpose(0, 1).contiguous();

    // Step 3 : Pack the binary matrices

    start = std::chrono::high_resolution_clock::now();
    torch::Tensor A_packed = pack_binary_matrix_SIMD64(A_bin_int8);
    torch::Tensor B_packed = pack_binary_matrix_SIMD64(B_trans);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> pack_time = end - start;

    // TODO: make sure the type is lower or something
    auto output =  torch::zeros({M, N}, torch::kInt32);


    start = std::chrono::high_resolution_clock::now();
    // std::cout << "Neon res:" << std::endl;
    // Step 4 : Call the neon_binary_gemm function
    neon_binary_gemm(A_packed.data_ptr<uint8_t>(),
                     B_packed.data_ptr<uint8_t>(),
                     output.data_ptr<int32_t>(),
                     M, K, N);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gemm_time = end - start;
    // std::cout << "Gemm Elapsed Time: " << gemm_time.count() << " ms\n";

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_time = total_end - total_start;
    std::cout << "Total Elapsed Time: " << total_time.count() << " ms\n";

    double check_time1_fraction = (check_time1.count() / total_time.count()) * 100.0;
    double check_time2_fraction = (check_time2.count() / total_time.count()) * 100.0;
    double pack_fraction = (pack_time.count() / total_time.count()) * 100.0;
    double gemm_fraction = (gemm_time.count() / total_time.count()) * 100.0;

    // std::cout << "check_time1_fraction: " << check_time1_fraction << "%\n";
    // std::cout << "check_time2_fraction: " << check_time2_fraction << "%\n";
    // std::cout << "bit packing: " << pack_fraction << "%\n";
    // std::cout << "neon_binary_gemm: " << gemm_fraction << "%\n";


    return output;
}

// torch::Tensor binary_matmul128_2(torch::Tensor A_bin,
//                                     torch::Tensor B_bin) {

//     auto total_start = std::chrono::high_resolution_clock::now();
//     auto start = std::chrono::high_resolution_clock::now();
//     // Step 1 : Check the input tensors, any type is accepted as of now (float, double, int, etc.)
//     TORCH_CHECK(A_bin.size(1) == B_bin.size(0), "Expected second dimension of A_bin to match first dimension of B_bin");   
//     int K = A_bin.size(1);
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> check_time1 = end - start;

//     start = std::chrono::high_resolution_clock::now();
//     // Making sure only -1 and 1 are present in the matrices A_bin and B_bin
//     auto A_bin_int8 = A_bin.to(torch::kInt8);
//     auto B_bin_int8 = B_bin.to(torch::kInt8);
//     end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> check_time2 = end - start;
 
//     // Torch check the max and min val of A_bin_int8 make sure they are -1 and 1

//     int M = A_bin.size(0);
//     int N = B_bin.size(1);  

//     // Step 2: Transpose B_matrix because in neon_binary_gemm we need B to be in column major order
//     torch::Tensor B_trans = B_bin_int8.transpose(0, 1);

//     // Step 3 : Pack the binary matrices
//     start = std::chrono::high_resolution_clock::now();
//     torch::Tensor A_packed = pack_binary_matrix_SIMD128(A_bin_int8);
//     torch::Tensor B_packed = pack_binary_matrix_SIMD128(B_trans);
//     end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> pack_time = end - start;

//     // TODO: make sure the type is lower or something
//     auto output =  torch::zeros({M, N}, torch::kInt32);


//     start = std::chrono::high_resolution_clock::now();
//     // std::cout << "Neon res:" << std::endl;
//     // Step 4 : Call the neon_binary_gemm function
//     neon_binary_gemm2(A_packed.data_ptr<uint8_t>(),
//                      B_packed.data_ptr<uint8_t>(),
//                      output.data_ptr<int32_t>(),
//                      M, K, N);
//     end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> gemm_time = end - start;
//     std::cout << "Gemm Elapsed Time: " << gemm_time.count() << " ms\n";

//     auto total_end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> total_time = total_end - total_start;
//     std::cout << "Total Elapsed Time: " << total_time.count() << " ms\n";

//     double check_time1_fraction = (check_time1.count() / total_time.count()) * 100.0;
//     double check_time2_fraction = (check_time2.count() / total_time.count()) * 100.0;
//     double pack_fraction = (pack_time.count() / total_time.count()) * 100.0;
//     double gemm_fraction = (gemm_time.count() / total_time.count()) * 100.0;

//     std::cout << "check_time1_fraction: " << check_time1_fraction << "%\n";
//     std::cout << "check_time2_fraction: " << check_time2_fraction << "%\n";
//     std::cout << "bit packing: " << pack_fraction << "%\n";
//     std::cout << "neon_binary_gemm: " << gemm_fraction << "%\n";


//     return output;
// }


torch::Tensor binary_matmul128(torch::Tensor A_bin,
                                    torch::Tensor B_bin) {

    std::cout << "===== binary_matmul128 ======" << std::endl;
    auto total_start = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    // Step 1 : Check the input tensors, any type is accepted as of now (float, double, int, etc.)
    TORCH_CHECK(A_bin.size(1) == B_bin.size(0), "Expected second dimension of A_bin to match first dimension of B_bin");   
    int K = A_bin.size(1);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> check_time1 = end - start;

    start = std::chrono::high_resolution_clock::now();
    // Making sure only -1 and 1 are present in the matrices A_bin and B_bin
    auto A_bin_int8 = A_bin.to(torch::kInt8);
    auto B_bin_int8 = B_bin.to(torch::kInt8);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> check_time2 = end - start;
 
    // Torch check the max and min val of A_bin_int8 make sure they are -1 and 1

    int M = A_bin.size(0);
    int N = B_bin.size(1);  

    // Step 2: Transpose B_matrix because in neon_binary_gemm we need B to be in column major order

    // Step 3 : Pack the binary matrices
    start = std::chrono::high_resolution_clock::now();
    torch::Tensor B_trans = B_bin_int8.transpose(0, 1);
    torch::Tensor A_packed = pack_binary_matrix_SIMD128(A_bin_int8);
    torch::Tensor B_packed = pack_binary_matrix_SIMD128(B_trans);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> pack_time = end - start;

    // TODO: make sure the type is lower or something
    auto output =  torch::zeros({M, N}, torch::kInt32);


    start = std::chrono::high_resolution_clock::now();
    // std::cout << "Neon res:" << std::endl;
    // Step 4 : Call the neon_binary_gemm function
    neon_binary_gemm(A_packed.data_ptr<uint8_t>(),
                     B_packed.data_ptr<uint8_t>(),
                     output.data_ptr<int32_t>(),
                     M, K, N);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gemm_time = end - start;
    std::cout << "Gemm Elapsed Time: " << gemm_time.count() << " ms\n";

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_time = total_end - total_start;
    std::cout << "Total Elapsed Time: " << total_time.count() << " ms\n";

    double check_time1_fraction = (check_time1.count() / total_time.count()) * 100.0;
    double check_time2_fraction = (check_time2.count() / total_time.count()) * 100.0;
    double pack_fraction = (pack_time.count() / total_time.count()) * 100.0;
    double gemm_fraction = (gemm_time.count() / total_time.count()) * 100.0;

    std::cout << "check_time1_fraction: " << check_time1_fraction << "%\n";
    std::cout << "check_time2_fraction: " << check_time2_fraction << "%\n";
    std::cout << "bit packing: " << pack_fraction << "%\n";
    std::cout << "neon_binary_gemm: " << gemm_fraction << "%\n";

    std::cout << "===== binary_matmul128 ======" << std::endl;

    return output;
}


torch::Tensor binary_matmul128_int(torch::Tensor A_bin,
                                    torch::Tensor B_bin) {

    auto total_start = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();

    // Assert A_bin and B_bin are kInt8
    TORCH_CHECK(A_bin.dtype() == torch::kInt8, "A_bin must be of type int8");
    TORCH_CHECK(B_bin.dtype() == torch::kInt8, "B_bin must be of type int8");
    // Step 1 : Check the input tensors, any type is accepted as of now (float, double, int, etc.)
    TORCH_CHECK(A_bin.size(1) == B_bin.size(0), "Expected second dimension of A_bin to match first dimension of B_bin");   
    int K = A_bin.size(1);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> check_time1 = end - start;

    start = std::chrono::high_resolution_clock::now();
    // Making sure only -1 and 1 are present in the matrices A_bin and B_bin
    auto A_bin_int8 = A_bin.to(torch::kInt8);
    auto B_bin_int8 = B_bin.to(torch::kInt8);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> check_time2 = end - start;
 
        // Torch check the max and min val of A_bin_int8 make sure they are -1 and 1
    

    int M = A_bin.size(0);
    int N = B_bin.size(1);  

    // Step 2: Transpose B_matrix because in neon_binary_gemm we need B to be in column major order
    torch::Tensor B_trans = B_bin_int8.transpose(0, 1);

    // Step 3 : Pack the binary matrices
    start = std::chrono::high_resolution_clock::now();
    torch::Tensor A_packed = pack_binary_matrix_SIMD128(A_bin_int8);
    torch::Tensor B_packed = pack_binary_matrix_SIMD128(B_trans);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> pack_time = end - start;

    // TODO: make sure the type is lower or something
    auto output =  torch::zeros({M, N}, torch::kInt32);


    start = std::chrono::high_resolution_clock::now();
    // std::cout << "Neon res:" << std::endl;
    // Step 4 : Call the neon_binary_gemm function
    neon_binary_gemm(A_packed.data_ptr<uint8_t>(),
                     B_packed.data_ptr<uint8_t>(),
                     output.data_ptr<int32_t>(),
                     M, K, N);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gemm_time = end - start;
    std::cout << "Gemm Elapsed Time: " << gemm_time.count() << " ms\n";

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_time = total_end - total_start;
    std::cout << "Total Elapsed Time: " << total_time.count() << " ms\n";

    double check_time1_fraction = (check_time1.count() / total_time.count()) * 100.0;
    double check_time2_fraction = (check_time2.count() / total_time.count()) * 100.0;
    double pack_fraction = (pack_time.count() / total_time.count()) * 100.0;
    double gemm_fraction = (gemm_time.count() / total_time.count()) * 100.0;

    std::cout << "check_time1_fraction: " << check_time1_fraction << "%\n";
    std::cout << "check_time2_fraction: " << check_time2_fraction << "%\n";
    std::cout << "bit packing: " << pack_fraction << "%\n";
    std::cout << "neon_binary_gemm: " << gemm_fraction << "%\n";


    return output;
}


torch::Tensor binary_matmul128_improved(torch::Tensor A_bin,
                                    torch::Tensor B_bin) {

    auto total_start = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    // Step 1 : Check the input tensors, any type is accepted as of now (float, double, int, etc.)
    TORCH_CHECK(A_bin.size(1) == B_bin.size(0), "Expected second dimension of A_bin to match first dimension of B_bin");   
    int K = A_bin.size(1);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> check_time1 = end - start;

    start = std::chrono::high_resolution_clock::now();
    // Making sure only -1 and 1 are present in the matrices A_bin and B_bin
    auto A_bin_int8 = A_bin.to(torch::kInt8);
    auto B_bin_int8 = B_bin.to(torch::kInt8);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> check_time2 = end - start;
 
        // Torch check the max and min val of A_bin_int8 make sure they are -1 and 1
    

    int M = A_bin.size(0);
    int N = B_bin.size(1);  

    // Step 2: Transpose B_matrix because in neon_binary_gemm we need B to be in column major order
    torch::Tensor B_trans = B_bin_int8.transpose(0, 1);

    // Step 3 : Pack the binary matrices
    start = std::chrono::high_resolution_clock::now();
    torch::Tensor A_packed = pack_binary_matrix_SIMD128(A_bin_int8);
    torch::Tensor B_packed = pack_binary_matrix_SIMD128(B_trans);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> pack_time = end - start;

    // TODO: make sure the type is lower or something
    auto output =  torch::zeros({M, N}, torch::kInt32);


    start = std::chrono::high_resolution_clock::now();
    // std::cout << "Neon res:" << std::endl;
    // Step 4 : Call the neon_binary_gemm function
    neon_binary_gemm_improved(A_packed.data_ptr<uint8_t>(),
                            B_packed.data_ptr<uint8_t>(),
                            output.data_ptr<int32_t>(),
                            M, K, N);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gemm_time = end - start;
    std::cout << "Gemm Elapsed Time: " << gemm_time.count() << " ms\n";

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_time = total_end - total_start;
    std::cout << "Total Elapsed Time: " << total_time.count() << " ms\n";

    double check_time1_fraction = (check_time1.count() / total_time.count()) * 100.0;
    double check_time2_fraction = (check_time2.count() / total_time.count()) * 100.0;
    double pack_fraction = (pack_time.count() / total_time.count()) * 100.0;
    double gemm_fraction = (gemm_time.count() / total_time.count()) * 100.0;

    std::cout << "check_time1_fraction: " << check_time1_fraction << "%\n";
    std::cout << "check_time2_fraction: " << check_time2_fraction << "%\n";
    std::cout << "bit packing: " << pack_fraction << "%\n";
    std::cout << "neon_binary_gemm: " << gemm_fraction << "%\n";


    return output;
}


torch::Tensor binary_matmul128_tiled(torch::Tensor A_bin,
                                    torch::Tensor B_bin) {

    std::cout << "===== binary_matmul128_tiled ======" << std::endl;
    auto total_start = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    // Step 1 : Check the input tensors, any type is accepted as of now (float, double, int, etc.)
    TORCH_CHECK(A_bin.size(1) == B_bin.size(0), "Expected second dimension of A_bin to match first dimension of B_bin");   
    int K = A_bin.size(1);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> check_time1 = end - start;

    start = std::chrono::high_resolution_clock::now();
    // Making sure only -1 and 1 are present in the matrices A_bin and B_bin
    auto A_bin_int8 = A_bin.to(torch::kInt8);
    auto B_bin_int8 = B_bin.to(torch::kInt8);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> check_time2 = end - start;
 
        // Torch check the max and min val of A_bin_int8 make sure they are -1 and 1
    

    int M = A_bin.size(0);
    int N = B_bin.size(1);  

    // Step 2: Transpose B_matrix because in neon_binary_gemm we need B to be in column major order
    torch::Tensor B_trans = B_bin_int8.transpose(0, 1);

    // Step 3 : Pack the binary matrices
    start = std::chrono::high_resolution_clock::now();
    torch::Tensor A_packed = pack_binary_matrix_SIMD128(A_bin_int8);
    torch::Tensor B_packed = pack_binary_matrix_SIMD128(B_trans);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> pack_time = end - start;

    // TODO: make sure the type is lower or something
    auto output =  torch::zeros({M, N}, torch::kInt32);


    start = std::chrono::high_resolution_clock::now();
    // std::cout << "Neon res:" << std::endl;
    // Step 4 : Call the neon_binary_gemm function
    neon_binary_gemm_tiled(A_packed.data_ptr<uint8_t>(),
                            B_packed.data_ptr<uint8_t>(),
                            output.data_ptr<int32_t>(),
                            M, K, N);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gemm_time = end - start;
    std::cout << "Gemm Elapsed Time: " << gemm_time.count() << " ms\n";

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_time = total_end - total_start;
    std::cout << "Total Elapsed Time: " << total_time.count() << " ms\n";

    double check_time1_fraction = (check_time1.count() / total_time.count()) * 100.0;
    double check_time2_fraction = (check_time2.count() / total_time.count()) * 100.0;
    double pack_fraction = (pack_time.count() / total_time.count()) * 100.0;
    double gemm_fraction = (gemm_time.count() / total_time.count()) * 100.0;

    std::cout << "check_time1_fraction: " << check_time1_fraction << "%\n";
    std::cout << "check_time2_fraction: " << check_time2_fraction << "%\n";
    std::cout << "bit packing: " << pack_fraction << "%\n";
    std::cout << "neon_binary_gemm: " << gemm_fraction << "%\n";

    std::cout << "===== binary_matmul128_tiled ======" << std::endl;

    return output;
}



torch::Tensor binary_matmul128_tiled_unrolled(torch::Tensor A_bin,
                                              torch::Tensor B_bin) {

    auto total_start = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    // Step 1 : Check the input tensors, any type is accepted as of now (float, double, int, etc.)
    TORCH_CHECK(A_bin.size(1) == B_bin.size(0), "Expected second dimension of A_bin to match first dimension of B_bin");   
    int K = A_bin.size(1);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> check_time1 = end - start;

    start = std::chrono::high_resolution_clock::now();
    // Making sure only -1 and 1 are present in the matrices A_bin and B_bin
    auto A_bin_int8 = A_bin.to(torch::kInt8);
    auto B_bin_int8 = B_bin.to(torch::kInt8);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> check_time2 = end - start;
 
        // Torch check the max and min val of A_bin_int8 make sure they are -1 and 1
    

    int M = A_bin.size(0);
    int N = B_bin.size(1);  

    // Step 2: Transpose B_matrix because in neon_binary_gemm we need B to be in column major order
    torch::Tensor B_trans = B_bin_int8.transpose(0, 1);

    // Step 3 : Pack the binary matrices
    start = std::chrono::high_resolution_clock::now();
    torch::Tensor A_packed = pack_binary_matrix_SIMD128(A_bin_int8);
    torch::Tensor B_packed = pack_binary_matrix_SIMD128(B_trans);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> pack_time = end - start;

    // TODO: make sure the type is lower or something
    auto output =  torch::zeros({M, N}, torch::kInt32);


    start = std::chrono::high_resolution_clock::now();
    // std::cout << "Neon res:" << std::endl;
    // Step 4 : Call the neon_binary_gemm function
    neon_binary_gemm_tiled_unrolled(A_packed.data_ptr<uint8_t>(),
                            B_packed.data_ptr<uint8_t>(),
                            output.data_ptr<int32_t>(),
                            M, K, N);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gemm_time = end - start;
    std::cout << "Gemm Elapsed Time: " << gemm_time.count() << " ms\n";

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_time = total_end - total_start;
    std::cout << "Total Elapsed Time: " << total_time.count() << " ms\n";

    double check_time1_fraction = (check_time1.count() / total_time.count()) * 100.0;
    double check_time2_fraction = (check_time2.count() / total_time.count()) * 100.0;
    double pack_fraction = (pack_time.count() / total_time.count()) * 100.0;
    double gemm_fraction = (gemm_time.count() / total_time.count()) * 100.0;

    std::cout << "check_time1_fraction: " << check_time1_fraction << "%\n";
    std::cout << "check_time2_fraction: " << check_time2_fraction << "%\n";
    std::cout << "bit packing: " << pack_fraction << "%\n";
    std::cout << "neon_binary_gemm: " << gemm_fraction << "%\n";


    return output;
}


torch::Tensor binary_matmul128_tiled_old(torch::Tensor A_bin,
                                        torch::Tensor B_bin) {

    std::cout << "===== binary_matmul128_tiled_old ======" << std::endl;
    auto total_start = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    // Step 1 : Check the input tensors, any type is accepted as of now (float, double, int, etc.)
    TORCH_CHECK(A_bin.size(1) == B_bin.size(0), "Expected second dimension of A_bin to match first dimension of B_bin");   
    int K = A_bin.size(1);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> check_time1 = end - start;

    start = std::chrono::high_resolution_clock::now();
    // Making sure only -1 and 1 are present in the matrices A_bin and B_bin
    auto A_bin_int8 = A_bin.to(torch::kInt8);
    auto B_bin_int8 = B_bin.to(torch::kInt8);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> check_time2 = end - start;
 
        // Torch check the max and min val of A_bin_int8 make sure they are -1 and 1
    

    int M = A_bin.size(0);
    int N = B_bin.size(1);  

    // Step 2: Transpose B_matrix because in neon_binary_gemm we need B to be in column major order
    torch::Tensor B_trans = B_bin_int8.transpose(0, 1);

    // Step 3 : Pack the binary matrices
    start = std::chrono::high_resolution_clock::now();
    torch::Tensor A_packed = pack_binary_matrix_SIMD128(A_bin_int8);
    torch::Tensor B_packed = pack_binary_matrix_SIMD128(B_trans);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> pack_time = end - start;

    // TODO: make sure the type is lower or something
    auto output =  torch::zeros({M, N}, torch::kInt32);


    start = std::chrono::high_resolution_clock::now();
    // std::cout << "Neon res:" << std::endl;
    // Step 4 : Call the neon_binary_gemm function
    neon_binary_gemm_tiled_old(A_packed.data_ptr<uint8_t>(),
                            B_packed.data_ptr<uint8_t>(),
                            output.data_ptr<int32_t>(),
                            M, K, N);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gemm_time = end - start;
    std::cout << "Gemm Elapsed Time: " << gemm_time.count() << " ms\n";

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_time = total_end - total_start;
    std::cout << "Total Elapsed Time: " << total_time.count() << " ms\n";

    double check_time1_fraction = (check_time1.count() / total_time.count()) * 100.0;
    double check_time2_fraction = (check_time2.count() / total_time.count()) * 100.0;
    double pack_fraction = (pack_time.count() / total_time.count()) * 100.0;
    double gemm_fraction = (gemm_time.count() / total_time.count()) * 100.0;

    std::cout << "check_time1_fraction: " << check_time1_fraction << "%\n";
    std::cout << "check_time2_fraction: " << check_time2_fraction << "%\n";
    std::cout << "bit packing: " << pack_fraction << "%\n";
    std::cout << "neon_binary_gemm: " << gemm_fraction << "%\n";

    std::cout << "===== binary_matmul128_tiled_old ======" << std::endl;

    return output;
}





torch::Tensor binary_matmul_float_bitpack(torch::Tensor A_bin,
                                    torch::Tensor B_bin) {

    // Note to reader: I am doing some pretty dangerous pointer manipulation here 
    //                 in order to squeeze out the best performance possible. I also 
    //                 omitted safety check and assume that the input tensors are in the 
    //                 correct format (i.e., only -1 and 1) and shape. 
    auto total_start = std::chrono::high_resolution_clock::now();
   
    // Step 1 : Check the input tensors, any type is accepted as of now (float, double, int, etc.)
    // auto start00 = std::chrono::high_resolution_clock::now();
    TORCH_CHECK(A_bin.size(1) == B_bin.size(0), "Expected second dimension of A_bin to match first dimension of B_bin");   
    int K = A_bin.size(1);
    // auto end00 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration00 = end00 - start00;


    int M = A_bin.size(0);
    int N = B_bin.size(1);  

    // Step 2: Transpose B_matrix because in neon_binary_gemm we need B to be in column major order
    // auto start000 = std::chrono::high_resolution_clock::now();
    torch::Tensor B_trans = B_bin.transpose(0, 1);
    // auto end000 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration000 = end000 - start000;
    // std::cout << "contiguous: " << duration000.count() << " ms\n";

    // Step 3 : Pack the binary matrices
    auto start1 = std::chrono::high_resolution_clock::now();
    torch::Tensor A_packed = pack_binary_matrix_SIMD128_float_input(A_bin);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration1 = end1 - start1;
    std::cout << "Pack A: " << duration1.count() << " ms\n";

    auto start2 = std::chrono::high_resolution_clock::now();
    torch::Tensor B_packed = pack_binary_matrix_SIMD128_float_input(B_trans);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration2 = end2 - start2;
    std::cout << "Pack B: " << duration2.count() << " ms\n";

    // TODO: make sure the type is lower or something

    // auto start22 = std::chrono::high_resolution_clock::now();
    auto output =  torch::zeros({N, M}, torch::kInt32);
    // auto end22 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration22 = end22 - start22;


    // auto start = std::chrono::high_resolution_clock::now();
    // Step 1 : Check the input tensors, any type is accepted as of now (float, double, int, etc.)

    // Step 4 : Call the neon_binary_gemm function
    auto start = std::chrono::high_resolution_clock::now();
    neon_binary_gemm(A_packed.data_ptr<uint8_t>(),
                     B_packed.data_ptr<uint8_t>(),
                     output.data_ptr<int32_t>(),
                     M, K, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration3 = end - start;
    std::cout << "Gemm Elapsed Time: " << duration3.count() << " ms\n";
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_time = total_end - total_start;
    std::cout << "Total Elapsed Time: " << total_time.count() << " ms\n";

    // double d00_f = (duration00.count() / total_time.count()) * 100.0;
    // double d000_f = (duration000.count() / total_time.count()) * 100.0;
    double d1_f = (duration1.count() / total_time.count()) * 100.0;
    double d2_f = (duration2.count() / total_time.count()) * 100.0;
    // double d22_f = (duration22.count() / total_time.count()) * 100.0;
    double d3_f = (duration3.count() / total_time.count()) * 100.0;
    // std::cout << "d00_f: " << d00_f << "%\n";
    // std::cout << "d000_f: " << d000_f << "%\n";
    std::cout << "d1_f: " << d1_f << "%\n";
    std::cout << "d2_f: " << d2_f << "%\n";
    // std::cout << "d22_f: " << d22_f << "%\n";
    std::cout << "d3_f: " << d3_f << "%\n";

    // std::cout << "check_time1_fraction: " << check_time1_fraction << "%\n";
    // std::cout << "check_time2_fraction: " << check_time2_fraction << "%\n";
    // std::cout << "bit packing: " << pack_fraction << "%\n";
    // std::cout << "neon_binary_gemm: " << gemm_fraction << "%\n";

    return output;
}

torch::Tensor binary_matmul_tiled_with_params(torch::Tensor A_bin,
                                             torch::Tensor B_bin,
                                             int tile_m, int tile_n, int tile_k) {

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
                                     tile_m, tile_n, tile_k);

    return output;
}
