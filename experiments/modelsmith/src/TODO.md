
# TODO:

1. BinaryLinear should pack rows of binary weights into 8-bit or 32-bit data to achieve up to 32x storage saving [ ]
 - Bit packing might increase latency during training, but since it doesn't need to be modified during inference it will not increase latency during inference
 - However, incoming activations MIGHT not already be bitpacked, and due to that we might need to bitpack activations during inference, and that can introduce a little bit of latency
2. The custom kernel (binary_matmul) should be modified to work with 1., as now the weights and potentially activations are bitpacked before going into the cpp kernel [ ]
3. Binary Mat-mul optimizations:
a. Bitpack matrices even if they are not a multiple of 8/16 [ ]
b. If rows are not a multiple of 128, zero-pad it and Use SIMD128 like normal [ ]
c. Implement SIMD64 for matmul, instead of using SIMD128 directly this can save enrgy and time [ ] 
d. If rows are not a multiple of 64, zero-pad it and Use SIMD64 like normal
e. Explore using aligned_alloc instead of raw pointers [ ]
f. Explore aligning pointers to 16 bits or something [ ]
g. Remove checking of elements, or at least find a way to make them more efficient [ ]
h. do we really need 
auto A_bin_int8 = A_bin.to(torch::kInt8).contiguous();
auto B_bin_int8 = B_bin.to(torch::kInt8).contiguous();
? [ ]

Other potential optimization:
Query Cache Sizes Dynamically and use different tile sizes depends on the machine
Loop unrolling to reduce loop overhead
Prefetching data 
Use Strassen's algorithm (if haven't already)
/*

// Strassen matrix multiplication
Matrix strassen(const Matrix& A, const Matrix& B) {
    int n = A.size();
    if (n <= 2) {
        return naive_mul(A, B);  // base case
    }

    int half = n / 2;
    Matrix A11(half, std::vector<int>(half));
    Matrix A12(half, std::vector<int>(half));
    Matrix A21(half, std::vector<int>(half));
    Matrix A22(half, std::vector<int>(half));
    Matrix B11(half, std::vector<int>(half));
    Matrix B12(half, std::vector<int>(half));
    Matrix B21(half, std::vector<int>(half));
    Matrix B22(half, std::vector<int>(half));

    // Divide matrices
    for (int i = 0; i < half; ++i) {
        for (int j = 0; j < half; ++j) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + half];
            A21[i][j] = A[i + half][j];
            A22[i][j] = A[i + half][j + half];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + half];
            B21[i][j] = B[i + half][j];
            B22[i][j] = B[i + half][j + half];
        }
    }

    // M1 to M7
    Matrix M1 = strassen(add(A11, A22), add(B11, B22));
    Matrix M2 = strassen(add(A21, A22), B11);
    Matrix M3 = strassen(A11, subtract(B12, B22));
    Matrix M4 = strassen(A22, subtract(B21, B11));
    Matrix M5 = strassen(add(A11, A12), B22);
    Matrix M6 = strassen(subtract(A21, A11), add(B11, B12));
    Matrix M7 = strassen(subtract(A12, A22), add(B21, B22));

    // Assemble result matrix
    Matrix C(n, std::vector<int>(n));
    for (int i = 0; i < half; ++i) {
        for (int j = 0; j < half; ++j) {
            C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C[i][j + half] = M3[i][j] + M5[i][j];
            C[i + half][j] = M2[i][j] + M4[i][j];
            C[i + half][j + half] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }
    }

    return C;
}
*/

i. Look into JIT (Just-Tn-Time) compiling 
j. In machine learning the matrix dimensions are predictable, because of this, we can compile the matrix multiplication code to optimize for the dimension (e.g., more cache-friendly and stuff.)