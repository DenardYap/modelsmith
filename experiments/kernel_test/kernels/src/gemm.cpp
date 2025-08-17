#include <iostream>
#include <arm_neon.h>
#include <vector>
#include <stdint.h>
#include <random>
#include <chrono>
#include <arm_neon.h>
#include <cassert>
#include <omp.h>

#define TILE_M 8
#define TILE_N 4
#define TILE_K 128

#define TILE_K_BYTES (TILE_K / 8)

// /**
//  * 
//  * Args:
//  * 
//  * A_packed : 2d matrix of size MxK
//  * B_packed : 2d matrix of size NxK (transposed)
//  * 
//  * EXAMPLE:
//  \\ a =  -1 1 -1 1 -1 1 -1   1
//  \\ b =   1 -1 -1 1 -1 -1 -1 1
//  \\ dot(a, b) = -1 -1 +1 +1 +1 -1 +1 +1 = 2

//  \\ a =  0 1 0 1 0 1 0 1
//  \\ b =  1 0 0 1 0 0 0 1
//  \\ XNOR(a, b) = 0 0 1 1 1 0 1 1
//  \\ POPCOUNT(XNOR(a, b)) = 2 * 5 - 8 = 2 

//  * TODO: make B_packed less confusing
//  * NOTE: This function will only be faster than normal gemm if 
//  * K > 128, the larger the K, the faster it is compared to normal gemm
// */
// void neon_binary_gemm(const uint8_t* A_packed,
//                       const uint8_t* B_packed,
//                       int32_t* C,
//                       int M, int K, int N)
// {   
//     int packed_K = (K + 7) / 8;  // Number of bytes per row/column

//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < N; ++j) {

//             uint32_t sum = 0;

//             int k_byte = 0;
//             for (; k_byte <= packed_K - 16; k_byte += 16) {

                
//                 // Intrinsic functions 
//                 uint8x16_t a_vec = vld1q_u8(A_packed + i * packed_K + k_byte);
//                 uint8x16_t b_vec = vld1q_u8(B_packed + j * packed_K + k_byte);

//                 // XNOR: ~(a ^ b)
//                 uint8x16_t xnor_vec = vmvnq_u8(veorq_u8(a_vec, b_vec));
//                 uint8x16_t popcnt = vcntq_u8(xnor_vec);
//                 auto count = 0; 
//                 #if defined(__aarch64__)  // ARMv8 (64-bit)
//                     count = vaddvq_u8(popcnt);

//                 #else  // ARMv7 (32-bit) fallback

//                     // POPCOUNT
//                     // Sum across lanes (16 lanes of uint8)
//                     uint8x8_t low = vget_low_u8(popcnt);
//                     uint8x8_t high = vget_high_u8(popcnt);
//                     // If low = [a0, a1, a2, a3, a4, a5, a6, a7], the result is:
//                     // [ (a0 + a1), (a2 + a3), (a4 + a5), (a6 + a7) ]
//                     uint16x4_t pair_sum_lo = vpaddl_u8(low);
//                     uint16x4_t pair_sum_hi = vpaddl_u8(high);
                    
//                     // If pair_sum_lo = [b0, b1, b2, b3], the result is:
//                     // [ (b0 + b1), (b2 + b3) ]
//                     uint32x2_t quad_sum_lo = vpaddl_u16(pair_sum_lo);
//                     uint32x2_t quad_sum_hi = vpaddl_u16(pair_sum_hi);

//                     // uint64x1_t octa_sum = vpaddl_u32(vadd_u32(quad_sum_lo, quad_sum_hi));
//                     uint64x1_t octa_sum_lo = vpaddl_u32(quad_sum_lo);
//                     uint64x1_t octa_sum_hi = vpaddl_u32(quad_sum_hi);
//                     uint64x1_t total_sum = vadd_u64(octa_sum_lo, octa_sum_hi);
//                     count = (uint64_t)total_sum;
//                 #endif

//                 // TODO: make this work even for non-128-bit registers
//                 sum += 2 * count - 8 * 16;
                
//                 // If this is the last block, it's possible that the matrix is not
//                 // multiple of 8, and therefore the 16th vector is zero-padded 
//                 // For example, for K = 121, the last 7 bits are zero padded, meaning
//                 // 7 ones are contributed to the final sum, we have to subtract it 
//                 if (k_byte + 16 >= packed_K ) {
//                     int padded_bits = packed_K * 8 - K;
//                     sum -= padded_bits;
//                 }
//             }
            
            
//             // Handle leftovers (less than 16 bytes)
//             for (; k_byte < packed_K; ++k_byte) {
//                 uint8_t a_byte = A_packed[i * packed_K + k_byte];
//                 uint8_t b_byte = B_packed[j * packed_K + k_byte];
//                 uint8_t xnor_byte = ~(a_byte ^ b_byte);

//                 // For the last byte, only count the valid bits
//                 int bit_offset = k_byte * 8;

//                 int nbits = std::min(8, K - bit_offset);
//                 uint8_t mask = ((1 << nbits) - 1) << (8 - nbits);
//                 int popc = __builtin_popcount(xnor_byte & mask);
//                 sum += 2 * popc - nbits;
//             }

//             // *(C + i * N + j) = sum;
//             *(C + i * N + j) = sum;
            
//         }
//     }
// }

// Perform dot product between two vectors using vectorized instructions
// void neon_sdot(int n, const uint8_t *v1, const uint8_t *v2){

// }

// void neon_sdot(int n, const float *v1, const float *v2){
// }


void neon_binary_gemm(const uint8_t* A_packed,
                      const uint8_t* B_packed,
                      int32_t* C,
                      int M, int K, int N)
{   

    int packed_K = (K + 7) / 8;  // Number of bytes per row/column

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {

            uint32_t sum = 0;

            int k_byte = 0;
            for (; k_byte <= packed_K - 16; k_byte += 16) {

                
                // Intrinsic functions 
                uint8x16_t a_vec = vld1q_u8(A_packed + i * packed_K + k_byte);
                uint8x16_t b_vec = vld1q_u8(B_packed + j * packed_K + k_byte);

                // XNOR: ~(a ^ b)
                uint8x16_t xnor_vec = vmvnq_u8(veorq_u8(a_vec, b_vec));
                uint8x16_t popcnt = vcntq_u8(xnor_vec);

                auto count = 0; 
                #if defined(__aarch64__)  // ARMv8 (64-bit)
                    count = vaddvq_u8(popcnt);
                #else  // ARMv7 (32-bit) fallback

                    // POPCOUNT
                    // Sum across lanes (16 lanes of uint8)
                    uint8x8_t low = vget_low_u8(popcnt);
                    uint8x8_t high = vget_high_u8(popcnt);
                    // If low = [a0, a1, a2, a3, a4, a5, a6, a7], the result is:
                    // [ (a0 + a1), (a2 + a3), (a4 + a5), (a6 + a7) ]
                    uint16x4_t pair_sum_lo = vpaddl_u8(low);
                    uint16x4_t pair_sum_hi = vpaddl_u8(high);
                    
                    // If pair_sum_lo = [b0, b1, b2, b3], the result is:
                    // [ (b0 + b1), (b2 + b3) ]
                    uint32x2_t quad_sum_lo = vpaddl_u16(pair_sum_lo);
                    uint32x2_t quad_sum_hi = vpaddl_u16(pair_sum_hi);

                    // uint64x1_t octa_sum = vpaddl_u32(vadd_u32(quad_sum_lo, quad_sum_hi));
                    uint64x1_t octa_sum_lo = vpaddl_u32(quad_sum_lo);
                    uint64x1_t octa_sum_hi = vpaddl_u32(quad_sum_hi);
                    uint64x1_t total_sum = vadd_u64(octa_sum_lo, octa_sum_hi);
                    count = (uint64_t)total_sum;
                #endif

                // TODO: make this work even for non-128-bit registers
                sum += 2 * count - 8 * 16;
                
                // If this is the last block, it's possible that the matrix is not
                // multiple of 8, and therefore the 16th vector is zero-padded 
                // For example, for K = 121, the last 7 bits are zero padded, meaning
                // 7 ones are contributed to the final sum, we have to subtract it 
                if (k_byte + 16 >= packed_K ) {
                    int padded_bits = packed_K * 8 - K;
                    sum -= padded_bits;
                }
            } // k_byte
            
            
            // Handle leftovers (less than 16 bytes)
            for (; k_byte < packed_K; ++k_byte) {
                uint8_t a_byte = A_packed[i * packed_K + k_byte];
                uint8_t b_byte = B_packed[j * packed_K + k_byte];
                uint8_t xnor_byte = ~(a_byte ^ b_byte);

                // For the last byte, only count the valid bits
                int bit_offset = k_byte * 8;

                int nbits = std::min(8, K - bit_offset);
                uint8_t mask = ((1 << nbits) - 1) << (8 - nbits);
                int popc = __builtin_popcount(xnor_byte & mask);
                sum += 2 * popc - nbits;
            } // k_byte
            
            *(C + i * N + j) = sum;
            
        } // j 
    } // i 
}
/**
 * 
 * [1 2 3 4   [1 2
 *  2 3 4 5    2 3
 *  3 4 5 6]   3 4
 *             4 5]
 */
// neon_binary_gemm only leverage SIMD when the input 
// row size is a multiple of 128, but we can actually 
// improve it by zero-padding A_packed and B_packed 
// with zeros when it's not a multiple of 128, or at least,
// use 64-bit instruction (vld1_u8 instead of vld1q_u8) instructions

/** Method 1:
 *  Load one vector and cache them instead of keep reloading them, 
 *  e.g. the first row first, then comptue all the columns 
 * 
 *  Method 2:
 *  "Wrap Around" SIMD, load elements that are not on the same rows into 
 *  SIMD, but need to figure out exactly how to get it to work
 * 
 *  1 2 3 4 2 3 
 *  + + + + * * 
 * 
 *  Method 3: 
 *  Zero-pad inputs, or load in the garbage entries but ensure to shift them when
 *  computing C[i][j] 
 * 
 *  Imagine we are using an 8-bit SIMD, if row = [1, 2, 3, 4, 5, 6, 7], then using 
 *  the previous approach we won't load row into SIMD, because it's not a multiple of 8
 * 
 *  But what if we just do it anyway? vld1q_u8(row + 0) = {1, 2, 3, 4, 5, 6, 7, ?}
 *  Sicne the last entry is technically undefined, we need to construct a zero-mask
 *  mask = {1, 2, 3, 4, 5, 6, 7, 0}, then do vandq_u8(vld1q_u8(row + 0), mask) 
 *  = {1, 2, 3, 4, 5, 6, 7, 0}, now when computing this particular entry, the 0 will be ignored
 * 
 */
void neon_binary_gemm_improved(const uint8_t* A_packed,
                      const uint8_t* B_packed,
                      int32_t* C,
                      int M, int K, int N)
{   
    // VERY IMPORTANT: C is required to be an MxN ZERO-tensor here.
    int packed_K = (K + 7) / 8;  // Number of bytes per row/column

    /**
     * Example (for simplicity assume the SIMD is 8-bit)
     * 
     * [1 2 3 4 5 6 7 8 9  [1 1
     *  2 3 4 5 6 7 8 9 1]  1 2 
     *                      2 3 
     *                      3 4  
     *                      6 7 
     *                      9 1 
     *                      1 2 
     *                      3 1  
     *                      2 3] 
     * 
     * C = [0 0; 0 0]
     * for i in M:
     *   First load first row, first part:
     *      a_vec = {1 2 3 4 5 6 7 8}
     *      for j in N:
     *         Then load first col, first part:
     *           b_vec = {1 1 2 3 6 9 1 3}
     *           sum = 1 * 1 + 2 *1 + 3 * 2 + ... + 8 * 3 = 136
     *           C[0][0] += sum
     *         Then load second col, first part:
     *           b_vec = {1 2 3 4 7 1 2 1}
     *           sum = 1 * 1 + 2 * 2 + 3 * 3 + ... + 8 * 3 = 93
     *           C[0][1] += sum
     *       * C = [136 93; 0 0]
     *   Then load first row, second part (leftover):
     *       a_vec = {9}
     *         Then load first col, second part:
     *           b_vec = {2} 
     *           sum = 9 * 2 
     *           C[0][0] += 18 
     *         Then load second  col, second part:
     *           b_vec = {3} 
     *           sum = 9 * 3 
     *           C[0][0] += 27
     *       * C = [154 120; 0 0]
     *      
     *  ...
     * 
     * 
     */
    
    for (int i = 0; i < M; ++i) {
        int k_byte = 0;
        // Case 1: Non-leftover part
        for (; k_byte <= packed_K - 16; k_byte += 16) {
            uint8x16_t a_vec = vld1q_u8(A_packed + i * packed_K + k_byte);
            /** Two options here:
             *  
             *  a. Maintain a list of N-length, and update the ith row only after 
             *     the two inner loop is done 
             *  
             *  -> Pros: Since this N-length list is gonna be smaller, it likely will fit into 
             *           a lower level cache = better caching 
             *  -> Cons: Slightly more write 
             * 
             *  b. Just write to C partially every loop 
             * 
             *  -> Pros: easier to code 
             *  -> Cons: might be less efficient as we are not leveraging L1/L2-cache 
             * 
             */
            for (int j = 0; j < N; ++j) {
                uint8x16_t b_vec = vld1q_u8(B_packed + j * packed_K + k_byte);

                // XNOR: ~(a ^ b)
                uint8x16_t xnor_vec = vmvnq_u8(veorq_u8(a_vec, b_vec));
                uint8x16_t popcnt = vcntq_u8(xnor_vec);
                auto count = 0; 
                #if defined(__aarch64__)  // ARMv8 (64-bit)
                    count = vaddvq_u8(popcnt);
                #else  // ARMv7 (32-bit) fallback

                    // POPCOUNT
                    // Sum across lanes (16 lanes of uint8)
                    uint8x8_t low = vget_low_u8(popcnt);
                    uint8x8_t high = vget_high_u8(popcnt);
                    // If low = [a0, a1, a2, a3, a4, a5, a6, a7], the result is:
                    // [ (a0 + a1), (a2 + a3), (a4 + a5), (a6 + a7) ]
                    uint16x4_t pair_sum_lo = vpaddl_u8(low);
                    uint16x4_t pair_sum_hi = vpaddl_u8(high);
                    
                    // If pair_sum_lo = [b0, b1, b2, b3], the result is:
                    // [ (b0 + b1), (b2 + b3) ]
                    uint32x2_t quad_sum_lo = vpaddl_u16(pair_sum_lo);
                    uint32x2_t quad_sum_hi = vpaddl_u16(pair_sum_hi);

                    // uint64x1_t octa_sum = vpaddl_u32(vadd_u32(quad_sum_lo, quad_sum_hi));
                    uint64x1_t octa_sum_lo = vpaddl_u32(quad_sum_lo);
                    uint64x1_t octa_sum_hi = vpaddl_u32(quad_sum_hi);
                    uint64x1_t total_sum = vadd_u64(octa_sum_lo, octa_sum_hi);
                    count = (uint64_t)total_sum;
                #endif

                // TODO: make this work even for non-128-bit registers
                uint32_t sum = 2 * count - 8 * 16;
                
                // If this is the last block, it's possible that the matrix is not
                // multiple of 8, and therefore the 16th vector is zero-padded 
                // For example, for K = 121, the last 7 bits are zero padded, meaning
                // 7 ones are contributed to the final sum, we have to subtract it 
                if ( k_byte + 16 >= packed_K ) {
                    int padded_bits = packed_K * 8 - K;
                    sum -= padded_bits;
                }


                // VERY IMPORTANT: C is required to be an MxN ZERO-tensor here.
                *(C + i * N + j) += sum;
            }  // N
        } // k_byte

        // Case 2: Leftover part
        for (int j = 0; j < N; ++j) {

            uint32_t sum = 0; 
            int k_byte2 = k_byte;
            for (; k_byte2 < packed_K; ++k_byte2) {

                uint8_t a_byte = A_packed[i * packed_K + k_byte2];
                uint8_t b_byte = B_packed[j * packed_K + k_byte2];
                uint8_t xnor_byte = ~(a_byte ^ b_byte);

                // For the last byte, only count the valid bits
                int bit_offset = k_byte2 * 8;

                int nbits = std::min(8, K - bit_offset);
                uint8_t mask = ((1 << nbits) - 1) << (8 - nbits);
                int popc = __builtin_popcount(xnor_byte & mask);
                sum += 2 * popc - nbits;
            } // k_byte2
            
            *(C + i * N + j) += sum;
        } // N

    } // M
}

void neon_binary_gemm_tiled_old(const uint8_t* A_packed,
                            const uint8_t* B_packed,
                            int32_t* C,
                            int M, int K, int N)
{   
    int packed_K = (K + 7) / 8;  // Number of bytes per row/column

    // TODO: find out the best tile size, prob differ machine to machine
    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < M; i0 += TILE_M) {
        for (int j0 = 0; j0 < N; j0 += TILE_N) {
            int32_t acc[TILE_M][TILE_N] = {0};
            int k_byte = 0;
            for (; k_byte <= packed_K - TILE_K_BYTES; k_byte += TILE_K_BYTES) {
                for (int ti = 0; ti < std::min(TILE_M, M - i0); ++ti) {
                    // Rewrite the condition above by setting ti = i0
                    
                    uint8x16_t a_vec = vld1q_u8(A_packed + (i0 + ti) * packed_K + k_byte);

                    for (int tj = 0; tj < std::min(TILE_N, N - j0); ++tj) {
                        uint8x16_t b_vec = vld1q_u8(B_packed + (j0 + tj) * packed_K + k_byte);

                        // XNOR then popcount
                        uint8x16_t xnor = vmvnq_u8(veorq_u8(a_vec, b_vec));
                        uint8x16_t popcnt = vcntq_u8(xnor);
                        auto count = 0; 

                        #if defined(__aarch64__)  // ARMv8 (64-bit)
                            count = vaddvq_u8(popcnt);

                        #else  // ARMv7 (32-bit) fallback

                            // POPCOUNT
                            // Sum across lanes (16 lanes of uint8)
                            uint8x8_t low = vget_low_u8(popcnt);
                            uint8x8_t high = vget_high_u8(popcnt);
                            // If low = [a0, a1, a2, a3, a4, a5, a6, a7], the result is:
                            // [ (a0 + a1), (a2 + a3), (a4 + a5), (a6 + a7) ]
                            uint16x4_t pair_sum_lo = vpaddl_u8(low);
                            uint16x4_t pair_sum_hi = vpaddl_u8(high);
                            
                            // If pair_sum_lo = [b0, b1, b2, b3], the result is:
                            // [ (b0 + b1), (b2 + b3) ]
                            uint32x2_t quad_sum_lo = vpaddl_u16(pair_sum_lo);
                            uint32x2_t quad_sum_hi = vpaddl_u16(pair_sum_hi);

                            // uint64x1_t octa_sum = vpaddl_u32(vadd_u32(quad_sum_lo, quad_sum_hi));
                            uint64x1_t octa_sum_lo = vpaddl_u32(quad_sum_lo);
                            uint64x1_t octa_sum_hi = vpaddl_u32(quad_sum_hi);
                            uint64x1_t total_sum = vadd_u64(octa_sum_lo, octa_sum_hi);
                            count = (uint64_t)total_sum;
                        #endif

                        acc[ti][tj] += 2 * count - 8 * TILE_K_BYTES;
                        if (k_byte + TILE_K_BYTES >= packed_K ) {
                            int padded_bits = packed_K * 8 - K;
                            acc[ti][tj] -= padded_bits;
                        }
                    }
                }
            }

            for (; k_byte < packed_K; ++k_byte) {
                for (int ti = 0; ti < TILE_M && (i0 + ti) < M; ++ti) {
                    for (int tj = 0; tj < TILE_N && (j0 + tj) < N; ++tj) {
                        uint8_t a_byte = A_packed[(i0 + ti) * packed_K + k_byte];
                        uint8_t b_byte = B_packed[(j0 + tj) * packed_K + k_byte];
                        uint8_t xnor_byte = ~(a_byte ^ b_byte);

                        // For the last byte, only count the valid bits
                        int bit_offset = k_byte * 8;

                        int nbits = std::min(8, K - bit_offset);
                        uint8_t mask = ((1 << nbits) - 1) << (8 - nbits);
                        int popc = __builtin_popcount(xnor_byte & mask);
                        acc[ti][tj] += 2 * popc - nbits;
                    }

                }
            }

            for (int ti = 0; ti < TILE_M && (i0 + ti) < M; ++ti) {
                for (int tj = 0; tj < TILE_N && (j0 + tj) < N; ++tj) {
                    C[(i0 + ti) * N + (j0 + tj)] = acc[ti][tj];
                }
            }

            
        }
    }
}



void neon_binary_gemm_tiled(const uint8_t* A_packed,
                            const uint8_t* B_packed,
                            int32_t* C,
                            int M, int K, int N)
{   
    int packed_K = (K + 7) / 8;  // Number of bytes per row/column
    
    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < M; i0 += TILE_M) {
        for (int j0 = 0; j0 < N; j0 += TILE_N) {
            
            int32_t acc[TILE_M][TILE_N] = {0};

            for (int k0 = 0; k0 < packed_K; k0 += TILE_K_BYTES) {
                for (int ti = 0; ti < TILE_M; ++ti) {
                    if (i0 + ti >= M) continue;
                    
                    const uint8_t* a_ptr = A_packed + (i0 + ti) * packed_K + k0;
                    
                    for (int tj = 0; tj < TILE_N; ++tj) {
                        if (j0 + tj >= N) continue;

                        const uint8_t* b_ptr = B_packed + (j0 + tj) * packed_K + k0;
                        
                        uint32x4_t sum_vec = vdupq_n_u32(0);

                        for (int k_byte = 0; k_byte < TILE_K_BYTES; k_byte += 16) {
                            if (k0 + k_byte >= packed_K) break;

                            uint8x16_t a_vec = vld1q_u8(a_ptr + k_byte);
                            uint8x16_t b_vec = vld1q_u8(b_ptr + k_byte);

                            uint8x16_t xnor_vec = vmvnq_u8(veorq_u8(a_vec, b_vec));
                            
                            uint8x16_t popcnt = vcntq_u8(xnor_vec);
                            
                            #if defined(__aarch64__)
                                acc[ti][tj] += (int32_t)vaddvq_u8(popcnt);
                            #else
                                // Fallback for ARMv7
                                uint16x8_t half_sum = vpaddlq_u8(popcnt);
                                uint32x4_t quarter_sum = vpaddlq_u16(half_sum);
                                uint64x2_t eighth_sum = vpaddlq_u32(quarter_sum);
                                acc[ti][tj] += vgetq_lane_u64(eighth_sum, 0) + vgetq_lane_u64(eighth_sum, 1);
                        #endif
                        }
                    }
                }
            }

            // Write accumulated results to C
            for (int ti = 0; ti < TILE_M; ++ti) {
                if (i0 + ti >= M) continue;
                for (int tj = 0; tj < TILE_N; ++tj) {
                    if (j0 + tj >= N) continue;
                    int k_rem = K % (TILE_K);
                    if (k_rem == 0) k_rem = TILE_K;
                    int popcnt_sum = acc[ti][tj];
                    C[(i0 + ti) * N + (j0 + tj)] = 2 * popcnt_sum - K;
                }
            }
        }
    }
}



void neon_binary_gemm_tiled_unrolled(const uint8_t* A_packed,
                            const uint8_t* B_packed,
                            int32_t* C,
                            int M, int K, int N)
{   
    int packed_K = (K + 7) / 8;  // Number of bytes per row/column
    int i_unroll = 4;
    int j_unroll = 4;
    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < M; i0 += TILE_M) {
        for (int j0 = 0; j0 < N; j0 += TILE_N) {
            int32_t acc[TILE_M][TILE_N] = {0};
            // TODO: handle leftover
            for (int ti = 0; ti < std::min(TILE_M, M - i0); ti += i_unroll) {

                for (int tj = 0; tj < std::min(TILE_N, N - j0); tj += j_unroll) {
                    int k_byte = 0;

                    for (; k_byte <= packed_K - TILE_K_BYTES; k_byte += TILE_K_BYTES) {
                        uint8x16_t a_vec0 = vld1q_u8(A_packed + (i0 + (ti + 0)) * packed_K + k_byte);
                        uint8x16_t b_vec0 = vld1q_u8(B_packed + (j0 + (tj + 0)) * packed_K + k_byte);

                        uint8x16_t a_vec1 = vld1q_u8(A_packed + (i0 + (ti + 1)) * packed_K + k_byte);
                        uint8x16_t b_vec1 = vld1q_u8(B_packed + (j0 + (tj + 1)) * packed_K + k_byte);

                        uint8x16_t a_vec2 = vld1q_u8(A_packed + (i0 + (ti + 2)) * packed_K + k_byte);
                        uint8x16_t b_vec2 = vld1q_u8(B_packed + (j0 + (tj + 2)) * packed_K + k_byte);

                        uint8x16_t a_vec3 = vld1q_u8(A_packed + (i0 + (ti + 3)) * packed_K + k_byte);
                        uint8x16_t b_vec3 = vld1q_u8(B_packed + (j0 + (tj + 3)) * packed_K + k_byte);

                        #if defined(__aarch64__)  // ARMv8 (64-bit)
                            auto count00 = vaddvq_u8(vcntq_u8(vmvnq_u8(veorq_u8(a_vec0, b_vec0))));
                            auto count01 = vaddvq_u8(vcntq_u8(vmvnq_u8(veorq_u8(a_vec0, b_vec1))));
                            auto count02 = vaddvq_u8(vcntq_u8(vmvnq_u8(veorq_u8(a_vec0, b_vec2))));
                            auto count03 = vaddvq_u8(vcntq_u8(vmvnq_u8(veorq_u8(a_vec0, b_vec3))));

                            auto count10 = vaddvq_u8(vcntq_u8(vmvnq_u8(veorq_u8(a_vec1, b_vec0))));
                            auto count11 = vaddvq_u8(vcntq_u8(vmvnq_u8(veorq_u8(a_vec1, b_vec1))));
                            auto count12 = vaddvq_u8(vcntq_u8(vmvnq_u8(veorq_u8(a_vec1, b_vec2))));
                            auto count13 = vaddvq_u8(vcntq_u8(vmvnq_u8(veorq_u8(a_vec1, b_vec3))));

                            auto count20 = vaddvq_u8(vcntq_u8(vmvnq_u8(veorq_u8(a_vec2, b_vec0))));
                            auto count21 = vaddvq_u8(vcntq_u8(vmvnq_u8(veorq_u8(a_vec2, b_vec1))));
                            auto count22 = vaddvq_u8(vcntq_u8(vmvnq_u8(veorq_u8(a_vec2, b_vec2))));
                            auto count23 = vaddvq_u8(vcntq_u8(vmvnq_u8(veorq_u8(a_vec2, b_vec3))));

                            auto count30 = vaddvq_u8(vcntq_u8(vmvnq_u8(veorq_u8(a_vec3, b_vec0))));
                            auto count31 = vaddvq_u8(vcntq_u8(vmvnq_u8(veorq_u8(a_vec3, b_vec1))));
                            auto count32 = vaddvq_u8(vcntq_u8(vmvnq_u8(veorq_u8(a_vec3, b_vec2))));
                            auto count33 = vaddvq_u8(vcntq_u8(vmvnq_u8(veorq_u8(a_vec3, b_vec3))));

                        #else  // ARMv7 (32-bit) fallback

                            // POPCOUNT
                            // Sum across lanes (16 lanes of uint8)
                            uint8x8_t low = vget_low_u8(popcnt0);
                            uint8x8_t high = vget_high_u8(popcnt0);
                            // If low = [a0, a1, a2, a3, a4, a5, a6, a7], the result is:
                            // [ (a0 + a1), (a2 + a3), (a4 + a5), (a6 + a7) ]
                            uint16x4_t pair_sum_lo = vpaddl_u8(low);
                            uint16x4_t pair_sum_hi = vpaddl_u8(high);
                            
                            // If pair_sum_lo = [b0, b1, b2, b3], the result is:
                            // [ (b0 + b1), (b2 + b3) ]
                            uint32x2_t quad_sum_lo = vpaddl_u16(pair_sum_lo);   
                            uint32x2_t quad_sum_hi = vpaddl_u16(pair_sum_hi);
  
                            // uint64x1_t octa_sum = vpaddl_u32(vadd_u32(quad_sum_lo, quad_sum_hi));
                            uint64x1_t octa_sum_lo = vpaddl_u32(quad_sum_lo);
                            uint64x1_t octa_sum_hi = vpaddl_u32(quad_sum_hi);
                            uint64x1_t total_sum = vadd_u64(octa_sum_lo, octa_sum_hi);
                            count0 = (uint64_t)total_sum;
                        #endif

                        acc[ti + 0][tj + 0] += 2 * count00 - 8 * TILE_K_BYTES;
                        acc[ti + 0][tj + 1] += 2 * count01 - 8 * TILE_K_BYTES;
                        acc[ti + 0][tj + 2] += 2 * count02 - 8 * TILE_K_BYTES;
                        acc[ti + 0][tj + 3] += 2 * count03 - 8 * TILE_K_BYTES;

                        acc[ti + 1][tj + 0] += 2 * count10 - 8 * TILE_K_BYTES;
                        acc[ti + 1][tj + 1] += 2 * count11 - 8 * TILE_K_BYTES;
                        acc[ti + 1][tj + 2] += 2 * count12 - 8 * TILE_K_BYTES;
                        acc[ti + 1][tj + 3] += 2 * count13 - 8 * TILE_K_BYTES;

                        acc[ti + 2][tj + 0] += 2 * count20 - 8 * TILE_K_BYTES;
                        acc[ti + 2][tj + 1] += 2 * count21 - 8 * TILE_K_BYTES;
                        acc[ti + 2][tj + 2] += 2 * count22 - 8 * TILE_K_BYTES;
                        acc[ti + 2][tj + 3] += 2 * count23 - 8 * TILE_K_BYTES;

                        acc[ti + 3][tj + 0] += 2 * count30 - 8 * TILE_K_BYTES;
                        acc[ti + 3][tj + 1] += 2 * count31 - 8 * TILE_K_BYTES;
                        acc[ti + 3][tj + 2] += 2 * count32 - 8 * TILE_K_BYTES;
                        acc[ti + 3][tj + 3] += 2 * count33 - 8 * TILE_K_BYTES;

                        if (k_byte + TILE_K_BYTES >= packed_K ) {
                            int padded_bits = packed_K * 8 - K;
                            acc[ti + 0][tj + 0] -= padded_bits;
                            acc[ti + 0][tj + 1] -= padded_bits;
                            acc[ti + 0][tj + 2] -= padded_bits;
                            acc[ti + 0][tj + 3] -= padded_bits;

                            acc[ti + 1][tj + 0] -= padded_bits;
                            acc[ti + 1][tj + 1] -= padded_bits;
                            acc[ti + 1][tj + 2] -= padded_bits;
                            acc[ti + 1][tj + 3] -= padded_bits;

                            acc[ti + 2][tj + 0] -= padded_bits;
                            acc[ti + 2][tj + 1] -= padded_bits;
                            acc[ti + 2][tj + 2] -= padded_bits;
                            acc[ti + 2][tj + 3] -= padded_bits;

                            acc[ti + 3][tj + 0] -= padded_bits;
                            acc[ti + 3][tj + 1] -= padded_bits;
                            acc[ti + 3][tj + 2] -= padded_bits;
                            acc[ti + 3][tj + 3] -= padded_bits;
                        }
                        
                    } // k_byte

                    for (; k_byte < packed_K; ++k_byte) {
                        uint8_t a_byte0 = A_packed[(i0 + (ti + 0)) * packed_K + k_byte];
                        uint8_t a_byte1 = A_packed[(i0 + (ti + 1)) * packed_K + k_byte];
                        uint8_t a_byte2 = A_packed[(i0 + (ti + 2)) * packed_K + k_byte];
                        uint8_t a_byte3 = A_packed[(i0 + (ti + 3)) * packed_K + k_byte];

                        uint8_t b_byte0 = B_packed[(j0 + (tj + 0)) * packed_K + k_byte];
                        uint8_t b_byte1 = B_packed[(j0 + (tj + 1)) * packed_K + k_byte];
                        uint8_t b_byte2 = B_packed[(j0 + (tj + 2)) * packed_K + k_byte];
                        uint8_t b_byte3 = B_packed[(j0 + (tj + 3)) * packed_K + k_byte];

                        // For the last byte, only count the valid bits
                        int bit_offset = k_byte * 8;

                        int nbits = std::min(8, K - bit_offset);
                        uint8_t mask = ((1 << nbits) - 1) << (8 - nbits);

                        acc[ti + 0][tj + 0] += 2 * __builtin_popcount(~(a_byte0 ^ b_byte0) & mask) - nbits;
                        acc[ti + 0][tj + 1] += 2 * __builtin_popcount(~(a_byte0 ^ b_byte1) & mask) - nbits;
                        acc[ti + 0][tj + 2] += 2 * __builtin_popcount(~(a_byte0 ^ b_byte2) & mask) - nbits;
                        acc[ti + 0][tj + 3] += 2 * __builtin_popcount(~(a_byte0 ^ b_byte3) & mask) - nbits;

                        acc[ti + 1][tj + 0] += 2 * __builtin_popcount(~(a_byte1 ^ b_byte0) & mask) - nbits;
                        acc[ti + 1][tj + 1] += 2 * __builtin_popcount(~(a_byte1 ^ b_byte1) & mask) - nbits;
                        acc[ti + 1][tj + 2] += 2 * __builtin_popcount(~(a_byte1 ^ b_byte2) & mask) - nbits;
                        acc[ti + 1][tj + 3] += 2 * __builtin_popcount(~(a_byte1 ^ b_byte3) & mask) - nbits;

                        acc[ti + 2][tj + 0] += 2 * __builtin_popcount(~(a_byte2 ^ b_byte0) & mask) - nbits;
                        acc[ti + 2][tj + 1] += 2 * __builtin_popcount(~(a_byte2 ^ b_byte1) & mask) - nbits;
                        acc[ti + 2][tj + 2] += 2 * __builtin_popcount(~(a_byte2 ^ b_byte2) & mask) - nbits;
                        acc[ti + 2][tj + 3] += 2 * __builtin_popcount(~(a_byte2 ^ b_byte3) & mask) - nbits;

                        acc[ti + 3][tj + 0] += 2 * __builtin_popcount(~(a_byte3 ^ b_byte0) & mask) - nbits;
                        acc[ti + 3][tj + 1] += 2 * __builtin_popcount(~(a_byte3 ^ b_byte1) & mask) - nbits;
                        acc[ti + 3][tj + 2] += 2 * __builtin_popcount(~(a_byte3 ^ b_byte2) & mask) - nbits;
                        acc[ti + 3][tj + 3] += 2 * __builtin_popcount(~(a_byte3 ^ b_byte3) & mask) - nbits;
                        
                    }

                } // tj
            } // ti


            for (int ti = 0; ti < TILE_M && (i0 + ti) < M; ++ti) {
                for (int tj = 0; tj < TILE_N && (j0 + tj) < N; ++tj) {
                    C[(i0 + ti) * N + (j0 + tj)] = acc[ti][tj];
                }
            }

            
        } // j0
    } // i0 
}



void naive_gemm_col_major(const uint8_t* A, 
                const uint8_t* B, 
                float* C, 
                int M, int K, int N) {

    for (int i = 0; i < M; ++i) {  // For each row of A
        for (int j = 0; j < N; ++j) {  // For each column of B
            for (int k = 0; k < K; ++k) {  // Sum the dot product
                *(C + i * N + j) += *(A + i * K + k) * *(B + k * N + j);
            }
        }
    }
}

void naive_gemm_col_major_with_transposed_B(const uint8_t* A, 
                const uint8_t* B, 
                float* C, 
                int M, int K, int N) {
    
    
    for (int i = 0; i < M; ++i) {  // For each row of A
        for (int j = 0; j < N; ++j) {  // For each column of B
            for (int k = 0; k < K; ++k) {  // Sum the dot product
                *(C + i * N + j) += *(A + i * K + k) * *(B + j * N + k);
            }
        }
    }
}


void naive_gemm_row_major_with_transposed_B(const uint8_t* A, 
                const uint8_t* B, 
                float* C, 
                int M, int K, int N) {
    
    
    for (int i = 0; i < M; ++i) {  // For each row of A
        for (int k = 0; k < K; ++k) {  // Sum the dot product
            for (int j = 0; j < N; ++j) {  // For each column of B
                *(C + i * N + j) += *(A + i * K + k) * *(B + j * N + k);
            }
        }
    }
}

void naive_gemm_row_major(const uint8_t* A, 
                const uint8_t* B, 
                double* C, 
                int M, int K, int N) {

    // #pragma omp parallel for
    for (int i = 0; i < M; ++i) { 
        for (int k = 0; k < K; ++k) { 
            for (int j = 0; j < N; ++j) { 
                // VERY INTERESTING BUG HERE: When M = K = N are very large, this can have some error
                *(C + i * N + j) += *(A + i * K + k) * *(B + k * N + j);

            }
        }
    }
}




// void naive_gemm_row_major_unrolled_ij(const uint8_t* A, 
//                 const uint8_t* B, 
//                 float* C, 
//                 int M, int K, int N) {

//     int m_unroll = 4;

//     int i = 0;
//     for (; i < M; i += m_unroll) {  // unroll 4 at a time
//         for (int k = 0; k < K; ++k) {

//             uint8_t a0 = *(A + (i + 0) * K + k);
//             uint8_t a1 = *(A + (i + 1) * K + k);
//             uint8_t a2 = *(A + (i + 2) * K + k);
//             uint8_t a3 = *(A + (i + 3) * K + k);
//             for (int j = 0; j < N; ++j) {  
//                 float b = *(B + k * N + j);
//                 *(C + (i + 0) * N + j) += a0 * b;
//                 *(C + (i + 1) * N + j) += a1 * b;
//                 *(C + (i + 2) * N + j) += a2 * b;
//                 *(C + (i + 3) * N + j) += a3 * b;
//             }

//         }
//     }

// }

// 
/**
 * Packs 4 rows of A starting at row `i` into A_packed: [a0_k0, a1_k0, a2_k0, a3_k0, a0_k1, ...]
 * The reason this function is needed and helpful is because in compute_4x4_tile_packed, we are 
 * originally accessing A in a strided pattern, which is not cache-friendly. By packing it into 
 * continuous memory access like below, it will be more cache-friendly.  
 */
void pack_A_4xK(const uint8_t* A, uint8_t* A_packed, int i, int K, int lda) {

    for (int k = 0; k < K; ++k) {
        A_packed[4 * k + 0] = A[(i + 0) * lda + k];
        A_packed[4 * k + 1] = A[(i + 1) * lda + k];
        A_packed[4 * k + 2] = A[(i + 2) * lda + k];
        A_packed[4 * k + 3] = A[(i + 3) * lda + k];
    }
}



// inline void compute_4x4_tile(const uint8_t* A, const uint8_t* B, double* C, 
//                              int i, int j, int K, int N) {
//     double c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
//     double c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;
//     double c20 = 0.0f, c21 = 0.0f, c22 = 0.0f, c23 = 0.0f;
//     double c30 = 0.0f, c31 = 0.0f, c32 = 0.0f, c33 = 0.0f;

//     for (int k = 0; k < K; ++k) {
//         double a0 = A[(i + 0) * K + k];
//         double a1 = A[(i + 1) * K + k];
//         double a2 = A[(i + 2) * K + k];
//         double a3 = A[(i + 3) * K + k];

//         double b0 = B[k * N + (j + 0)];
//         double b1 = B[k * N + (j + 1)];
//         double b2 = B[k * N + (j + 2)];
//         double b3 = B[k * N + (j + 3)];

//         c00 += a0 * b0;  c01 += a0 * b1;  c02 += a0 * b2;  c03 += a0 * b3;
//         c10 += a1 * b0;  c11 += a1 * b1;  c12 += a1 * b2;  c13 += a1 * b3;
//         c20 += a2 * b0;  c21 += a2 * b1;  c22 += a2 * b2;  c23 += a2 * b3;
//         c30 += a3 * b0;  c31 += a3 * b1;  c32 += a3 * b2;  c33 += a3 * b3;
//     }

//     C[(i + 0) * N + (j + 0)] += c00;
//     C[(i + 0) * N + (j + 1)] += c01;
//     C[(i + 0) * N + (j + 2)] += c02;
//     C[(i + 0) * N + (j + 3)] += c03;

//     C[(i + 1) * N + (j + 0)] += c10;
//     C[(i + 1) * N + (j + 1)] += c11;
//     C[(i + 1) * N + (j + 2)] += c12;
//     C[(i + 1) * N + (j + 3)] += c13;

//     C[(i + 2) * N + (j + 0)] += c20;
//     C[(i + 2) * N + (j + 1)] += c21;
//     C[(i + 2) * N + (j + 2)] += c22;
//     C[(i + 2) * N + (j + 3)] += c23;

//     C[(i + 3) * N + (j + 0)] += c30;
//     C[(i + 3) * N + (j + 1)] += c31;
//     C[(i + 3) * N + (j + 2)] += c32;
//     C[(i + 3) * N + (j + 3)] += c33;
// }

inline void compute_4x4_tile_packed(const uint8_t* A_packed, const uint8_t* B, double* C, 
                             int i, int j, int K, int N) {
    double c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
    double c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;
    double c20 = 0.0f, c21 = 0.0f, c22 = 0.0f, c23 = 0.0f;
    double c30 = 0.0f, c31 = 0.0f, c32 = 0.0f, c33 = 0.0f;

    for (int k = 0; k < K; ++k) {
        double a0 = A_packed[4 * k + 0];
        double a1 = A_packed[4 * k + 1];
        double a2 = A_packed[4 * k + 2];
        double a3 = A_packed[4 * k + 3];

        double b0 = B[k * N + (j + 0)];
        double b1 = B[k * N + (j + 1)];
        double b2 = B[k * N + (j + 2)];
        double b3 = B[k * N + (j + 3)];

        c00 += a0 * b0;  c01 += a0 * b1;  c02 += a0 * b2;  c03 += a0 * b3;
        c10 += a1 * b0;  c11 += a1 * b1;  c12 += a1 * b2;  c13 += a1 * b3;
        c20 += a2 * b0;  c21 += a2 * b1;  c22 += a2 * b2;  c23 += a2 * b3;
        c30 += a3 * b0;  c31 += a3 * b1;  c32 += a3 * b2;  c33 += a3 * b3;
    }

    C[(i + 0) * N + (j + 0)] += c00;
    C[(i + 0) * N + (j + 1)] += c01;
    C[(i + 0) * N + (j + 2)] += c02;
    C[(i + 0) * N + (j + 3)] += c03;

    C[(i + 1) * N + (j + 0)] += c10;
    C[(i + 1) * N + (j + 1)] += c11;
    C[(i + 1) * N + (j + 2)] += c12;
    C[(i + 1) * N + (j + 3)] += c13;

    C[(i + 2) * N + (j + 0)] += c20;
    C[(i + 2) * N + (j + 1)] += c21;
    C[(i + 2) * N + (j + 2)] += c22;
    C[(i + 2) * N + (j + 3)] += c23;

    C[(i + 3) * N + (j + 0)] += c30;
    C[(i + 3) * N + (j + 1)] += c31;
    C[(i + 3) * N + (j + 2)] += c32;
    C[(i + 3) * N + (j + 3)] += c33;
}


void naive_gemm_row_major_unrolled_ij(const uint8_t* A, 
                                      const uint8_t* B, 
                                      double* C, 
                                      int M, int K, int N) {

    int m_unroll = 4;
    int n_unroll = 4;

    int i = 0;
    for (; i <= M - m_unroll; i += m_unroll) {


        int j = 0;
        for (; j <= N - n_unroll; j += n_unroll) {
            // 4x4 tile accumulation
            compute_4x4_tile_packed(A, B, C, i, j, K, N);
        }

        // Handle tail columns
        for (; j < N; ++j) {
            for (int ii = 0; ii < m_unroll; ++ii) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    acc += A[(i + ii) * K + k] * B[k * N + j];
                }
                C[(i + ii) * N + j] += acc;
            }
        }
    }

    // Handle tail rows
    for (; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] += acc;
        }
    }
}


void naive_gemm_row_major_unrolled_ij_packed(const uint8_t* A, 
                                      const uint8_t* B, 
                                      double* C, 
                                      int M, int K, int N) {

    int m_unroll = 4;
    int n_unroll = 4;
    std::vector<uint8_t> A_packed(m_unroll * K);  // 4 x K block

    int i = 0;
    for (; i <= M - m_unroll; i += m_unroll) {

        pack_A_4xK(A, A_packed.data(), i, K, K);

        int j = 0;
        for (; j <= N - n_unroll; j += n_unroll) {
            // 4x4 tile accumulation
            compute_4x4_tile_packed(A_packed.data(), B, C, i, j, K, N);
        }

        // Handle tail columns
        for (; j < N; ++j) {
            for (int ii = 0; ii < m_unroll; ++ii) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    acc += A[(i + ii) * K + k] * B[k * N + j];
                }
                C[(i + ii) * N + j] += acc;
            }
        }
    }

    // Handle tail rows
    for (; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] += acc;
        }
    }
}


// Pack a tile of A of size block_M x block_K into row-major format
void pack_A_tile(const uint8_t* A, uint8_t* A_packed, int K,
                  int i0, int k0, int block_M, int block_K) {
    for (int i = 0; i < block_M; ++i) {
        for (int k = 0; k < block_K; ++k) {
            // [a00 a10 a20 a30...]
            A_packed[i * block_K + k] = *(A + (i0 + i) * K + (k0 + k));
        }
    }
}

// Pack a tile of B of size block_K x block_N into row-major format
void pack_B_tile(const uint8_t* B, uint8_t* B_packed, int N,
                 int k0, int j0, int block_K, int block_N) {
    for (int k = 0; k < block_K; ++k) {
        for (int j = 0; j < block_N; ++j) {
            B_packed[k * block_N + j] = *(B + (k0 + k) * N + (j0 + j));
        }
    }
}



void gemm_tiled_packed_unrolled(const uint8_t* A, 
                const uint8_t* B, 
                double* C, 
                int M, int K, int N) {
    const int block_size = 16;
    const int i_unroll = 4;
    const int j_unroll = 4;

    for (int ii = 0; ii < M; ii += block_size) {
        int block_M = std::min(block_size, M - ii);
        
        for (int jj = 0; jj < N; jj += block_size) {
            int block_N = std::min(block_size, N - jj);
            
            for (int kk = 0; kk < K; kk += block_size) {
                int block_K = std::min(block_size, K - kk);

                // Pack tiles
                uint8_t packed_A[block_M * block_K];  // Column-major
                uint8_t packed_B[block_K * block_N];  // Row-major
                
                // Pack this submatirx of A and B into packed_A and packed_B 
                // for sequential access instead of strided access
                pack_A_tile(A, packed_A, K, ii, kk, block_M, block_K);
                pack_B_tile(B, packed_B, N, kk, jj, block_K, block_N);

                // Compute with packed tiles
                for (int i = 0; i < block_M; i += i_unroll) {
                    for (int j = 0; j < block_N; j += j_unroll) {
                        double c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
                        double c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;
                        double c20 = 0.0f, c21 = 0.0f, c22 = 0.0f, c23 = 0.0f;
                        double c30 = 0.0f, c31 = 0.0f, c32 = 0.0f, c33 = 0.0f;

                        for (int k = 0; k < block_K; ++k) {
                            // Load 4 elements from packed_A (column-major)
                            double a0 = packed_A[k * block_M + (i + 0)];
                            double a1 = packed_A[k * block_M + (i + 1)];
                            double a2 = packed_A[k * block_M + (i + 2)];
                            double a3 = packed_A[k * block_M + (i + 3)];
                            
                            // Load 4 elements from packed_B (row-major)
                            double b0 = packed_B[k * block_N + (j + 0)];
                            double b1 = packed_B[k * block_N + (j + 1)];
                            double b2 = packed_B[k * block_N + (j + 2)];
                            double b3 = packed_B[k * block_N + (j + 3)];
                            
                            // Accumulate
                            c00 += a0 * b0;  c01 += a0 * b1;  c02 += a0 * b2;  c03 += a0 * b3;
                            c10 += a1 * b0;  c11 += a1 * b1;  c12 += a1 * b2;  c13 += a1 * b3;
                            c20 += a2 * b0;  c21 += a2 * b1;  c22 += a2 * b2;  c23 += a2 * b3;
                            c30 += a3 * b0;  c31 += a3 * b1;  c32 += a3 * b2;  c33 += a3 * b3;

                        } // k 
                        
                        int original_i = ii + i;
                        int original_j = jj + j;

                        *(C + (original_i + 0) * N + (original_j + 0)) += c00;
                        *(C + (original_i + 0) * N + (original_j + 1)) += c01;
                        *(C + (original_i + 0) * N + (original_j + 2)) += c02;
                        *(C + (original_i + 0) * N + (original_j + 3)) += c03;

                        *(C + (original_i + 1) * N + (original_j + 0)) += c10;
                        *(C + (original_i + 1) * N + (original_j + 1)) += c11;
                        *(C + (original_i + 1) * N + (original_j + 2)) += c12;
                        *(C + (original_i + 1) * N + (original_j + 3)) += c13;

                        *(C + (original_i + 2) * N + (original_j + 0)) += c20;
                        *(C + (original_i + 2) * N + (original_j + 1)) += c21;
                        *(C + (original_i + 2) * N + (original_j + 2)) += c22;
                        *(C + (original_i + 2) * N + (original_j + 3)) += c23;

                        *(C + (original_i + 3) * N + (original_j + 0)) += c30;
                        *(C + (original_i + 3) * N + (original_j + 1)) += c31;
                        *(C + (original_i + 3) * N + (original_j + 2)) += c32;
                        *(C + (original_i + 3) * N + (original_j + 3)) += c33;
                    } // j
                } // i 
            } // kk 
        } // jj 
    } // ii
}


void naive_gemm_row_major_tiled1(const uint8_t* A, 
                const uint8_t* B, 
                double* C, 
                int M, int K, int N) {
    int block_size = 16;
    int i_unroll = 4;
    int j_unroll = 4;
    std::vector<uint8_t> A_packed(block_size * block_size);  
    std::vector<uint8_t> B_packed(block_size * block_size);  

    for (int ii = 0; ii < M; ii += block_size) {
        for (int jj = 0; jj < N; jj += block_size) {
            for (int kk = 0; kk < K; kk += block_size) {


                // TODO: handle leftover too 
                for (int i = ii; i < std::min(M, ii + block_size); i += i_unroll){
                    // Instead of incrementing i one by one, now we increment i by 4 at a time
                    for (int j = jj; j < std::min(N, jj + block_size); j += j_unroll) {
                        double c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
                        double c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;
                        double c20 = 0.0f, c21 = 0.0f, c22 = 0.0f, c23 = 0.0f;
                        double c30 = 0.0f, c31 = 0.0f, c32 = 0.0f, c33 = 0.0f;

                        for (int k = kk; k < std::min(K, kk + block_size); k++){
                            // double a0 = *(A + (i + 0) * K + k);
                            // double a1 = *(A + (i + 1) * K + k);
                            // double a2 = *(A + (i + 2) * K + k);
                            // double a3 = *(A + (i + 3) * K + k);


                            double a0 = *(A + (i + 0) * K + k);
                            double a1 = *(A + (i + 1) * K + k);
                            double a2 = *(A + (i + 2) * K + k);
                            double a3 = *(A + (i + 3) * K + k);

                            double b0 = *(B + k * N + (j + 0));
                            double b1 = *(B + k * N + (j + 1));
                            double b2 = *(B + k * N + (j + 2));
                            double b3 = *(B + k * N + (j + 3));

                            c00 += a0 * b0;  c01 += a0 * b1;  c02 += a0 * b2;  c03 += a0 * b3;
                            c10 += a1 * b0;  c11 += a1 * b1;  c12 += a1 * b2;  c13 += a1 * b3;
                            c20 += a2 * b0;  c21 += a2 * b1;  c22 += a2 * b2;  c23 += a2 * b3;
                            c30 += a3 * b0;  c31 += a3 * b1;  c32 += a3 * b2;  c33 += a3 * b3;
                            
                        } // k
                        *(C + (i + 0) * N + (j + 0)) += c00;
                        *(C + (i + 0) * N + (j + 1)) += c01;
                        *(C + (i + 0) * N + (j + 2)) += c02;
                        *(C + (i + 0) * N + (j + 3)) += c03;

                        *(C + (i + 1) * N + (j + 0)) += c10;
                        *(C + (i + 1) * N + (j + 1)) += c11;
                        *(C + (i + 1) * N + (j + 2)) += c12;
                        *(C + (i + 1) * N + (j + 3)) += c13;

                        *(C + (i + 2) * N + (j + 0)) += c20;
                        *(C + (i + 2) * N + (j + 1)) += c21;
                        *(C + (i + 2) * N + (j + 2)) += c22;
                        *(C + (i + 2) * N + (j + 3)) += c23;

                        *(C + (i + 3) * N + (j + 0)) += c30;
                        *(C + (i + 3) * N + (j + 1)) += c31;
                        *(C + (i + 3) * N + (j + 2)) += c32;
                        *(C + (i + 3) * N + (j + 3)) += c33;
                    } // j 
                } // i
            } // jj 
        } // jj 
    } // ii
}



void naive_gemm_row_major_tiled2(const uint8_t* A, 
                const uint8_t* B, 
                float* C, 
                int M, int K, int N) {
    int block_size = 16;
    for (int ii = 0; ii < M; ii += block_size) 
        for (int kk = 0; kk < K; kk += block_size) 
            for (int jj = 0; jj < N; jj += block_size) 
                for (int i = ii; i < std::min(M, ii + block_size); i++)
                    for (int j = jj; j < std::min(N, jj + block_size); j++)
                        for (int k = kk; k < std::min(K, kk + block_size); k++)
                            *(C + i * N + j) += *(A + i * K + k) * *(B + k * N + j);
}

void naive_gemm_row_major_tiled3(const uint8_t* A, 
                const uint8_t* B, 
                float* C, 
                int M, int K, int N) {
    int block_size = 16;
    for (int ii = 0; ii < M; ii += block_size) 
        for (int kk = 0; kk < K; kk += block_size) 
            for (int jj = 0; jj < N; jj += block_size) 
                for (int i = ii; i < std::min(M, ii + block_size); i++)
                    for (int k = kk; k < std::min(K, kk + block_size); k++)
                        for (int j = jj; j < std::min(N, jj + block_size); j++)
                            *(C + i * N + j) += *(A + i * K + k) * *(B + k * N + j);
}


void naive_gemm_row_major_tiled4(const uint8_t* A, 
                const uint8_t* B, 
                float* C, 
                int M, int K, int N) {
    int block_size = 16;
    for (int ii = 0; ii < M; ii += block_size) 
        for (int jj = 0; jj < N; jj += block_size) 
            for (int kk = 0; kk < K; kk += block_size) 
                for (int i = ii; i < std::min(M, ii + block_size); i++)
                    for (int k = kk; k < std::min(K, kk + block_size); k++)
                        for (int j = jj; j < std::min(N, jj + block_size); j++)
                            *(C + i * N + j) += *(A + i * K + k) * *(B + k * N + j);
}



void naive_gemm_packed(const uint8_t* A, 
                const uint8_t* B, 
                float* C, 
                int M, int K, int N) {
    int block_size = 16;
    uint8_t packed_A_block[block_size * block_size];  

    for (int ii = 0; ii < M; ii += block_size) {

        for (int jj = 0; jj < N; jj += block_size) 
            for (int kk = 0; kk < K; kk += block_size) {
                    for (int i = ii; i < std::min(M, ii + block_size); i++)
                        for (int k = kk; k < std::min(K, kk + block_size); k++)
                            for (int j = jj; j < std::min(N, jj + block_size); j++)
                                *(C + i * N + j) += *(A + i * K + k) * *(B + k * N + j);
            }
    }
}


void neon_binary_gemm_tiled_with_params(const uint8_t* A_packed,
                                       const uint8_t* B_packed,
                                       int32_t* C,
                                       int M, int K, int N,
                                       int tile_m, int tile_n, int tile_k)
{   
    int packed_K = (K + 7) / 8;  // Number of bytes per row/column
    int tile_k_bytes = tile_k / 8;  // Convert tile_k to bytes
    
    // Validate tile_k is a multiple of 8
    if (tile_k % 8 != 0) {
        // Round up to nearest multiple of 8
        tile_k = ((tile_k + 7) / 8) * 8;
        tile_k_bytes = tile_k / 8;
    }

    const int padded_bits = packed_K * 8 - K;  // number of invalid bits at the very end (0..7)
    const int last_global_byte_index = packed_K - 1;
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i0 = 0; i0 < M; i0 += tile_m) {
        for (int j0 = 0; j0 < N; j0 += tile_n) {
            
            // Calculate actual tile sizes for this block
            int actual_tile_m = std::min(tile_m, M - i0);
            int actual_tile_n = std::min(tile_n, N - j0);
            
            // Use one contiguous accumulator buffer for better cache locality
            std::vector<int32_t> acc(actual_tile_m * actual_tile_n, 0);
            
            for (int k0 = 0; k0 < packed_K; k0 += tile_k_bytes) {
                const int actual_tile_k_bytes = std::min(tile_k_bytes, packed_K - k0);
                
                for (int ti = 0; ti < actual_tile_m; ++ti) {
                    const uint8_t* a_ptr = A_packed + (i0 + ti) * packed_K + k0;
                    
                    // Vectorized path over 16-byte chunks. We hoist a_vec outside the column loop.
                    int k_byte = 0;
                    for (; k_byte + 16 <= actual_tile_k_bytes; k_byte += 16) {
                        const uint8x16_t a_vec = vld1q_u8(a_ptr + k_byte);
                        const bool is_final_global_block = (k0 + k_byte + 16) >= packed_K;
                        
                        for (int tj = 0; tj < actual_tile_n; ++tj) {
                            const uint8_t* b_ptr = B_packed + (j0 + tj) * packed_K + k0;
                            // Light prefetch of upcoming B bytes to help L1/L2
                            __builtin_prefetch(b_ptr + k_byte + 64, 0, 1);
                            const uint8x16_t b_vec = vld1q_u8(b_ptr + k_byte);
                            
                            const uint8x16_t xnor_vec = vmvnq_u8(veorq_u8(a_vec, b_vec));
                            const uint8x16_t popcnt = vcntq_u8(xnor_vec);
                            
                            #if defined(__aarch64__)
                                acc[ti * actual_tile_n + tj] += (int32_t)vaddvq_u8(popcnt);
                            #else
                                uint16x8_t half_sum = vpaddlq_u8(popcnt);
                                uint32x4_t quarter_sum = vpaddlq_u16(half_sum);
                                uint64x2_t eighth_sum = vpaddlq_u32(quarter_sum);
                                acc[ti * actual_tile_n + tj] += (int32_t)(vgetq_lane_u64(eighth_sum, 0) + vgetq_lane_u64(eighth_sum, 1));
                            #endif
                            
                            // If this 16-byte block includes the very last byte overall, subtract padded bits once.
                            if (is_final_global_block && padded_bits > 0) {
                                acc[ti * actual_tile_n + tj] -= padded_bits;
                            }
                        }
                    }
                    
                    // Handle remaining bytes (less than 16) once after the vector loop
                    const int rem_bytes = actual_tile_k_bytes - k_byte;
                    if (rem_bytes > 0) {
                        for (int tj = 0; tj < actual_tile_n; ++tj) {
                            const uint8_t* b_ptr = B_packed + (j0 + tj) * packed_K + k0;
                            
                            // Fast path for all-but-possibly-last byte: add full popcounts
                            int local_sum = 0;
                            for (int b = 0; b < rem_bytes; ++b) {
                                const int global_byte_index = k0 + k_byte + b;
                                const uint8_t a_byte = a_ptr[k_byte + b];
                                const uint8_t b_byte = b_ptr[k_byte + b];
                                const uint8_t xnor_byte = (uint8_t)~(a_byte ^ b_byte);

                                if (global_byte_index == last_global_byte_index && padded_bits > 0) {
                                    // Only count valid bits in the last byte
                                    const int valid_bits = 8 - padded_bits; // in [1..7]
                                    const uint8_t mask = (uint8_t)(((1u << valid_bits) - 1u) << (8 - valid_bits));
                                    local_sum += __builtin_popcount((unsigned int)(xnor_byte & mask)) & 0xFF;
                                } else {
                                    // Full byte is valid
                                    local_sum += __builtin_popcount((unsigned int)xnor_byte) & 0xFF;
                                }
                            }
                            acc[ti * actual_tile_n + tj] += local_sum;
                        }
                    }
                }
            }

            // Write accumulated results to C
            for (int ti = 0; ti < actual_tile_m; ++ti) {
                for (int tj = 0; tj < actual_tile_n; ++tj) {
                    const int popcnt_sum = acc[ti * actual_tile_n + tj];
                    C[(i0 + ti) * N + (j0 + tj)] = 2 * popcnt_sum - K;
                }
            }
        }
    }
}