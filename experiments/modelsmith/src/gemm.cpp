#include <iostream>
#include <arm_neon.h>
#include <vector>
#include <stdint.h>
#include <random>
#include <chrono>
#include <arm_neon.h>
#include <cassert>

/**
 * 
 * Args:
 * 
 * A_packed : 2d matrix of size MxK
 * B_packed : 2d matrix of size NxK (transposed)
 * 
 * EXAMPLE:
 \\ a =  -1 1 -1 1 -1 1 -1   1
 \\ b =   1 -1 -1 1 -1 -1 -1 1
 \\ dot(a, b) = -1 -1 +1 +1 +1 -1 +1 +1 = 2

 \\ a =  0 1 0 1 0 1 0 1
 \\ b =  1 0 0 1 0 0 0 1
 \\ XNOR(a, b) = 0 0 1 1 1 0 1 1
 \\ POPCOUNT(XNOR(a, b)) = 2 * 5 - 8 = 2 

 * TODO: make B_packed less confusing
 * NOTE: This function will only be faster than normal gemm if 
 * K > 128, the larger the K, the faster it is compared to normal gemm
*/
void neon_binary_gemm(const uint8_t* A_packed,
                      const uint8_t* B_packed,
                      int32_t* C,
                      int M, int K, int N)
{   
    int packed_K = (K + 7) / 8;  // Number of bytes per row/column

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
            }
            
            
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
            }

            // *(C + i * N + j) = sum;
            *(C + i * N + j) = sum;
            
        }
    }
}


void naive_gemm(const uint8_t* A, 
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