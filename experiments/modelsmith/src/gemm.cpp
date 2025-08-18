#include <iostream>
#include <vector>
#include <stdint.h>
#include <random>
#include <chrono>
#include <cassert>
#include <algorithm>

// Architecture-specific SIMD headers
#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

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
            // AVX2 32-byte blocks (x86_64)
            #if defined(__AVX2__)
            {
                const __m256i all_ones_256 = _mm256_set1_epi8((char)0xFF);
                const __m128i lut128 = _mm_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
                const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
                for (; k_byte + 32 <= packed_K; k_byte += 32) {
                    const __m256i a_vec = _mm256_loadu_si256((const __m256i*)(A_packed + i * packed_K + k_byte));
                    const __m256i b_vec = _mm256_loadu_si256((const __m256i*)(B_packed + j * packed_K + k_byte));
                    const __m256i xnor_vec = _mm256_xor_si256(_mm256_xor_si256(a_vec, b_vec), all_ones_256);
                    const __m256i lo = _mm256_and_si256(xnor_vec, _mm256_set1_epi8(0x0F));
                    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(xnor_vec, 4), _mm256_set1_epi8(0x0F));
                    const __m256i cnt_lo = _mm256_shuffle_epi8(lut256, lo);
                    const __m256i cnt_hi = _mm256_shuffle_epi8(lut256, hi);
                    const __m256i popcnt_bytes = _mm256_add_epi8(cnt_lo, cnt_hi);
                    const __m256i sad = _mm256_sad_epu8(popcnt_bytes, _mm256_setzero_si256());
                    uint64_t s0 = (uint64_t)_mm256_extract_epi64(sad, 0);
                    uint64_t s1 = (uint64_t)_mm256_extract_epi64(sad, 1);
                    uint64_t s2 = (uint64_t)_mm256_extract_epi64(sad, 2);
                    uint64_t s3 = (uint64_t)_mm256_extract_epi64(sad, 3);
                    uint32_t count = (uint32_t)(s0 + s1 + s2 + s3);
                    sum += 2 * count - 8 * 32;
                    if (k_byte + 32 >= packed_K) {
                        int padded_bits = packed_K * 8 - K;
                        sum -= padded_bits;
                    }
                }
            }
            #endif

            // NEON 16-byte blocks (AArch64)
            #if defined(__aarch64__)
            for (; k_byte + 16 <= packed_K; k_byte += 16) {
                uint8x16_t a_vec = vld1q_u8(A_packed + i * packed_K + k_byte);
                uint8x16_t b_vec = vld1q_u8(B_packed + j * packed_K + k_byte);
                uint8x16_t xnor_vec = vmvnq_u8(veorq_u8(a_vec, b_vec));
                uint8x16_t popcnt = vcntq_u8(xnor_vec);
                uint32_t count = vaddvq_u8(popcnt);
                sum += 2 * count - 8 * 16;
                if (k_byte + 16 >= packed_K ) {
                    int padded_bits = packed_K * 8 - K;
                    sum -= padded_bits;
                }
            }
            #elif defined(__SSSE3__)
            // SSSE3 16-byte blocks (x86 fallback)
            if (k_byte + 16 <= packed_K) {
                const __m128i all_ones_128 = _mm_set1_epi8((char)0xFF);
                const __m128i lut = _mm_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
                for (; k_byte + 16 <= packed_K; k_byte += 16) {
                    const __m128i a_vec = _mm_loadu_si128((const __m128i*)(A_packed + i * packed_K + k_byte));
                    const __m128i b_vec = _mm_loadu_si128((const __m128i*)(B_packed + j * packed_K + k_byte));
                    const __m128i xnor_vec = _mm_xor_si128(_mm_xor_si128(a_vec, b_vec), all_ones_128);
                    const __m128i lo = _mm_and_si128(xnor_vec, _mm_set1_epi8(0x0F));
                    const __m128i hi = _mm_and_si128(_mm_srli_epi16(xnor_vec, 4), _mm_set1_epi8(0x0F));
                    const __m128i cnt_lo = _mm_shuffle_epi8(lut, lo);
                    const __m128i cnt_hi = _mm_shuffle_epi8(lut, hi);
                    const __m128i popcnt_bytes = _mm_add_epi8(cnt_lo, cnt_hi);
                    const __m128i sad = _mm_sad_epu8(popcnt_bytes, _mm_setzero_si128());
                    uint64_t s0 = (uint64_t)_mm_cvtsi128_si64(sad);
                    uint64_t s1 = (uint64_t)_mm_cvtsi128_si64(_mm_srli_si128(sad, 8));
                    uint32_t count = (uint32_t)(s0 + s1);
                    sum += 2 * count - 8 * 16;
                    if (k_byte + 16 >= packed_K ) {
                        int padded_bits = packed_K * 8 - K;
                        sum -= padded_bits;
                    }
                }
            }
            #endif
            
            
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
                    
                    // Vectorized path: prefer AVX2 (32B), then NEON/SSSE3 (16B)
                    int k_byte = 0;
                    
                    // AVX2 32-byte blocks
                    #if defined(__AVX2__)
                    {
                        const __m256i all_ones_256 = _mm256_set1_epi8((char)0xFF);
                        const __m128i lut128 = _mm_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
                        const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
                        for (; k_byte + 32 <= actual_tile_k_bytes; k_byte += 32) {
                            const bool is_final_global_block = (k0 + k_byte + 32) >= packed_K;
                            for (int tj = 0; tj < actual_tile_n; ++tj) {
                                const uint8_t* b_ptr = B_packed + (j0 + tj) * packed_K + k0;
                                __builtin_prefetch(b_ptr + k_byte + 64, 0, 1);
                                const __m256i a_vec = _mm256_loadu_si256((const __m256i*)(a_ptr + k_byte));
                                const __m256i b_vec = _mm256_loadu_si256((const __m256i*)(b_ptr + k_byte));
                                const __m256i xnor_vec = _mm256_xor_si256(_mm256_xor_si256(a_vec, b_vec), all_ones_256);
                                const __m256i lo = _mm256_and_si256(xnor_vec, _mm256_set1_epi8(0x0F));
                                const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(xnor_vec, 4), _mm256_set1_epi8(0x0F));
                                const __m256i cnt_lo = _mm256_shuffle_epi8(lut256, lo);
                                const __m256i cnt_hi = _mm256_shuffle_epi8(lut256, hi);
                                const __m256i popcnt_bytes = _mm256_add_epi8(cnt_lo, cnt_hi);
                                const __m256i sad = _mm256_sad_epu8(popcnt_bytes, _mm256_setzero_si256());
                                uint64_t s0 = (uint64_t)_mm256_extract_epi64(sad, 0);
                                uint64_t s1 = (uint64_t)_mm256_extract_epi64(sad, 1);
                                uint64_t s2 = (uint64_t)_mm256_extract_epi64(sad, 2);
                                uint64_t s3 = (uint64_t)_mm256_extract_epi64(sad, 3);
                                acc[ti * actual_tile_n + tj] += (int32_t)(s0 + s1 + s2 + s3);
                                if (is_final_global_block && padded_bits > 0) {
                                    acc[ti * actual_tile_n + tj] -= padded_bits;
                                }
                            }
                        }
                    }
                    #endif
                    
                    // NEON 16-byte blocks
                    #if defined(__aarch64__)
                    for (; k_byte + 16 <= actual_tile_k_bytes; k_byte += 16) {
                        const uint8x16_t a_vec = vld1q_u8(a_ptr + k_byte);
                        const bool is_final_global_block = (k0 + k_byte + 16) >= packed_K;
                        for (int tj = 0; tj < actual_tile_n; ++tj) {
                            const uint8_t* b_ptr = B_packed + (j0 + tj) * packed_K + k0;
                            __builtin_prefetch(b_ptr + k_byte + 64, 0, 1);
                            const uint8x16_t b_vec = vld1q_u8(b_ptr + k_byte);
                            const uint8x16_t xnor_vec = vmvnq_u8(veorq_u8(a_vec, b_vec));
                            const uint8x16_t popcnt = vcntq_u8(xnor_vec);
                            acc[ti * actual_tile_n + tj] += (int32_t)vaddvq_u8(popcnt);
                            if (is_final_global_block && padded_bits > 0) {
                                acc[ti * actual_tile_n + tj] -= padded_bits;
                            }
                        }
                    }
                    #elif defined(__SSSE3__)
                    // SSSE3 16-byte blocks
                    if (k_byte + 16 <= actual_tile_k_bytes) {
                        const __m128i all_ones_128 = _mm_set1_epi8((char)0xFF);
                        const __m128i lut = _mm_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
                        for (; k_byte + 16 <= actual_tile_k_bytes; k_byte += 16) {
                            const bool is_final_global_block = (k0 + k_byte + 16) >= packed_K;
                            for (int tj = 0; tj < actual_tile_n; ++tj) {
                                const uint8_t* b_ptr = B_packed + (j0 + tj) * packed_K + k0;
                                __builtin_prefetch(b_ptr + k_byte + 64, 0, 1);
                                const __m128i a_vec = _mm_loadu_si128((const __m128i*)(a_ptr + k_byte));
                                const __m128i b_vec = _mm_loadu_si128((const __m128i*)(b_ptr + k_byte));
                                const __m128i xnor_vec = _mm_xor_si128(_mm_xor_si128(a_vec, b_vec), all_ones_128);
                                const __m128i lo = _mm_and_si128(xnor_vec, _mm_set1_epi8(0x0F));
                                const __m128i hi = _mm_and_si128(_mm_srli_epi16(xnor_vec, 4), _mm_set1_epi8(0x0F));
                                const __m128i cnt_lo = _mm_shuffle_epi8(lut, lo);
                                const __m128i cnt_hi = _mm_shuffle_epi8(lut, hi);
                                const __m128i popcnt_bytes = _mm_add_epi8(cnt_lo, cnt_hi);
                                const __m128i sad = _mm_sad_epu8(popcnt_bytes, _mm_setzero_si128());
                                uint64_t s0 = (uint64_t)_mm_cvtsi128_si64(sad);
                                uint64_t s1 = (uint64_t)_mm_cvtsi128_si64(_mm_srli_si128(sad, 8));
                                acc[ti * actual_tile_n + tj] += (int32_t)(s0 + s1);
                                if (is_final_global_block && padded_bits > 0) {
                                    acc[ti * actual_tile_n + tj] -= padded_bits;
                                }
                            }
                        }
                    }
                    #endif
                    
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


void neon_binary_gemm_tiled_with_params_backup(const uint8_t* A_packed,
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
                    
                    // Vectorized path: prefer AVX2 (32B), then NEON/SSSE3 (16B)
                    int k_byte = 0;
                    
                    // AVX2 32-byte blocks
                    #if defined(__AVX2__)
                    {
                        const __m256i all_ones_256 = _mm256_set1_epi8((char)0xFF);
                        const __m128i lut128 = _mm_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
                        const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
                        for (; k_byte + 32 <= actual_tile_k_bytes; k_byte += 32) {
                            const bool is_final_global_block = (k0 + k_byte + 32) >= packed_K;
                            for (int tj = 0; tj < actual_tile_n; ++tj) {
                                const uint8_t* b_ptr = B_packed + (j0 + tj) * packed_K + k0;
                                __builtin_prefetch(b_ptr + k_byte + 64, 0, 1);
                                const __m256i a_vec = _mm256_loadu_si256((const __m256i*)(a_ptr + k_byte));
                                const __m256i b_vec = _mm256_loadu_si256((const __m256i*)(b_ptr + k_byte));
                                const __m256i xnor_vec = _mm256_xor_si256(_mm256_xor_si256(a_vec, b_vec), all_ones_256);
                                const __m256i lo = _mm256_and_si256(xnor_vec, _mm256_set1_epi8(0x0F));
                                const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(xnor_vec, 4), _mm256_set1_epi8(0x0F));
                                const __m256i cnt_lo = _mm256_shuffle_epi8(lut256, lo);
                                const __m256i cnt_hi = _mm256_shuffle_epi8(lut256, hi);
                                const __m256i popcnt_bytes = _mm256_add_epi8(cnt_lo, cnt_hi);
                                const __m256i sad = _mm256_sad_epu8(popcnt_bytes, _mm256_setzero_si256());
                                uint64_t s0 = (uint64_t)_mm256_extract_epi64(sad, 0);
                                uint64_t s1 = (uint64_t)_mm256_extract_epi64(sad, 1);
                                uint64_t s2 = (uint64_t)_mm256_extract_epi64(sad, 2);
                                uint64_t s3 = (uint64_t)_mm256_extract_epi64(sad, 3);
                                acc[ti * actual_tile_n + tj] += (int32_t)(s0 + s1 + s2 + s3);
                                if (is_final_global_block && padded_bits > 0) {
                                    acc[ti * actual_tile_n + tj] -= padded_bits;
                                }
                            }
                        }
                    }
                    #endif
                    
                    // NEON 16-byte blocks
                    #if defined(__aarch64__)
                    for (; k_byte + 16 <= actual_tile_k_bytes; k_byte += 16) {
                        const uint8x16_t a_vec = vld1q_u8(a_ptr + k_byte);
                        const bool is_final_global_block = (k0 + k_byte + 16) >= packed_K;
                        for (int tj = 0; tj < actual_tile_n; ++tj) {
                            const uint8_t* b_ptr = B_packed + (j0 + tj) * packed_K + k0;
                            __builtin_prefetch(b_ptr + k_byte + 64, 0, 1);
                            const uint8x16_t b_vec = vld1q_u8(b_ptr + k_byte);
                            const uint8x16_t xnor_vec = vmvnq_u8(veorq_u8(a_vec, b_vec));
                            const uint8x16_t popcnt = vcntq_u8(xnor_vec);
                            acc[ti * actual_tile_n + tj] += (int32_t)vaddvq_u8(popcnt);
                            if (is_final_global_block && padded_bits > 0) {
                                acc[ti * actual_tile_n + tj] -= padded_bits;
                            }
                        }
                    }
                    #elif defined(__SSSE3__)
                    // SSSE3 16-byte blocks
                    if (k_byte + 16 <= actual_tile_k_bytes) {
                        const __m128i all_ones_128 = _mm_set1_epi8((char)0xFF);
                        const __m128i lut = _mm_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
                        for (; k_byte + 16 <= actual_tile_k_bytes; k_byte += 16) {
                            const bool is_final_global_block = (k0 + k_byte + 16) >= packed_K;
                            for (int tj = 0; tj < actual_tile_n; ++tj) {
                                const uint8_t* b_ptr = B_packed + (j0 + tj) * packed_K + k0;
                                __builtin_prefetch(b_ptr + k_byte + 64, 0, 1);
                                const __m128i a_vec = _mm_loadu_si128((const __m128i*)(a_ptr + k_byte));
                                const __m128i b_vec = _mm_loadu_si128((const __m128i*)(b_ptr + k_byte));
                                const __m128i xnor_vec = _mm_xor_si128(_mm_xor_si128(a_vec, b_vec), all_ones_128);
                                const __m128i lo = _mm_and_si128(xnor_vec, _mm_set1_epi8(0x0F));
                                const __m128i hi = _mm_and_si128(_mm_srli_epi16(xnor_vec, 4), _mm_set1_epi8(0x0F));
                                const __m128i cnt_lo = _mm_shuffle_epi8(lut, lo);
                                const __m128i cnt_hi = _mm_shuffle_epi8(lut, hi);
                                const __m128i popcnt_bytes = _mm_add_epi8(cnt_lo, cnt_hi);
                                const __m128i sad = _mm_sad_epu8(popcnt_bytes, _mm_setzero_si128());
                                uint64_t s0 = (uint64_t)_mm_cvtsi128_si64(sad);
                                uint64_t s1 = (uint64_t)_mm_cvtsi128_si64(_mm_srli_si128(sad, 8));
                                acc[ti * actual_tile_n + tj] += (int32_t)(s0 + s1);
                                if (is_final_global_block && padded_bits > 0) {
                                    acc[ti * actual_tile_n + tj] -= padded_bits;
                                }
                            }
                        }
                    }
                    #endif
                    
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