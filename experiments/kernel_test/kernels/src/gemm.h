#pragma once
#include <iostream>
#include <arm_neon.h>
#include <vector>
#include <stdint.h>
#include <random>
#include <chrono>

void neon_binary_gemm(const uint8_t*, const uint8_t*, uint32_t*,int, int, int);
void neon_gemm(const uint8_t*, const uint8_t*, float*, int, int, int);
void naive_gemm(const uint8_t*, const uint8_t*, float*, int, int, int);