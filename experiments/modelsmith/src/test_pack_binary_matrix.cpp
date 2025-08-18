#include <torch/torch.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

// Forward decl from kernel_helper.cpp
torch::Tensor pack_binary_matrix_SIMD128(torch::Tensor input);

static torch::Tensor make_int8_row(const std::vector<int>& vals) {
    std::vector<int8_t> data(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) {
        data[i] = static_cast<int8_t>(vals[i]);
    }
    auto t = torch::from_blob(data.data(), {(long)1, (long)vals.size()}, torch::TensorOptions().dtype(torch::kInt8)).clone();
    return t;
}

// Interpret the first K bits from packed (MSB-first within each byte) as a single integer
static uint64_t bits_to_uint_msb_first(const torch::Tensor& packed, int K) {
    const int packed_cols = (K + 7) / 8;
    assert(packed.dtype() == torch::kUInt8);
    assert(packed.size(0) == 1);
    assert(packed.size(1) == packed_cols);
    const uint8_t* p = packed.data_ptr<uint8_t>();
    uint64_t value = 0;
    for (int k = 0; k < K; ++k) {
        const int byte_index = k / 8;
        const int bit_in_byte = 7 - (k % 8);
        const bool bit = (p[byte_index] >> bit_in_byte) & 1u;
        value = (value << 1) | (bit ? 1u : 0u);
    }
    return value;
}

int main() {
    // Test A: 10 elements alternating {-1, 1, ...} -> bits 0101010101 (MSB-first) = 341
    {
        std::vector<int> v = {-1, 1, -1, 1, -1, 1, -1, 1, -1, 1};
        auto t = make_int8_row(v);
        auto packed = pack_binary_matrix_SIMD128(t);
        uint64_t val = bits_to_uint_msb_first(packed, (int)v.size());
        std::cout << "Test A value: " << val << " (expected 341)\n";
        assert(val == 341u);
    }

    // Test B: 8 elements {-1, 1, -1, 1, -1, 1, -1, 1} -> 01010101b = 85
    {
        std::vector<int> v = {-1, 1, -1, 1, -1, 1, -1, 1};
        auto t = make_int8_row(v);
        auto packed = pack_binary_matrix_SIMD128(t);
        const uint8_t byte = packed.data_ptr<uint8_t>()[0];
        std::cout << "Test B byte: " << (int)byte << " (expected 85)\n";
        assert(byte == 85);
    }

    // Test C: 10 elements {-1, -1, 1, -1, 1, 1, 1, -1, 1, -1}
    // bits: 0 0 1 0 1 1 1 0 1 0 (MSB-first) = 0b0010111010 = 186
    {
        std::vector<int> v = {-1, -1, 1, -1, 1, 1, 1, -1, 1, -1};
        auto t = make_int8_row(v);
        auto packed = pack_binary_matrix_SIMD128(t);
        uint64_t val = bits_to_uint_msb_first(packed, (int)v.size());
        std::cout << "Test C value: " << val << " (expected 186)\n";
        assert(val == 186u);
    }

    // Test D: 1 row of 32 alternating bits to exercise 2 full bytes output (should repeat 0x55)
    {
        std::vector<int> v;
        v.reserve(32);
        for (int i = 0; i < 32; ++i) v.push_back((i % 2 == 0) ? -1 : 1); // 0101...
        auto t = make_int8_row(v);
        auto packed = pack_binary_matrix_SIMD128(t);
        auto p = packed.data_ptr<uint8_t>();
        std::cout << "Test D bytes: " << (int)p[0] << ", " << (int)p[1] << ", " << (int)p[2] << ", " << (int)p[3] << " (expected all 85)\n";
        assert(p[0] == 85 && p[1] == 85 && p[2] == 85 && p[3] == 85);
    }

    // Test E: trailing bits not multiple of 8 (e.g., 17 bits): first two bytes encode first 16, third byte has 1 valid bit at MSB
    {
        std::vector<int> v(17, -1);
        // set first bit to +1
        v[0] = 1;
        auto t = make_int8_row(v);
        auto packed = pack_binary_matrix_SIMD128(t);
        auto p = packed.data_ptr<uint8_t>();
        // First bit 1 at MSB -> 1000 0000 = 128; remainder -1 -> zeros
        std::cout << "Test E first byte: " << (int)p[0] << " (expected 128)\n";
        assert(p[0] == 128);
    }

    std::cout << "All pack_binary_matrix_SIMD128 tests passed.\n";
    return 0;
}


