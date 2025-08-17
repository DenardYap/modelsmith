
#include <iostream>
#include "../src/kernel_helper.h"
#include <torch/torch.h>
#include <iostream>
#include <cassert>
#include <bitset>

// Forward declaration
torch::Tensor pack_binary_matrix_SIMD128_float_input(torch::Tensor mat);

void test_small_2x8() {

    torch::Tensor mat = torch::tensor({{1,-1,1,-1,1,-1,1,-1},
                         {-1,1,-1,1,-1,1,-1,1}}, torch::kFloat32);

    torch::Tensor packed = pack_binary_matrix_SIMD128_float_input(mat);
    assert(packed.size(0) == 2);  
    assert(packed.size(1) == 1);  
    // Row 0: 10101010 -> 0b10101010 == 0xAA
    std::cout << std::bitset<8>(packed[0][0].item<int>()) << std::endl;
    std::cout << std::bitset<8>(packed[1][0].item<int>()) << std::endl;
    assert(std::bitset<8>(packed[0][0].item<int>()) ==  0b10101010);
    // // Row 1: 01010101 -> 0b01010101 == 0x55
    assert(std::bitset<8>(packed[1][0].item<int>()) == 0b01010101);
    std::cout << "test_small_2x8 passed!\n";
}


void test_small_3x9() {

    torch::Tensor mat = torch::tensor(
                        {{1,-1,1,-1,1,-1,1,-1,1},
                         {-1,1,-1,1,-1,1,-1,1,-1},
                         {1,1,1,1,-1,-1,-1,-1,-1}}, torch::kFloat32);

    torch::Tensor packed = pack_binary_matrix_SIMD128_float_input(mat);
    assert(packed.size(0) == 3);  
    assert(packed.size(1) == 2);  
    assert(std::bitset<8>(packed[0][0].item<uint8_t>()) ==  0b10101010);
    assert(std::bitset<8>(packed[0][1].item<uint8_t>()) ==  0b10000000);
    assert(std::bitset<8>(packed[1][0].item<uint8_t>()) == 0b01010101);
    assert(std::bitset<8>(packed[1][1].item<uint8_t>()) == 0b00000000);
    assert(std::bitset<8>(packed[2][0].item<uint8_t>()) == 0b11110000);
    assert(std::bitset<8>(packed[2][1].item<uint8_t>()) == 0b00000000);
    std::cout << "test_small_3x9 passed!\n";
}

void test_single_row() {
    // 1 row, 8 columns (exactly 1 byte)
    torch::Tensor mat = torch::tensor({{1, -1, -1, 1, 1, -1, 1, -1}}, torch::kFloat32);
    torch::Tensor packed = pack_binary_matrix_SIMD128_float_input(mat);
    assert(packed.size(0) == 1);
    assert(packed.size(1) == 1);
    // 1 -1 -1 1 1 -1 1 -1 -> 1 0 0 1 1 0 1 0 -> 0b10011010
    assert(std::bitset<8>(packed[0][0].item<uint8_t>()) == 0b10011010);
    std::cout << "test_single_row passed!\n";
}

void test_single_column() {
    // 8 rows, 1 column (should pack as 8 bytes, each with 1 bit set)
    torch::Tensor mat = torch::tensor({{1}, {-1}, {1}, {-1}, {1}, {-1}, {1}, {-1}}, torch::kFloat32);
    torch::Tensor packed = pack_binary_matrix_SIMD128_float_input(mat);
    assert(packed.size(0) == 8);
    assert(packed.size(1) == 1);
    for (int i = 0; i < 8; ++i) {
        uint8_t expected = (i % 2 == 0) ? 0b10000000 : 0b00000000;
        assert(std::bitset<8>(packed[i][0].item<uint8_t>()) == expected);
    }
    std::cout << "test_single_column passed!\n";
}

void test_768() {
    // Testing a 768x1 vector 
    torch::Tensor mat = torch::ones({768, 1}, torch::kFloat32);

    for (int i = 0; i < 768; ++i){
        if (i % 2 == 0){
            mat[i][0] = 1;
        } else {
            mat[i][0] = -1;
        }
    }

    torch::Tensor packed = pack_binary_matrix_SIMD128_float_input(mat);
    assert(packed.size(0) == 768);
    assert(packed.size(1) == 1);
    for (int i = 0; i < 768; ++i) {
        uint8_t expected = (i % 2 == 0) ? 0b10000000 : 0b00000000;
        assert(std::bitset<8>(packed[i][0].item<uint8_t>()) == expected);
    }

    // Testing a 1x768 vector 
    mat = torch::ones({1, 768}, torch::kFloat32);

    for (int i = 0; i < 768; ++i){
        if (i % 2 == 0){
            mat[0][i] = 1;
        } else {
            mat[0][i] = -1;
        }
    }

    packed = pack_binary_matrix_SIMD128_float_input(mat);
    assert(packed.size(0) == 1);
    assert(packed.size(1) == 96);
    for (int i = 0; i < 96; ++i) {
        uint8_t expected = 0b10101010;
        assert(std::bitset<8>(packed[0][i].item<uint8_t>()) == expected);
    }

    // Testing a 1x769 vector 
    mat = torch::ones({1, 769}, torch::kFloat32);

    for (int i = 0; i < 769; ++i){
        if (i % 2 == 0){
            mat[0][i] = 1;
        } else {
            mat[0][i] = -1;
        }
    }

    packed = pack_binary_matrix_SIMD128_float_input(mat);
    assert(packed.size(0) == 1);
    assert(packed.size(1) == 97);
    for (int i = 0; i < 97; ++i) {
        uint8_t expected = 0b10101010;
        if (i == 96){
            expected = 0b10000000;
        }
        assert(std::bitset<8>(packed[0][i].item<uint8_t>()) == expected);
    }

    std::cout << "test_768 passed!\n";
}

void test_all_minus_one() {
    // All -1s, should pack to all 0s
    torch::Tensor mat = torch::full({4, 16}, -1, torch::kFloat32);
    torch::Tensor packed = pack_binary_matrix_SIMD128_float_input(mat);
    assert(torch::all(packed == 0).item<bool>());
    std::cout << "test_all_minus_one passed!\n";
}

void test_all_one() {
    // All 1s, should pack to all 0xFF
    torch::Tensor mat = torch::ones({4, 16}, torch::kFloat32);
    torch::Tensor packed = pack_binary_matrix_SIMD128_float_input(mat);
    assert(torch::all(packed == 0xFF).item<bool>());
    std::cout << "test_all_one passed!\n";
}

void test_large_matrix() {
    // Stress test: 1000 x 1000, alternating pattern
    int rows = 1000, cols = 1002;
    torch::Tensor mat = torch::zeros({rows, cols}, torch::kFloat32);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat[i][j] = ((i + j) % 2 == 0) ? 1 : -1;
    torch::Tensor packed = pack_binary_matrix_SIMD128_float_input(mat);
    // Just check shape and that code runs
    assert(packed.size(0) == rows);
    assert(packed.size(1) == (cols + 7) / 8);

    for (int row = 0; row < rows; row++){
        for (int col = 0; col < (cols + 7) / 8; col++){
            // 10101010 
            if (col == 125){
                if (row % 2 == 0){
                    assert(std::bitset<8>(packed[row][col].item<uint8_t>()) == 0b10000000);
                } else {
                    assert(std::bitset<8>(packed[row][col].item<uint8_t>()) == 0b01000000);
                }
                    
            } else {
                if (row % 2 == 0){
                    assert(std::bitset<8>(packed[row][col].item<uint8_t>()) == 0b10101010);
                } else {
                    assert(std::bitset<8>(packed[row][col].item<uint8_t>()) == 0b01010101);
                }
            }
        }
    }
    std::cout << "test_large_matrix passed!\n";

}

void test_6x21_matrix() {
    // 6x21 matrix, alternating pattern per row
    torch::Tensor mat = torch::zeros({6, 21}, torch::kFloat32);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 21; ++j)
            mat[i][j] = ((i + j) % 2 == 0) ? 1 : -1;
    torch::Tensor packed = pack_binary_matrix_SIMD128_float_input(mat);
    assert(packed.size(0) == 6);
    assert(packed.size(1) == 3); // ceil(21/8) = 3

    // Check packed bytes for first row (should be 10101010 10101010 10100000)
    assert(std::bitset<8>(packed[0][0].item<uint8_t>()) == 0b10101010);
    assert(std::bitset<8>(packed[0][1].item<uint8_t>()) == 0b10101010);
    assert((packed[0][2].item<uint8_t>() & 0b11100000) == 0b10100000); // Only first 5 bits used

    std::cout << "test_6x21_matrix passed!\n";
}

void test_33x10_matrix() {
    // 33x10 matrix, alternating pattern per row
    torch::Tensor mat = torch::zeros({33, 10}, torch::kFloat32);
    for (int i = 0; i < 33; ++i)
        for (int j = 0; j < 10; ++j)
            mat[i][j] = ((i + j) % 2 == 0) ? 1 : -1;
    torch::Tensor packed = pack_binary_matrix_SIMD128_float_input(mat);
    assert(packed.size(0) == 33);
    assert(packed.size(1) == 2); // ceil(10/8) = 2

    // Check packed bytes for first row (should be 10101010 10xxxxxx)
    assert(std::bitset<8>(packed[0][0].item<uint8_t>()) == 0b10101010);
    assert((packed[0][1].item<uint8_t>() & 0b11000000) == 0b10000000); // Only first 2 bits used
    std::cout << "test_33x10_matrix passed!\n";
}

void test_non_multiple_of_8() {
    // 3x13 matrix (13 not divisible by 8)
    torch::Tensor mat = torch::tensor({
        {1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1},
        {-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1},
        {1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1}
    }, torch::kFloat32);
    torch::Tensor packed = pack_binary_matrix_SIMD128_float_input(mat);
    assert(packed.size(0) == 3);
    assert(packed.size(1) == 2); // ceil(13/8) = 2
    // Check first row: 1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1 -> 10101010 10101010 (last 5 bits unused)
    assert(std::bitset<8>(packed[0][0].item<uint8_t>()) == 0b10101010);
    assert((packed[0][1].item<uint8_t>() & 0b11111000) == 0b10101000); // Only first 5 bits used
    std::cout << "test_non_multiple_of_8 passed!\n";
}

void test_empty(){
    // Empty matrix
    torch::Tensor mat = torch::empty({0, 0}, torch::kFloat32);
    torch::Tensor packed = pack_binary_matrix_SIMD128_float_input(mat);
    assert(packed.size(0) == 0);
    assert(packed.size(1) == 0);
    std::cout << "test_empty passed!\n";
}

/**
 * Normal 2x8 X
 * Normal 2x9 or something X
 * Test all ones X
 * Test all negative ones X
 * Test very large matrix X
 * Test empty matrix X
 * Test 1d matrices row X
 * Test 1d matrices column X
 * Test some other arbitrary sizes like 6x21 X
 * Test initialize a matrix with 1s but every 8 starting/ending bit is -1 
 * Test initialize a matrix -1s 1s but every 8 starting/ending bit is 1 
 */
int main() {
    test_small_2x8();
    test_small_3x9();
    test_single_row();
    test_single_column();
    test_all_minus_one();
    test_all_one();
    test_large_matrix();
    test_6x21_matrix();
    test_33x10_matrix();
    test_non_multiple_of_8();
    test_empty();
    test_768();
    return 0;
}