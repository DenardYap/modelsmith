#include <torch/extension.h>
#include <vector>

/**
 * Packs a binary matrix (values must be 0 or 1, or -1/1 which get converted) into a uint8-packed tensor.
 * Padding is added if columns are not divisible by 8.
 *
 * @param mat: Tensor of shape (rows, cols), must be uint8 or int8, containing 0/1 or -1/1.
 * @return Packed tensor of shape (rows, ceil(cols / 8)), dtype = uint8.
 * 
 * TODO: Look into whether if we can use SIMD to speed this up, good enough for now
 */
torch::Tensor pack_binary_matrix(torch::Tensor mat) {
    TORCH_CHECK(mat.dim() == 2, "Input must be a 2D tensor");
    mat = mat.to(torch::kUInt8).contiguous(); // Ensure uint8 and contiguous

    int rows = mat.size(0);
    int cols = mat.size(1);
    int packed_cols = (cols + 7) / 8;

    auto packed = torch::zeros({rows, packed_cols}, torch::dtype(torch::kUInt8));
    // Pointer access is much faster than mat[row][col]
    auto mat_ptr = mat.data_ptr<uint8_t>();
    auto packed_ptr = packed.data_ptr<uint8_t>();

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int val = mat_ptr[row * cols + col];
            if (val == 1) {
                int byte_idx = col / 8;
                int bit_idx = col % 8;
                packed_ptr[row * packed_cols + byte_idx] |= (1 << (7 - bit_idx));
            }
        }
    }

    return packed;
}


/**
 * Pack binary matrix implemented using SIMD instructions.
 */

 // TODO: TEST
 // TODO: Maybe even pack 32 or 64 elements at a time depends on the matrix size 
 // TODO: For cols not in the multiple of 8, figure out how to handle it 
torch::Tensor pack_binary_matrix_SIMD64(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2, "Input must be a 2D tensor");

    int rows = input.size(0);
    int cols = input.size(1);
    int packed_cols = (cols + 7) / 8;

    torch::Tensor packed = torch::empty({rows, packed_cols}, torch::kUInt8);
    const int8_t* input_ptr = input.data_ptr<int8_t>();
    uint8_t* packed_ptr = packed.data_ptr<uint8_t>();

    for (int row = 0; row < rows; ++row) {
        const int8_t* row_ptr = input_ptr + row * cols;
        uint8_t* out_ptr = packed_ptr + row * packed_cols;

        int col = 0;
        int out_byte_idx = 0;

        // Process 8 columns at a time
        for (; col + 8 <= cols; col += 8) {
            int8x8_t vals = vld1_s8(row_ptr + col);             // Load 8 int8 values
            uint8x8_t gt_zero = vcgt_s8(vals, vdup_n_s8(0));    // Compare > 0 → 0xFF for 1, 0x00 for -1
            uint8x8_t bits = vand_u8(gt_zero, vdup_n_u8(1));    // Mask lower bit
            int8x8_t shift_mask = {7,6,5,4,3,2,1,0}; // same shape as bits
            uint8x8_t shifted = vshl_u8(bits, shift_mask);  // shift by mask

            uint8_t byte = vaddv_u8(shifted);   

            // uint8_t byte =
            //     (vget_lane_u8(bits, 0) << 7) | (vget_lane_u8(bits, 1) << 6) |
            //     (vget_lane_u8(bits, 2) << 5) | (vget_lane_u8(bits, 3) << 4) |
            //     (vget_lane_u8(bits, 4) << 3) | (vget_lane_u8(bits, 5) << 2) |
            //     (vget_lane_u8(bits, 6) << 1) | (vget_lane_u8(bits, 7) << 0);

            out_ptr[out_byte_idx++] = byte;
        }

        // Handle leftover columns
        if (col < cols) {
            uint8_t byte = 0;
            for (int b = 0; col < cols && b < 8; ++b, ++col) {
                int8_t v = row_ptr[col];
                uint8_t bit = (v > 0) ? 1 : 0;
                byte |= (bit << (7 - b));
            }
            out_ptr[out_byte_idx++] = byte;
        }
    }

    return packed;
}


/**
 * Pack binary matrix implemented using SIMD instructions.
 */
torch::Tensor pack_binary_matrix_SIMD128(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2, "Input must be a 2D tensor");

    int rows = input.size(0);
    int cols = input.size(1);
    int packed_cols = (cols + 7) / 8;

    torch::Tensor packed = torch::empty({rows, packed_cols}, torch::kUInt8);
    const int8_t* input_ptr = input.data_ptr<int8_t>();
    uint8_t* packed_ptr = packed.data_ptr<uint8_t>();

    for (int row = 0; row < rows; ++row) {
        const int8_t* row_ptr = input_ptr + row * cols;
        uint8_t* out_ptr = packed_ptr + row * packed_cols;

        int col = 0;
        int out_byte_idx = 0;

        // Process 16 columns at a time
        for (; col + 16 <= cols; col += 16) {
            int8x16_t vals = vld1q_s8(row_ptr + col);             // Load 16 int8 values
            uint8x16_t gt_zero = vcgtq_s8(vals, vdupq_n_s8(0));    
            uint8x16_t bits = vandq_u8(gt_zero, vdupq_n_u8(1));    
            int8x16_t shift_mask = {7,6,5,4,3,2,1,0, 7,6,5,4,3,2,1,0};

            uint8x16_t shifted = vshlq_u8(bits, shift_mask);  // shift by mask
             
            // Example: 
            // bits =      {1, 0, 1, 1, 0, 1, 0, 1 }
            // shift_mask = {7, 6, 5, 4, 3, 2, 1, 0 }
            // This should pack to 10110101 
            // shifted =   {128, 0, 32, 16, 0, 4, 0, 1, ... }
            // vaddv_u8 perform a horizontal add = 128 + 32 + 16 +4 + 1 = 181 = 10110101
            uint8x8_t low = vget_low_u8(shifted);
            uint8x8_t high = vget_high_u8(shifted);
            uint8_t byte0 = vaddv_u8(low);   
            uint8_t byte1 = vaddv_u8(high);  
                
            out_ptr[out_byte_idx++] = byte0;
            out_ptr[out_byte_idx++] = byte1;
        }

        // Handle leftover columns (up to 16 bits)
        while (col < cols) {
            uint8_t byte = 0;
            for (int b = 0; col < cols && b < 8; ++b, ++col) {
                int8_t v = row_ptr[col];
                uint8_t bit = (v > 0) ? 1 : 0;
                byte |= (bit << (7 - b));
            }
            out_ptr[out_byte_idx++] = byte;
        }
    }

    return packed;
}
