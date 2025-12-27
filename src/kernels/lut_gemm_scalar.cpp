#include "lutmac/lut_gemm.hpp"
#include "lutmac/thread_pool.hpp"
#include "lutmac/transform.hpp"
#include <cstdio>
namespace lutmac {

void lut_gemv_scalar(const BitPlaneBlock *blocks, size_t num_blocks,
                     const float *activations, float *output, size_t M,
                     size_t K) {
  // blocks points to the start of the weight matrix
  // The matrix has M rows and K columns.
  // Weights are packed into blocks of size 256.
  // With padding, we use div_ceil.

  const size_t blocks_per_row =
      (K + BitPlaneBlock::BLOCK_SIZE - 1) / BitPlaneBlock::BLOCK_SIZE;

  // Debug logging removed for clean output

  for (size_t row = 0; row < M; ++row) {
    float acc = 0.0f;

    const BitPlaneBlock *row_blocks = blocks + row * blocks_per_row;

    for (size_t b = 0; b < blocks_per_row; ++b) {
      const auto &block = row_blocks[b];
      float block_scale = block.scale;

      float block_sum = 0.0f;
      const float *block_input = activations + b * BitPlaneBlock::BLOCK_SIZE;

      // Process 256 weights
      // We can process 8 at a time (byte)
      for (size_t i = 0; i < BitPlaneBlock::BLOCK_SIZE / 8; ++i) {
        uint8_t signs = block.sign_plane[i];
        uint8_t zeros = block.zero_plane[i];

        // Bit j corresponds to weight (i*8 + j)
        // pack_to_bitplanes uses MSB-first: weight 0 is bit 7, weight 7 is bit
        // 0.
        for (int j = 0; j < 8; ++j) {
          int bit_idx = 7 - j;
          bool z = (zeros >> bit_idx) & 1;
          bool s = (signs >> bit_idx) & 1;

          if (!z) {
            float val = block_input[i * 8 + j];
            if (s)
              block_sum -= val;
            else
              block_sum += val;
          }
        }
      }

      acc += block_sum * block_scale;
    }

    output[row] = acc;
  }
}

// Forward declaration
void lut_gemv_binary_scalar(const BinaryBlock *blocks, size_t num_blocks,
                            const float *activations, float *output, size_t M,
                            size_t K);

void lut_linear(const PackedTensor &weight, const float *bias,
                const float *input, float *output, size_t batch_size) {
  if (weight.shape.size() != 2)
    return;
  size_t M = weight.shape[0];
  size_t K = weight.shape[1];

  // For each item in batch
  for (size_t b = 0; b < batch_size; ++b) {
    const float *curr_input = input + b * K;
    float *curr_output = output + b * M;

    // Zero-pad input if K is not a multiple of 256 (block size)
    // This is required because SIMD kernels (NEON/AVX) often read full blocks
    // and we want ensuring those extra values are 0 avoids garbage results.
    // Static buffer to minimize reallocation overhead for sequential
    // processing.
    static thread_local std::vector<float> padded_input_buffer;
    const float *effective_input = curr_input;

    size_t k_aligned = (K + 255) / 256 * 256;

    // Determine if we need to apply Hadamard Rotation (for bits <= 2)
    // This matches the quantizer logic.
    bool needs_rotation = (weight.quant_bits <= 2 && weight.quant_bits > 0) ||
                          (!weight.blocks.empty()); // Blocks implies ternary

    if (k_aligned > K || needs_rotation) {
      // Force copy if we need padding OR rotation (since rotation is in-place)
      if (padded_input_buffer.size() < k_aligned) {
        padded_input_buffer.resize(k_aligned);
      }
      // Copy valid data
      std::memcpy(padded_input_buffer.data(), curr_input, K * sizeof(float));
      // Zero pad
      if (k_aligned > K) {
        std::memset(padded_input_buffer.data() + K, 0,
                    (k_aligned - K) * sizeof(float));
      }

      if (needs_rotation) {
        lutmac::apply_hadamard_rotation(padded_input_buffer.data(), k_aligned);
      }

      effective_input = padded_input_buffer.data();
    }

    if (weight.is_quantized) {
      if (weight.quant_bits == 4 && !weight.int4_blocks.empty()) {
        // 4-bit quantization path
        int4_gemv(weight.int4_blocks.data(), weight.int4_blocks.size(),
                  effective_input, curr_output, M, K);
      } else if (weight.quant_bits == 8 && !weight.int8_blocks.empty()) {
        // Int8 path
        int8_gemv(weight.int8_blocks.data(), weight.int8_blocks.size(),
                  effective_input, curr_output, M, K);
      } else if (weight.quant_bits == 1 && !weight.binary_blocks.empty()) {
        // Binary (1-bit) path
#if defined(LUTMAC_HAS_NEON) || defined(LUTMAC_NEON)
        lut_gemv_binary_neon(weight.binary_blocks.data(),
                             weight.binary_blocks.size(), effective_input,
                             curr_output, M, K);
#else
        lut_gemv_binary_scalar(weight.binary_blocks.data(),
                               weight.binary_blocks.size(), effective_input,
                               curr_output, M, K);
#endif
      } else if (weight.quant_bits == 2 && !weight.int2_blocks.empty()) {
        // 2-bit path (RRQ)
        // effective_input is already rotated if logic above holds
        int2_gemv(weight.int2_blocks.data(), weight.int2_blocks.size(),
                  effective_input, curr_output, M, K);
      } else if (weight.quant_bits == 3 && !weight.int3_blocks.empty()) {
        // 3-bit path
        int3_gemv(weight.int3_blocks.data(), weight.int3_blocks.size(),
                  effective_input, curr_output, M, K);
      } else {
        // Ternary (1.58-bit) path
        lut_gemv(weight.blocks.data(), weight.blocks.size(), effective_input,
                 curr_output, M, K);
      }
    } else {
      // Raw FP32 fallback
      for (size_t i = 0; i < M; ++i) {
        if (i % 5000 == 0) {
          fprintf(stderr, "*");
          fflush(stderr);
        }
        float acc = 0.0f;
        const float *w_row = weight.raw_data.data() + i * K;
        for (size_t k = 0; k < K; ++k) {
          acc += w_row[k] * curr_input[k];
        }
        curr_output[i] = acc;
      }
    }

    // Add bias if present
    if (bias) {
      for (size_t i = 0; i < M; ++i) {
        curr_output[i] += bias[i];
      }
    }
  }
}

// ============================================================================
// Int4 GEMV Scalar Implementation
// ============================================================================

void int4_gemv_scalar(const Int4Block *blocks, size_t num_blocks,
                      const float *activations, float *output, size_t M,
                      size_t K) {
  const size_t blocks_per_row =
      (K + Int4Block::BLOCK_SIZE - 1) / Int4Block::BLOCK_SIZE;

  for (size_t row = 0; row < M; ++row) {
    float acc = 0.0f;

    const Int4Block *row_blocks = blocks + row * blocks_per_row;

    for (size_t b = 0; b < blocks_per_row; ++b) {
      const Int4Block &block = row_blocks[b];
      float block_scale = block.scale;

      float block_sum = 0.0f;
      const float *block_input = activations + b * Int4Block::BLOCK_SIZE;

      // Process 256 weights (128 bytes, 2 weights per byte)
      for (size_t i = 0; i < Int4Block::BLOCK_SIZE; ++i) {
        // Get quantized weight
        size_t byte_idx = i / 2;
        bool is_high = (i % 2) == 0;
        uint8_t nibble = is_high ? (block.data[byte_idx] >> 4)
                                 : (block.data[byte_idx] & 0x0F);
        int8_t signed_val = static_cast<int8_t>(nibble) - 8; // Centered at 0

        // w * x
        block_sum += signed_val * block_input[i];
      }

      acc += block_sum * block_scale;
    }

    output[row] = acc;
  }
}

// ============================================================================
// Binary (1-bit) GEMV Scalar Implementation
// ============================================================================

void lut_gemv_binary_scalar(const BinaryBlock *blocks, size_t num_blocks,
                            const float *activations, float *output, size_t M,
                            size_t K) {
  const size_t blocks_per_row =
      (K + BinaryBlock::BLOCK_SIZE - 1) / BinaryBlock::BLOCK_SIZE;

  // Process rows in parallel
  ThreadPool::instance().parallel_for(0, M, [&](size_t row) {
    float acc = 0.0f;
    const BinaryBlock *row_blocks = blocks + row * blocks_per_row;

    for (size_t b = 0; b < blocks_per_row; ++b) {
      const BinaryBlock &block = row_blocks[b];
      float block_scale = block.scale;
      float block_sum = 0.0f;
      const float *block_input = activations + b * BinaryBlock::BLOCK_SIZE;

      // Process 256 weights (32 bytes)
      for (size_t i = 0; i < BinaryBlock::BLOCK_SIZE / 8; ++i) {
        uint8_t byte = block.data[i];
        // Process 8 bits using popcount optimization
        for (int j = 0; j < 8; ++j) {
          bool bit = (byte >> (7 - j)) & 1;
          float val = block_input[i * 8 + j];
          block_sum += bit ? val : -val;
        }
      }
      acc += block_sum * block_scale;
    }
    output[row] = acc;
  });
}

// ============================================================================
// Int8 GEMV Scalar Implementation
// ============================================================================

void int8_gemv_scalar(const Int8Block *blocks, size_t num_blocks,
                      const float *activations, float *output, size_t M,
                      size_t K) {
  const size_t blocks_per_row =
      (K + Int8Block::BLOCK_SIZE - 1) / Int8Block::BLOCK_SIZE;

  for (size_t row = 0; row < M; ++row) {
    float acc = 0.0f;
    const Int8Block *row_blocks = blocks + row * blocks_per_row;

    for (size_t b = 0; b < blocks_per_row; ++b) {
      const Int8Block &block = row_blocks[b];
      float block_scale = block.scale;

      // Accumulate in float to avoid overflow, though int32 would be faster
      // usually
      float block_sum = 0.0f;
      const float *block_input = activations + b * Int8Block::BLOCK_SIZE;

      for (size_t i = 0; i < Int8Block::BLOCK_SIZE; ++i) {
        block_sum += static_cast<float>(block.data[i]) * block_input[i];
      }

      acc += block_sum * block_scale;
    }
    output[row] = acc;
  }
}

// ============================================================================
// Int2 GEMV Scalar Implementation
// ============================================================================

void int2_gemv_scalar(const Int2Block *blocks, size_t num_blocks,
                      const float *activations, float *output, size_t M,
                      size_t K) {
  const size_t blocks_per_row =
      (K + Int2Block::BLOCK_SIZE - 1) / Int2Block::BLOCK_SIZE;

  for (size_t row = 0; row < M; ++row) {
    float acc = 0.0f;
    const Int2Block *row_blocks = blocks + row * blocks_per_row;

    for (size_t b = 0; b < blocks_per_row; ++b) {
      const Int2Block &block = row_blocks[b];
      float block_scale = block.scale;
      float block_sum = 0.0f;
      const float *block_input = activations + b * Int2Block::BLOCK_SIZE;

      for (size_t i = 0; i < Int2Block::BLOCK_SIZE; ++i) {
        block_sum += block.get(i) * block_input[i] / block_scale;
      }
      acc += block_sum * block_scale;
    }
    output[row] = acc;
  }
}

// ============================================================================
// Int3 GEMV Scalar Implementation
// ============================================================================

void int3_gemv_scalar(const Int3Block *blocks, size_t num_blocks,
                      const float *activations, float *output, size_t M,
                      size_t K) {
  const size_t blocks_per_row =
      (K + Int3Block::BLOCK_SIZE - 1) / Int3Block::BLOCK_SIZE;

  for (size_t row = 0; row < M; ++row) {
    float acc = 0.0f;
    const Int3Block *row_blocks = blocks + row * blocks_per_row;

    for (size_t b = 0; b < blocks_per_row; ++b) {
      const Int3Block &block = row_blocks[b];
      float block_scale = block.scale;
      float block_sum = 0.0f;
      const float *block_input = activations + b * Int3Block::BLOCK_SIZE;

      for (size_t i = 0; i < Int3Block::BLOCK_SIZE; ++i) {
        block_sum += block.get(i) * block_input[i] / block_scale;
      }
      acc += block_sum * block_scale;
    }
    output[row] = acc;
  }
}

} // namespace lutmac
