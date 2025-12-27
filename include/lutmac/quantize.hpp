#pragma once

/**
 * LutMac: Quantization Engine
 *
 * Converts FP16/FP32 weights to ternary bit-plane packed format.
 * Uses Sign-Value Decomposition (SVID) for stable quantization.
 */

#include "types.hpp"
#include <algorithm>
#include <cmath>

namespace lutmac {

// ============================================================================
// Quantization Statistics
// ============================================================================

struct QuantizationStats {
  float mean = 0.0f;
  float std_dev = 0.0f;
  float min_val = 0.0f;
  float max_val = 0.0f;
  float kurtosis = 0.0f;
  float sparsity = 0.0f; // Fraction of zeros

  void compute(const float *data, size_t n) {
    if (n == 0)
      return;

    // Mean
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
      sum += data[i];
    }
    mean = static_cast<float>(sum / n);

    // Variance and extremes
    double var_sum = 0.0;
    min_val = data[0];
    max_val = data[0];
    size_t zero_count = 0;

    for (size_t i = 0; i < n; ++i) {
      double diff = data[i] - mean;
      var_sum += diff * diff;
      min_val = std::min(min_val, data[i]);
      max_val = std::max(max_val, data[i]);
      if (std::abs(data[i]) < 1e-7f)
        zero_count++;
    }
    std_dev = static_cast<float>(std::sqrt(var_sum / n));
    sparsity = static_cast<float>(zero_count) / n;

    // Kurtosis (for calibration)
    if (std_dev > 1e-7f) {
      double kurt_sum = 0.0;
      for (size_t i = 0; i < n; ++i) {
        double z = (data[i] - mean) / std_dev;
        kurt_sum += z * z * z * z;
      }
      kurtosis = static_cast<float>(kurt_sum / n - 3.0); // Excess kurtosis
    }
  }
};

// ============================================================================
// Ternary Quantization
// ============================================================================

/**
 * Quantize a single weight to ternary {-1, 0, +1}
 *
 * Uses absmean quantization from BitNet b1.58:
 * threshold = mean(|weights|)
 *
 * @param value Input weight value
 * @param threshold Zero-detection threshold
 * @param scale Output scale factor
 * @return Ternary value
 */
inline TernaryValue quantize_ternary(float value, float threshold) {
  if (std::abs(value) < threshold) {
    return TernaryValue::ZERO;
  }
  return value > 0 ? TernaryValue::POS_ONE : TernaryValue::NEG_ONE;
}

/**
 * Compute optimal threshold using absmean method
 */
inline float compute_absmean_threshold(const float *data, size_t n) {
  double abs_sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    abs_sum += std::abs(data[i]);
  }
  return static_cast<float>(abs_sum / n);
}

/**
 * Compute scale factor for a group of weights
 * Scale = mean of absolute values of non-zero weights
 */
inline float compute_group_scale(const float *data, size_t n, float threshold) {
  double scale_sum = 0.0;
  size_t count = 0;

  for (size_t i = 0; i < n; ++i) {
    if (std::abs(data[i]) >= threshold) {
      scale_sum += std::abs(data[i]);
      count++;
    }
  }

  return count > 0 ? static_cast<float>(scale_sum / count) : 1.0f;
}

// ============================================================================
// Bit-Plane Packing
// ============================================================================

/**
 * Pack ternary values into bit-planes
 *
 * For each weight:
 * - sign_bit = (value < 0) ? 1 : 0
 * - zero_bit = (value == 0) ? 1 : 0
 *
 * @param ternary Input ternary values
 * @param n Number of values (must be multiple of 8)
 * @param sign_plane Output sign bits
 * @param zero_plane Output zero-flag bits
 */
inline void pack_to_bitplanes(const TernaryValue *ternary, size_t n,
                              uint8_t *sign_plane, uint8_t *zero_plane) {
  for (size_t i = 0; i < n; i += 8) {
    uint8_t sign_byte = 0;
    uint8_t zero_byte = 0;

    for (size_t j = 0; j < 8; ++j) {
      if (i + j < n) {
        int8_t val = static_cast<int8_t>(ternary[i + j]);
        if (val < 0)
          sign_byte |= (1 << (7 - j));
        if (val == 0)
          zero_byte |= (1 << (7 - j));
      }
    }

    sign_plane[i / 8] = sign_byte;
    zero_plane[i / 8] = zero_byte;
  }
}

/**
 */
inline void unpack_from_bitplanes(const uint8_t *sign_plane,
                                  const uint8_t *zero_plane, size_t n,
                                  TernaryValue *ternary) {
  for (size_t i = 0; i < n; i += 8) {
    uint8_t sign_byte = sign_plane[i / 8];
    uint8_t zero_byte = zero_plane[i / 8];

    for (size_t j = 0; j < 8; ++j) {
      if (i + j < n) {
        bool is_neg = (sign_byte >> (7 - j)) & 1;
        bool is_zero = (zero_byte >> (7 - j)) & 1;

        if (is_zero) {
          ternary[i + j] = TernaryValue::ZERO;
        } else if (is_neg) {
          ternary[i + j] = TernaryValue::NEG_ONE;
        } else {
          ternary[i + j] = TernaryValue::POS_ONE;
        }
      }
    }
  }
}

// ============================================================================
// Full Tensor Quantization
// ============================================================================

/**
 * Quantize a full weight tensor to LutMac format
 *
 * @param data Input FP32 weights
 * @param shape Tensor shape
 * @return Packed tensor in bit-plane format
 */
inline PackedTensor quantize_tensor(const float *data,
                                    const std::vector<size_t> &shape,
                                    const std::string &name = "") {
  PackedTensor result;
  result.name = name;
  result.shape = shape;

  size_t n = 1;
  for (auto s : shape)
    n *= s;

  // Compute global threshold using absmean
  float threshold = compute_absmean_threshold(data, n);
  result.global_scale = threshold;

  // Quantize to ternary
  std::vector<TernaryValue> ternary(n);
  for (size_t i = 0; i < n; ++i) {
    ternary[i] = quantize_ternary(data[i], 0.5f * threshold);
  }

  // Pack into blocks
  // Important: If tensor is 2D [M, K], we must pad K to BLOCK_SIZE to ensure
  // each row starts at a block boundary for efficient GEMM.
  // If tensor is >2D or 1D, we treat as flat for now, or assume last dim is K.
  // Usually weights are 2D [Out, In].

  size_t padded_n = n;
  size_t blocks_per_row = 0;

  if (shape.size() == 2) {
    size_t M = shape[0];
    size_t K = shape[1];
    size_t blocks_k = div_ceil(K, BitPlaneBlock::BLOCK_SIZE);
    blocks_per_row = blocks_k;
    padded_n = M * (blocks_k * BitPlaneBlock::BLOCK_SIZE);
  } else {
    // 1D or other: just flat pack
    size_t blocks_n = div_ceil(n, BitPlaneBlock::BLOCK_SIZE);
    padded_n = blocks_n * BitPlaneBlock::BLOCK_SIZE;
    blocks_per_row = blocks_n; // treat as one giant row
  }

  size_t num_blocks = padded_n / BitPlaneBlock::BLOCK_SIZE;
  result.blocks.resize(num_blocks);

  // If we padded, we need to handle data copying carefully.
  // If shape is 2D, we iterate rows.
  if (shape.size() == 2) {
    size_t M = shape[0];
    size_t K = shape[1];

    for (size_t r = 0; r < M; ++r) {
      // Quantize row `r`
      size_t row_offset = r * K;
      size_t block_offset = r * blocks_per_row;

      for (size_t b = 0; b < blocks_per_row; ++b) {
        BitPlaneBlock &block = result.blocks[block_offset + b];
        // Clear block
        std::memset(&block, 0, sizeof(BitPlaneBlock));

        // Input data for this block
        // May partially overlap with valid data, rest is padding (0)
        size_t valid_len = 0;
        size_t start_col = b * BitPlaneBlock::BLOCK_SIZE;
        if (start_col < K) {
          valid_len = std::min(BitPlaneBlock::BLOCK_SIZE, K - start_col);
        }

        // Temp ternary buffer for block
        std::vector<TernaryValue> block_ternary(BitPlaneBlock::BLOCK_SIZE,
                                                TernaryValue::ZERO);
        float scale_sum = 0.0;
        size_t count = 0;

        for (size_t i = 0; i < valid_len; ++i) {
          float val = data[row_offset + start_col + i];
          TernaryValue t = quantize_ternary(val, 0.5f * threshold);
          block_ternary[i] = t;

          // Stats for scale
          if (std::abs(val) >= threshold) {
            scale_sum += std::abs(val);
            count++;
          }
        }
        // Padding (len to BLOCK_SIZE) remains ZERO

        // Pack
        pack_to_bitplanes(block_ternary.data(), BitPlaneBlock::BLOCK_SIZE,
                          block.sign_plane, block.zero_plane);

        // Block scale
        block.scale = count > 0 ? static_cast<float>(scale_sum / count) : 1.0f;
      }
    }
  } else {
    // Fallback for flat tensors (embeddings, etc)
    // Existing logic is technically fine for Embedding since K=Hidden which is
    // usually aligned-ish, but Embedding unpacking logic handles misalignment.
    // However, to be consistent, let's keep flat logic for non-2D.

    for (size_t b = 0; b < num_blocks; ++b) {
      BitPlaneBlock &block = result.blocks[b];
      size_t block_start = b * BitPlaneBlock::BLOCK_SIZE;
      // Check bounds against real N
      size_t block_size = 0;
      if (block_start < n) {
        block_size = std::min(BitPlaneBlock::BLOCK_SIZE, n - block_start);
      }

      // Initialize planes to zero
      std::memset(&block, 0, sizeof(BitPlaneBlock));

      if (block_size == 0)
        continue; // Should not happen if size calcs correct

      // Quantize segment
      std::vector<TernaryValue> block_ternary(BitPlaneBlock::BLOCK_SIZE,
                                              TernaryValue::ZERO);
      double scale_sum = 0.0;
      size_t count = 0;

      for (size_t i = 0; i < block_size; ++i) {
        float val = data[block_start + i];
        block_ternary[i] = quantize_ternary(val, 0.5f * threshold);
        if (std::abs(val) >= 0.5f * threshold) {
          scale_sum += std::abs(val);
          count++;
        }
      }

      // Pack ternary values to bit-planes
      pack_to_bitplanes(block_ternary.data(), BitPlaneBlock::BLOCK_SIZE,
                        block.sign_plane, block.zero_plane);

      // Compute block scale (one scale for 256 weights)
      block.scale = count > 0 ? static_cast<float>(scale_sum / count) : 1.0f;
    }
  }

  return result;
}

/**
 * Dequantize tensor (for verification)
 */
inline std::vector<float> dequantize_tensor(const PackedTensor &packed) {
  size_t n = packed.num_elements();
  std::vector<float> result(n);

  size_t idx = 0;
  for (const auto &block : packed.blocks) {
    std::vector<TernaryValue> ternary(BitPlaneBlock::BLOCK_SIZE);
    unpack_from_bitplanes(block.sign_plane, block.zero_plane,
                          BitPlaneBlock::BLOCK_SIZE, ternary.data());

    float scale = block.scale;
    for (size_t i = 0; i < BitPlaneBlock::BLOCK_SIZE && idx < n; ++i) {
      result[idx++] = static_cast<int8_t>(ternary[i]) * scale;
    }
  }

  return result;
}

// ============================================================================
// Data-Free Calibration
// ============================================================================

/**
 * Adjust threshold based on weight distribution kurtosis
 *
 * High kurtosis (heavy tails) -> lower threshold (more non-zeros)
 * Low kurtosis (flat distribution) -> higher threshold (more zeros)
 */
inline float calibrate_threshold(float base_threshold, float kurtosis) {
  // Kurtosis adjustment factor
  // Normal distribution has kurtosis = 0 (excess)
  // Positive kurtosis = heavier tails
  float adjustment = 1.0f - 0.1f * std::tanh(kurtosis);
  return base_threshold * adjustment;
}

// ============================================================================
// 4-Bit Quantization
// ============================================================================

/**
 * Quantize tensor to 4-bit symmetric format
 *
 * Range: values mapped to [-8, +7] * scale
 */
inline PackedTensor quantize_tensor_int4(const float *data,
                                         const std::vector<size_t> &shape,
                                         const std::string &name = "") {
  PackedTensor result;
  result.name = name;
  result.shape = shape;
  result.is_quantized = true;
  result.quant_bits = 4;

  size_t n = 1;
  for (auto s : shape)
    n *= s;

  // Handle 2D tensors (weight matrices) with row padding
  size_t padded_n = n;
  size_t blocks_per_row = 0;

  if (shape.size() == 2) {
    size_t M = shape[0];
    size_t K = shape[1];
    size_t blocks_k = div_ceil(K, Int4Block::BLOCK_SIZE);
    blocks_per_row = blocks_k;
    padded_n = M * (blocks_k * Int4Block::BLOCK_SIZE);
  } else {
    size_t blocks_n = div_ceil(n, Int4Block::BLOCK_SIZE);
    padded_n = blocks_n * Int4Block::BLOCK_SIZE;
    blocks_per_row = blocks_n;
  }

  size_t num_blocks = padded_n / Int4Block::BLOCK_SIZE;
  result.int4_blocks.resize(num_blocks);

  if (shape.size() == 2) {
    size_t M = shape[0];
    size_t K = shape[1];

    for (size_t r = 0; r < M; ++r) {
      size_t row_offset = r * K;
      size_t block_offset = r * blocks_per_row;

      for (size_t b = 0; b < blocks_per_row; ++b) {
        Int4Block &block = result.int4_blocks[block_offset + b];
        std::memset(&block, 0, sizeof(Int4Block));

        size_t start_col = b * Int4Block::BLOCK_SIZE;
        size_t valid_len = 0;
        if (start_col < K) {
          valid_len = std::min(Int4Block::BLOCK_SIZE, K - start_col);
        }

        // Find absmax for this block to compute scale
        float absmax = 0.0f;
        for (size_t i = 0; i < valid_len; ++i) {
          absmax = std::max(absmax, std::abs(data[row_offset + start_col + i]));
        }

        // Scale: absmax / 7 for symmetric [-7, +7] range
        block.scale = absmax / 7.0f;
        float inv_scale = (block.scale > 1e-10f) ? (1.0f / block.scale) : 0.0f;

        // Quantize each weight
        for (size_t i = 0; i < Int4Block::BLOCK_SIZE; ++i) {
          float val = 0.0f;
          if (i < valid_len) {
            val = data[row_offset + start_col + i];
          }

          // Quantize to [-7, +7] range, then offset to [0, 14] (we use 8 as
          // zero)
          int q = static_cast<int>(std::round(val * inv_scale));
          q = std::max(-7, std::min(7, q)); // Clamp to [-7, 7]
          uint8_t uq =
              static_cast<uint8_t>(q + 8); // Offset to [1, 15], 8 = zero

          block.set(i, uq);
        }
      }
    }
  } else {
    // 1D or flat packing
    for (size_t b = 0; b < num_blocks; ++b) {
      Int4Block &block = result.int4_blocks[b];
      std::memset(&block, 0, sizeof(Int4Block));

      size_t start_idx = b * Int4Block::BLOCK_SIZE;
      size_t valid_len = std::min(Int4Block::BLOCK_SIZE, n - start_idx);
      if (start_idx >= n)
        valid_len = 0;

      float absmax = 0.0f;
      for (size_t i = 0; i < valid_len; ++i) {
        absmax = std::max(absmax, std::abs(data[start_idx + i]));
      }

      block.scale = absmax / 7.0f;
      float inv_scale = (block.scale > 1e-10f) ? (1.0f / block.scale) : 0.0f;

      for (size_t i = 0; i < Int4Block::BLOCK_SIZE; ++i) {
        float val = (i < valid_len) ? data[start_idx + i] : 0.0f;
        int q = static_cast<int>(std::round(val * inv_scale));
        q = std::max(-7, std::min(7, q));
        uint8_t uq = static_cast<uint8_t>(q + 8);
        block.set(i, uq);
      }
    }
  }

  return result;
}

/**
 * ============================================================================
 * Binary (1-Bit) Quantization
 * ============================================================================
 *
 * Quantize tensor to 1-bit binary format {-1, +1} * scale
 *
 * Algorithm:
 * 1. Compute scale = mean(|weights|) for each block
 * 2. Set bit = 1 if weight >= 0 (maps to +scale)
 *    Set bit = 0 if weight < 0  (maps to -scale)
 */
inline PackedTensor quantize_tensor_binary(const float *data,
                                           const std::vector<size_t> &shape,
                                           const std::string &name = "") {
  PackedTensor result;
  result.name = name;
  result.shape = shape;
  result.is_quantized = true;
  result.quant_bits = 1;

  size_t n = 1;
  for (auto s : shape)
    n *= s;

  // Calculate number of blocks
  size_t padded_n = n;
  size_t blocks_per_row = 0;

  if (shape.size() == 2) {
    size_t M = shape[0];
    size_t K = shape[1];
    size_t blocks_k = div_ceil(K, BinaryBlock::BLOCK_SIZE);
    blocks_per_row = blocks_k;
    padded_n = M * (blocks_k * BinaryBlock::BLOCK_SIZE);
  } else {
    size_t blocks_n = div_ceil(n, BinaryBlock::BLOCK_SIZE);
    padded_n = blocks_n * BinaryBlock::BLOCK_SIZE;
    blocks_per_row = blocks_n;
  }

  size_t num_blocks = padded_n / BinaryBlock::BLOCK_SIZE;
  result.binary_blocks.resize(num_blocks);

  if (shape.size() == 2) {
    size_t M = shape[0];
    size_t K = shape[1];

    for (size_t r = 0; r < M; ++r) {
      size_t row_offset = r * K;
      size_t block_offset = r * blocks_per_row;

      for (size_t b = 0; b < blocks_per_row; ++b) {
        BinaryBlock &block = result.binary_blocks[block_offset + b];
        std::memset(&block, 0, sizeof(BinaryBlock));

        size_t start_col = b * BinaryBlock::BLOCK_SIZE;
        size_t valid_len = 0;
        if (start_col < K) {
          valid_len = std::min(BinaryBlock::BLOCK_SIZE, K - start_col);
        }

        // 1. Compute scale (Mean of Absolute Values)
        double sum_abs = 0.0;
        for (size_t i = 0; i < valid_len; ++i) {
          sum_abs += std::abs(data[row_offset + start_col + i]);
        }
        // Avoid division by zero
        block.scale =
            (valid_len > 0) ? static_cast<float>(sum_abs / valid_len) : 0.0f;

        // 2. Quantize bits
        for (size_t i = 0; i < BinaryBlock::BLOCK_SIZE; ++i) {
          float val = 0.0f;
          if (i < valid_len) {
            val = data[row_offset + start_col + i];
          }
          // >= 0 -> +1 (bit 1), < 0 -> -1 (bit 0)
          bool bit = (val >= 0.0f);
          block.set(i, bit);
        }
      }
    }
  } else {
    // 1D / Flat packing
    for (size_t b = 0; b < num_blocks; ++b) {
      BinaryBlock &block = result.binary_blocks[b];
      std::memset(&block, 0, sizeof(BinaryBlock));

      size_t start_idx = b * BinaryBlock::BLOCK_SIZE;
      size_t valid_len = std::min(BinaryBlock::BLOCK_SIZE, n - start_idx);
      if (start_idx >= n)
        valid_len = 0;

      double sum_abs = 0.0;
      for (size_t i = 0; i < valid_len; ++i) {
        sum_abs += std::abs(data[start_idx + i]);
      }
      block.scale =
          (valid_len > 0) ? static_cast<float>(sum_abs / valid_len) : 0.0f;

      for (size_t i = 0; i < BinaryBlock::BLOCK_SIZE; ++i) {
        float val = (i < valid_len) ? data[start_idx + i] : 0.0f;
        bool bit = (val >= 0.0f);
        block.set(i, bit);
      }
    }
  }

  return result;
}

/**
 * ============================================================================
 * Int8 Quantization
 * ============================================================================
 *
 * Quantize tensor to 8-bit symmetric format
 * Range: [-127, 127] * scale
 */
inline PackedTensor quantize_tensor_int8(const float *data,
                                         const std::vector<size_t> &shape,
                                         const std::string &name = "") {
  PackedTensor result;
  result.name = name;
  result.shape = shape;
  result.is_quantized = true;
  result.quant_bits = 8;

  size_t n = 1;
  for (auto s : shape)
    n *= s;

  // Calculate number of blocks
  size_t padded_n = n;
  size_t blocks_per_row = 0;

  if (shape.size() == 2) {
    size_t M = shape[0];
    size_t K = shape[1];
    size_t blocks_k = div_ceil(K, Int8Block::BLOCK_SIZE);
    blocks_per_row = blocks_k;
    padded_n = M * (blocks_k * Int8Block::BLOCK_SIZE);
  } else {
    size_t blocks_n = div_ceil(n, Int8Block::BLOCK_SIZE);
    padded_n = blocks_n * Int8Block::BLOCK_SIZE;
    blocks_per_row = blocks_n;
  }

  size_t num_blocks = padded_n / Int8Block::BLOCK_SIZE;
  result.int8_blocks.resize(num_blocks);

  if (shape.size() == 2) {
    size_t M = shape[0];
    size_t K = shape[1];

    for (size_t r = 0; r < M; ++r) {
      size_t row_offset = r * K;
      size_t block_offset = r * blocks_per_row;

      for (size_t b = 0; b < blocks_per_row; ++b) {
        Int8Block &block = result.int8_blocks[block_offset + b];
        std::memset(&block, 0, sizeof(Int8Block));

        size_t start_col = b * Int8Block::BLOCK_SIZE;
        size_t valid_len = 0;
        if (start_col < K) {
          valid_len = std::min(Int8Block::BLOCK_SIZE, K - start_col);
        }

        // Use helper to set (computes scale and quantizes)
        std::vector<float> buffer(Int8Block::BLOCK_SIZE, 0.0f);
        for (size_t i = 0; i < valid_len; ++i) {
          buffer[i] = data[row_offset + start_col + i];
        }
        block.set(buffer.data());
      }
    }
  } else {
    // 1D / Flat
    for (size_t b = 0; b < num_blocks; ++b) {
      Int8Block &block = result.int8_blocks[b];
      std::memset(&block, 0, sizeof(Int8Block));

      size_t start_idx = b * Int8Block::BLOCK_SIZE;
      size_t valid_len = std::min(Int8Block::BLOCK_SIZE, n - start_idx);
      if (start_idx >= n)
        valid_len = 0;

      std::vector<float> buffer(Int8Block::BLOCK_SIZE, 0.0f);
      for (size_t i = 0; i < valid_len; ++i) {
        buffer[i] = data[start_idx + i];
      }
      block.set(buffer.data());
    }
  }

  return result;
}

/**
 * ============================================================================
 * 2-Bit Quantization
 * ============================================================================
 */
inline PackedTensor quantize_tensor_int2(const float *data,
                                         const std::vector<size_t> &shape,
                                         const std::string &name = "") {
  PackedTensor result;
  result.name = name;
  result.shape = shape;
  result.is_quantized = true;
  result.quant_bits = 2;

  size_t n = 1;
  for (auto s : shape)
    n *= s;

  size_t padded_n = n;
  size_t blocks_per_row = 0;

  if (shape.size() == 2) {
    size_t M = shape[0];
    size_t K = shape[1];
    size_t blocks_k = div_ceil(K, Int2Block::BLOCK_SIZE);
    blocks_per_row = blocks_k;
    padded_n = M * (blocks_k * Int2Block::BLOCK_SIZE);
  } else {
    size_t blocks_n = div_ceil(n, Int2Block::BLOCK_SIZE);
    padded_n = blocks_n * Int2Block::BLOCK_SIZE;
    blocks_per_row = blocks_n;
  }

  size_t num_blocks = padded_n / Int2Block::BLOCK_SIZE;
  result.int2_blocks.resize(num_blocks);

  for (size_t b = 0; b < num_blocks; ++b) {
    Int2Block &block = result.int2_blocks[b];
    std::memset(&block, 0, sizeof(Int2Block));

    size_t start_idx = b * Int2Block::BLOCK_SIZE;
    size_t valid_len =
        (start_idx < n) ? std::min(Int2Block::BLOCK_SIZE, n - start_idx) : 0;

    float absmax = 0.0f;
    for (size_t i = 0; i < valid_len; ++i) {
      absmax = std::max(absmax, std::abs(data[start_idx + i]));
    }

    block.scale = absmax / 2.0f;
    float inv_scale = (block.scale > 1e-10f) ? (1.0f / block.scale) : 0.0f;

    for (size_t i = 0; i < Int2Block::BLOCK_SIZE; ++i) {
      float val = (i < valid_len) ? data[start_idx + i] : 0.0f;
      int q = static_cast<int>(std::round(val * inv_scale));
      q = std::max(-2, std::min(1, q));
      uint8_t uq = static_cast<uint8_t>(q + 2);
      block.set(i, uq);
    }
  }

  return result;
}

/**
 * ============================================================================
 * 3-Bit Quantization
 * ============================================================================
 */
inline PackedTensor quantize_tensor_int3(const float *data,
                                         const std::vector<size_t> &shape,
                                         const std::string &name = "") {
  PackedTensor result;
  result.name = name;
  result.shape = shape;
  result.is_quantized = true;
  result.quant_bits = 3;

  size_t n = 1;
  for (auto s : shape)
    n *= s;

  size_t padded_n = n;
  size_t blocks_per_row = 0;

  if (shape.size() == 2) {
    size_t M = shape[0];
    size_t K = shape[1];
    size_t blocks_k = div_ceil(K, Int3Block::BLOCK_SIZE);
    blocks_per_row = blocks_k;
    padded_n = M * (blocks_k * Int3Block::BLOCK_SIZE);
  } else {
    size_t blocks_n = div_ceil(n, Int3Block::BLOCK_SIZE);
    padded_n = blocks_n * Int3Block::BLOCK_SIZE;
    blocks_per_row = blocks_n;
  }

  size_t num_blocks = padded_n / Int3Block::BLOCK_SIZE;
  result.int3_blocks.resize(num_blocks);

  for (size_t b = 0; b < num_blocks; ++b) {
    Int3Block &block = result.int3_blocks[b];
    std::memset(&block, 0, sizeof(Int3Block));

    size_t start_idx = b * Int3Block::BLOCK_SIZE;
    size_t valid_len =
        (start_idx < n) ? std::min(Int3Block::BLOCK_SIZE, n - start_idx) : 0;

    float absmax = 0.0f;
    for (size_t i = 0; i < valid_len; ++i) {
      absmax = std::max(absmax, std::abs(data[start_idx + i]));
    }

    block.scale = absmax / 4.0f;
    float inv_scale = (block.scale > 1e-10f) ? (1.0f / block.scale) : 0.0f;

    for (size_t i = 0; i < Int3Block::BLOCK_SIZE; ++i) {
      float val = (i < valid_len) ? data[start_idx + i] : 0.0f;
      int q = static_cast<int>(std::round(val * inv_scale));
      q = std::max(-4, std::min(3, q));
      uint8_t uq = static_cast<uint8_t>(q + 4);
      block.set(i, uq);
    }
  }

  return result;
}

} // namespace lutmac
