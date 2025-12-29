#pragma once

/**
 * LutMac: LUT-Based GEMM Interface
 *
 * Precompute-and-lookup matrix multiplication for ternary weights.
 * No floating-point multiplications - only additions and table lookups.
 */

#include "types.hpp"
#include <cstring>

namespace lutmac {

// ============================================================================
// LUT Precomputation
// ============================================================================

/**
 * Precompute LUT for 4 activations
 *
 * For activations [a0, a1, a2, a3]:
 * LUT[0b0000] = 0
 * LUT[0b0001] = a3
 * LUT[0b0010] = a2
 * LUT[0b0011] = a2 + a3
 * ...
 * LUT[0b1111] = a0 + a1 + a2 + a3
 */
inline void precompute_lut(const float *activations, float *lut) {
  lut[0] = 0.0f;
  for (int i = 1; i < 16; ++i) {
    float sum = 0.0f;
    if (i & 0b1000)
      sum += activations[0];
    if (i & 0b0100)
      sum += activations[1];
    if (i & 0b0010)
      sum += activations[2];
    if (i & 0b0001)
      sum += activations[3];
    lut[i] = sum;
  }
}

/**
 * Precompute LUTs for an entire activation vector
 *
 * @param activations Input activation vector
 * @param n Length of activation vector
 * @param luts Output LUT array (n/4 tables, each with 16 entries)
 */
inline void precompute_luts(const float *activations, size_t n, float *luts) {
  size_t num_groups = n / LUT_GROUP_SIZE;
  for (size_t g = 0; g < num_groups; ++g) {
    precompute_lut(activations + g * LUT_GROUP_SIZE, luts + g * LUT_ENTRIES);
  }
}

// ============================================================================
// Weight Index Extraction
// ============================================================================

/**
 * Extract 4-bit LUT indices from bit-planes
 *
 * For ternary weights, we need two lookups:
 * - positive_idx: bits where weight = +1
 * - negative_idx: bits where weight = -1
 *
 * Result = LUT[positive_idx] - LUT[negative_idx]
 *
 * @param sign_plane Sign bits (1 = negative)
 * @param zero_plane Zero flags (1 = zero)
 * @param group_idx Index of 4-weight group within block
 * @param pos_idx Output: index for positive weights
 * @param neg_idx Output: index for negative weights
 */
inline void extract_indices(const uint8_t *sign_plane,
                            const uint8_t *zero_plane, size_t group_idx,
                            uint8_t &pos_idx, uint8_t &neg_idx) {
  // Each group is 4 weights = 4 bits
  // Groups are packed 2 per byte
  size_t byte_idx = group_idx / 2;
  size_t bit_offset = (group_idx % 2) * 4;

  // Extract 4 bits for this group
  uint8_t sign_bits = (sign_plane[byte_idx] >> (4 - bit_offset)) & 0x0F;
  uint8_t zero_bits = (zero_plane[byte_idx] >> (4 - bit_offset)) & 0x0F;

  // non_zero = ~zero_bits & 0x0F
  uint8_t non_zero = (~zero_bits) & 0x0F;

  // positive: non-zero AND not negative
  pos_idx = non_zero & (~sign_bits);

  // negative: non-zero AND negative
  neg_idx = non_zero & sign_bits;
}

// ============================================================================
// Scalar GEMV Implementation
// ============================================================================

/**
 * LUT-based GEMV (General Matrix-Vector multiply)
 *
 * Computes: y = W * x
 * Where W is stored in bit-plane format
 *
 * @param blocks Packed weight blocks
 * @param num_blocks Number of blocks
 * @param activations Input vector (length = M * BLOCK_SIZE)
 * @param output Output vector (length = num_blocks after reduction)
 * @param M Number of rows (output dimension)
 * @param K Number of columns (input dimension)
 */
void lut_gemv_scalar(const BitPlaneBlock *blocks, size_t num_blocks,
                     const float *activations, float *output, size_t M,
                     size_t K);

// ============================================================================
// SIMD GEMV Implementations
// ============================================================================

#if defined(LUTMAC_HAS_AVX2) || defined(LUTMAC_AVX2)
void lut_gemv_avx2(const BitPlaneBlock *blocks, size_t num_blocks,
                   const float *activations, float *output, size_t M, size_t K);
#endif

#if defined(LUTMAC_HAS_AVX512) || defined(LUTMAC_AVX512)
void lut_gemv_avx512(const BitPlaneBlock *blocks, size_t num_blocks,
                     const float *activations, float *output, size_t M,
                     size_t K);
#endif

#if defined(LUTMAC_HAS_NEON) || defined(LUTMAC_NEON)
void lut_gemv_neon(const BitPlaneBlock *blocks, size_t num_blocks,
                   const float *activations, float *output, size_t M, size_t K);

void lut_gemv_binary_neon(const BinaryBlock *blocks, size_t num_blocks,
                          const float *activations, float *output, size_t M,
                          size_t K);
#endif

// ============================================================================
// Dispatcher
// ============================================================================

/**
 * Binary GEMV kernel (scalar)
 */
void lut_gemv_binary_scalar(const BinaryBlock *blocks, size_t num_blocks,
                            const float *activations, float *output, size_t M,
                            size_t K);

/**
 * Binary GEMV dispatcher
 */
inline void lut_gemv_binary(const BinaryBlock *blocks, size_t num_blocks,
                            const float *activations, float *output, size_t M,
                            size_t K) {
#if defined(LUTMAC_HAS_AVX512) || defined(LUTMAC_AVX512)
  lut_gemv_binary_avx512(blocks, num_blocks, activations, output, M, K);
#elif defined(LUTMAC_HAS_AVX2) || defined(LUTMAC_AVX2)
  lut_gemv_binary_avx2(blocks, num_blocks, activations, output, M, K);
#elif defined(LUTMAC_HAS_NEON) || defined(LUTMAC_NEON)
  lut_gemv_binary_neon(blocks, num_blocks, activations, output, M, K);
#else
  lut_gemv_binary_scalar(blocks, num_blocks, activations, output, M, K);
#endif
}

/**
 * Automatically select best GEMV implementation
 */
inline void lut_gemv(const BitPlaneBlock *blocks, size_t num_blocks,
                     const float *activations, float *output, size_t M,
                     size_t K) {
  /* Force scalar fallback for verification
  lut_gemv_scalar(blocks, num_blocks, activations, output, M, K);
  */
#if defined(LUTMAC_HAS_AVX512) || defined(LUTMAC_AVX512)
  lut_gemv_avx512(blocks, num_blocks, activations, output, M, K);
#elif defined(LUTMAC_HAS_AVX2) || defined(LUTMAC_AVX2)
  lut_gemv_avx2(blocks, num_blocks, activations, output, M, K);
#elif defined(LUTMAC_HAS_NEON) || defined(LUTMAC_NEON)
  lut_gemv_neon(blocks, num_blocks, activations, output, M, K);
#else
  lut_gemv_scalar(blocks, num_blocks, activations, output, M, K);
#endif
}

// ============================================================================
// Linear Layer
// ============================================================================

/**
 * Int4 GEMV kernel (scalar)
 */
void int4_gemv_scalar(const Int4Block *blocks, size_t num_blocks,
                      const float *activations, float *output, size_t M,
                      size_t K);

#if defined(LUTMAC_HAS_AVX2) || defined(LUTMAC_AVX2)
void int4_gemv_avx2(const Int4Block *blocks, size_t num_blocks,
                    const float *activations, float *output, size_t M,
                    size_t K);
#endif

#if defined(LUTMAC_HAS_AVX512) || defined(LUTMAC_AVX512)
void int4_gemv_avx512(const Int4Block *blocks, size_t num_blocks,
                      const float *activations, float *output, size_t M,
                      size_t K);
#endif

#if defined(LUTMAC_HAS_NEON) || defined(LUTMAC_NEON)
void int4_gemv_neon(const Int4Block *blocks, size_t num_blocks,
                    const float *activations, float *output, size_t M,
                    size_t K);
#endif

/**
 * Int4 GEMV dispatcher
 */
inline void int4_gemv(const Int4Block *blocks, size_t num_blocks,
                      const float *activations, float *output, size_t M,
                      size_t K) {
#if defined(LUTMAC_HAS_AVX512) || defined(LUTMAC_AVX512)
  int4_gemv_avx512(blocks, num_blocks, activations, output, M, K);
#elif defined(LUTMAC_HAS_AVX2) || defined(LUTMAC_AVX2)
  int4_gemv_avx2(blocks, num_blocks, activations, output, M, K);
#elif defined(LUTMAC_HAS_NEON) || defined(LUTMAC_NEON)
  int4_gemv_neon(blocks, num_blocks, activations, output, M, K);
#else
  int4_gemv_scalar(blocks, num_blocks, activations, output, M, K);
#endif
}

/**
 * Int2 GEMV kernel (scalar)
 */
void int2_gemv_scalar(const Int2Block *blocks, size_t num_blocks,
                      const float *activations, float *output, size_t M,
                      size_t K);

#if defined(LUTMAC_HAS_AVX2) || defined(LUTMAC_AVX2)
void int2_gemv_avx2(const Int2Block *blocks, size_t num_blocks,
                    const float *activations, float *output, size_t M,
                    size_t K);
#endif

#if defined(LUTMAC_HAS_AVX512) || defined(LUTMAC_AVX512)
void int2_gemv_avx512(const Int2Block *blocks, size_t num_blocks,
                      const float *activations, float *output, size_t M,
                      size_t K);
#endif

#if defined(LUTMAC_HAS_NEON) || defined(LUTMAC_NEON)
void int2_gemv_neon(const Int2Block *blocks, size_t num_blocks,
                    const float *activations, float *output, size_t M,
                    size_t K);
#endif

/**
 * Int2 GEMV dispatcher
 */
inline void int2_gemv(const Int2Block *blocks, size_t num_blocks,
                      const float *activations, float *output, size_t M,
                      size_t K) {
#if defined(LUTMAC_HAS_AVX512) || defined(LUTMAC_AVX512)
  int2_gemv_avx512(blocks, num_blocks, activations, output, M, K);
#elif defined(LUTMAC_HAS_AVX2) || defined(LUTMAC_AVX2)
  int2_gemv_avx2(blocks, num_blocks, activations, output, M, K);
#elif defined(LUTMAC_HAS_NEON) || defined(LUTMAC_NEON)
  int2_gemv_neon(blocks, num_blocks, activations, output, M, K);
#else
  int2_gemv_scalar(blocks, num_blocks, activations, output, M, K);
#endif
}

/**
 * Int3 GEMV kernel (scalar)
 */
void int3_gemv_scalar(const Int3Block *blocks, size_t num_blocks,
                      const float *activations, float *output, size_t M,
                      size_t K);

#if defined(LUTMAC_HAS_AVX2) || defined(LUTMAC_AVX2)
void int3_gemv_avx2(const Int3Block *blocks, size_t num_blocks,
                    const float *activations, float *output, size_t M,
                    size_t K);
#endif

#if defined(LUTMAC_HAS_AVX512) || defined(LUTMAC_AVX512)
void int3_gemv_avx512(const Int3Block *blocks, size_t num_blocks,
                      const float *activations, float *output, size_t M,
                      size_t K);
#endif

#if defined(LUTMAC_HAS_NEON) || defined(LUTMAC_NEON)
void int3_gemv_neon(const Int3Block *blocks, size_t num_blocks,
                    const float *activations, float *output, size_t M,
                    size_t K);
#endif

/**
 * Int3 GEMV dispatcher
 */
inline void int3_gemv(const Int3Block *blocks, size_t num_blocks,
                      const float *activations, float *output, size_t M,
                      size_t K) {
#if defined(LUTMAC_HAS_AVX512) || defined(LUTMAC_AVX512)
  int3_gemv_avx512(blocks, num_blocks, activations, output, M, K);
#elif defined(LUTMAC_HAS_AVX2) || defined(LUTMAC_AVX2)
  int3_gemv_avx2(blocks, num_blocks, activations, output, M, K);
#elif defined(LUTMAC_HAS_NEON) || defined(LUTMAC_NEON)
  int3_gemv_neon(blocks, num_blocks, activations, output, M, K);
#else
  int3_gemv_scalar(blocks, num_blocks, activations, output, M, K);
#endif
}

/**
 * Int5 GEMV kernel (scalar)
 */
void int5_gemv_scalar(const Int5Block *blocks, size_t num_blocks,
                      const float *activations, float *output, size_t M,
                      size_t K);

/**
 * Int5 GEMV dispatcher
 */
inline void int5_gemv(const Int5Block *blocks, size_t num_blocks,
                      const float *activations, float *output, size_t M,
                      size_t K) {
  int5_gemv_scalar(blocks, num_blocks, activations, output, M, K);
}

/**
 * Int6 GEMV kernel (scalar)
 */
void int6_gemv_scalar(const Int6Block *blocks, size_t num_blocks,
                      const float *activations, float *output, size_t M,
                      size_t K);

/**
 * Int6 GEMV dispatcher
 */
inline void int6_gemv(const Int6Block *blocks, size_t num_blocks,
                      const float *activations, float *output, size_t M,
                      size_t K) {
  int6_gemv_scalar(blocks, num_blocks, activations, output, M, K);
}

/**
 * Int8 GEMV kernel (scalar)
 */
void int8_gemv_scalar(const Int8Block *blocks, size_t num_blocks,
                      const float *activations, float *output, size_t M,
                      size_t K);

#if defined(LUTMAC_HAS_AVX2) || defined(LUTMAC_AVX2)
void int8_gemv_avx2(const Int8Block *blocks, size_t num_blocks,
                    const float *activations, float *output, size_t M,
                    size_t K);
#endif

#if defined(LUTMAC_HAS_AVX512) || defined(LUTMAC_AVX512)
void int8_gemv_avx512(const Int8Block *blocks, size_t num_blocks,
                      const float *activations, float *output, size_t M,
                      size_t K);
#endif

#if defined(LUTMAC_HAS_NEON) || defined(LUTMAC_NEON)
void int8_gemv_neon(const Int8Block *blocks, size_t num_blocks,
                    const float *activations, float *output, size_t M,
                    size_t K);
#endif

/**
 * Int8 GEMV dispatcher
 */
inline void int8_gemv(const Int8Block *blocks, size_t num_blocks,
                      const float *activations, float *output, size_t M,
                      size_t K) {
#if defined(LUTMAC_HAS_AVX512) || defined(LUTMAC_AVX512)
  int8_gemv_avx512(blocks, num_blocks, activations, output, M, K);
#elif defined(LUTMAC_HAS_AVX2) || defined(LUTMAC_AVX2)
  int8_gemv_avx2(blocks, num_blocks, activations, output, M, K);
#elif defined(LUTMAC_HAS_NEON) || defined(LUTMAC_NEON)
  int8_gemv_neon(blocks, num_blocks, activations, output, M, K);
#else
  int8_gemv_scalar(blocks, num_blocks, activations, output, M, K);
#endif
}

/**
 * LUT-based linear layer: y = W * x + b
 *
 * @param weight Packed weight tensor
 * @param bias Bias vector (can be nullptr)
 * @param input Input activations
 * @param output Output vector
 * @param batch_size Batch size
 */
void lut_linear(const PackedTensor &weight, const float *bias,
                const float *input, float *output, size_t batch_size = 1);

} // namespace lutmac
