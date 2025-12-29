#pragma once

/**
 * LutMac: Bit-Serial LUT Engine for Ultra-Low-Bit LLM Inference
 *
 * Core type definitions and configuration structures.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// Platform detection
#if defined(__AVX512F__) && defined(__AVX512BW__)
#define LUTMAC_HAS_AVX512 1
#endif

#if defined(__AVX2__)
#define LUTMAC_HAS_AVX2 1
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define LUTMAC_HAS_NEON 1
#endif

namespace lutmac {

// ============================================================================
// Constants
// ============================================================================

constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t SIMD_ALIGNMENT = 64; // For AVX-512
constexpr size_t LUT_GROUP_SIZE = 4;  // Number of activations per LUT
constexpr size_t LUT_ENTRIES = 16;    // 2^LUT_GROUP_SIZE

// Magic bytes for file format
constexpr char LUTMAC_MAGIC[8] = {'L', 'U', 'T', 'M', 'A', 'C', '0', '1'};
constexpr uint32_t LUTMAC_FORMAT_VERSION = 1;

// ============================================================================
// Basic Types
// ============================================================================

// Aligned allocator for SIMD operations
template <typename T, size_t Alignment = SIMD_ALIGNMENT>
struct AlignedAllocator {
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  template <typename U> struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  AlignedAllocator() noexcept = default;

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment> &) noexcept {}

  T *allocate(size_t n) {
    void *ptr = nullptr;
#if defined(_MSC_VER)
    ptr = _aligned_malloc(n * sizeof(T), Alignment);
#else
    if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
      ptr = nullptr;
    }
#endif
    if (!ptr)
      throw std::bad_alloc();
    return static_cast<T *>(ptr);
  }

  void deallocate(T *ptr, size_t) noexcept {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }

  template <typename U>
  bool operator==(const AlignedAllocator<U, Alignment> &) const noexcept {
    return true;
  }

  template <typename U>
  bool operator!=(const AlignedAllocator<U, Alignment> &) const noexcept {
    return false;
  }
};

// Aligned vector type
template <typename T> using AlignedVector = std::vector<T, AlignedAllocator<T>>;

// ============================================================================
// Quantization Types
// ============================================================================

/**
 * Ternary weight values: {-1, 0, +1}
 * Stored as 2 bits: sign bit + zero flag
 */
enum class TernaryValue : int8_t { NEG_ONE = -1, ZERO = 0, POS_ONE = 1 };

/**
 * Bit-plane packed ternary weights
 *
 * For ternary {-1, 0, +1}, we use sign-magnitude encoding:
 * - sign_plane: 1 if weight is negative, 0 otherwise
 * - zero_plane: 1 if weight is zero, 0 otherwise
 *
 * Actual value = (zero_plane ? 0 : (sign_plane ? -1 : +1))
 */
struct BitPlaneBlock {
  static constexpr size_t BLOCK_SIZE = 256; // Weights per block

  // Each plane is BLOCK_SIZE bits = 32 bytes = 256 bits
  // alignas(SIMD_ALIGNMENT) removed to reduce file size.
  // We handle alignment manually during SIMD load if needed.
  uint8_t sign_plane[BLOCK_SIZE / 8];
  uint8_t zero_plane[BLOCK_SIZE / 8];

  // Per-block scale (one scale for all 256 weights)
  // This reduces overhead from 256 bytes/block to 4 bytes/block
  float scale;

  // Get weight at index (returns dequantized value)
  inline float get(size_t idx) const {
    size_t byte_idx = idx / 8;
    size_t bit_idx = 7 - (idx % 8); // MSB-first bit order
    bool sign = (sign_plane[byte_idx] >> bit_idx) & 1;
    bool zero = (zero_plane[byte_idx] >> bit_idx) & 1;
    if (zero)
      return 0.0f;
    return (sign ? -1.0f : 1.0f) * scale;
  }

  // Set weight at index (using TernaryValue enum)
  inline void set(size_t idx, TernaryValue val) {
    size_t byte_idx = idx / 8;
    size_t bit_idx = 7 - (idx % 8);
    if (val == TernaryValue::ZERO) {
      zero_plane[byte_idx] |= (1 << bit_idx);
      sign_plane[byte_idx] &= ~(1 << bit_idx);
    } else {
      zero_plane[byte_idx] &= ~(1 << bit_idx);
      if (val == TernaryValue::NEG_ONE) {
        sign_plane[byte_idx] |= (1 << bit_idx);
      } else {
        sign_plane[byte_idx] &= ~(1 << bit_idx);
      }
    }
  }
};

/**
 * 4-bit quantized weights block
 *
 * Symmetric quantization: value = (int4 - 8) * scale
 * Range: [-8, +7] mapped to [-8*scale, +7*scale]
 */
struct Int4Block {
  static constexpr size_t BLOCK_SIZE = 256; // Weights per block

  // 256 weights * 4 bits / 8 = 128 bytes
  uint8_t data[BLOCK_SIZE / 2];

  // Per-block scale
  float scale;

  // Get weight at index (returns dequantized value)
  inline float get(size_t idx) const {
    size_t byte_idx = idx / 2;
    bool is_high = (idx % 2) == 0; // First weight in high nibble
    uint8_t nibble = is_high ? (data[byte_idx] >> 4) : (data[byte_idx] & 0x0F);
    int8_t signed_val = static_cast<int8_t>(nibble) - 8; // Centered at 0
    return signed_val * scale;
  }

  // Set weight at index
  inline void set(size_t idx, uint8_t val) {
    size_t byte_idx = idx / 2;
    bool is_high = (idx % 2) == 0;
    if (is_high) {
      data[byte_idx] = (data[byte_idx] & 0x0F) | (val << 4);
    } else {
      data[byte_idx] = (data[byte_idx] & 0xF0) | (val & 0x0F);
    }
  }
};

/**
 * 2-bit quantized weights block
 *
 * Symmetric quantization: values in [-2, +1] mapped to [0, 3]
 */
struct Int2Block {
  static constexpr size_t BLOCK_SIZE = 256;

  // 256 weights * 2 bits / 8 = 64 bytes
  uint8_t data[BLOCK_SIZE / 4];
  float scale;
  float scale2; // Secondary scale for RRQ residual

  inline float get(size_t idx) const {
    // Legacy/Debug accessor - not efficient
    size_t byte_idx = idx / 4;
    size_t shift = (idx % 4) * 2;
    uint8_t val = (data[byte_idx] >> (6 - shift)) & 0x03;

    // De-interleave bits for RRQ
    // val is [b1_i, b2_i] (2 bits)
    // Actually, let's say bit 1 (MSB of the pair) is B1 (primary), bit 0 (LSB)
    // is B2 (residual) val = (B1 << 1) | B2

    bool b1 = (val >> 1) & 1;
    bool b2 = val & 1;

    float v1 = b1 ? scale : -scale;
    float v2 = b2 ? scale2 : -scale2;
    return v1 + v2;
  }

  inline void set(size_t idx, uint8_t val) {
    size_t byte_idx = idx / 4;
    size_t shift = (idx % 4) * 2;
    data[byte_idx] &= ~(0x03 << (6 - shift));
    data[byte_idx] |= ((val & 0x03) << (6 - shift));
  }
};

/**
 * 3-bit quantized weights block
 *
 * Symmetric quantization: values in [-4, +3] mapped to [0, 7]
 */
struct Int3Block {
  static constexpr size_t BLOCK_SIZE = 256;

  // 256 weights * 3 bits / 8 = 96 bytes
  uint8_t data[BLOCK_SIZE * 3 / 8];
  float scale;

  inline float get(size_t idx) const {
    size_t bit_idx = idx * 3;
    size_t byte_idx = bit_idx / 8;
    size_t bit_offset = bit_idx % 8;

    uint16_t word = (static_cast<uint16_t>(data[byte_idx]) << 8);
    if (byte_idx + 1 < sizeof(data)) {
      word |= data[byte_idx + 1];
    }
    uint8_t val = (word >> (13 - bit_offset)) & 0x07;
    int8_t signed_val = static_cast<int8_t>(val) - 4;
    return signed_val * scale;
  }

  inline void set(size_t idx, uint8_t val) {
    size_t bit_idx = idx * 3;
    size_t byte_idx = bit_idx / 8;
    size_t bit_offset = bit_idx % 8;

    uint16_t mask = ~(0x0007 << (13 - bit_offset));
    uint16_t insert = static_cast<uint16_t>(val & 0x07) << (13 - bit_offset);

    uint16_t word = (static_cast<uint16_t>(data[byte_idx]) << 8);
    if (byte_idx + 1 < sizeof(data)) {
      word |= data[byte_idx + 1];
    }
    word = (word & mask) | insert;

    data[byte_idx] = (word >> 8) & 0xFF;
    if (byte_idx + 1 < sizeof(data)) {
      data[byte_idx + 1] = word & 0xFF;
    }
  }
};

/**
 * 5-bit quantized weights block
 *
 * Symmetric quantization: values in [-16, +15] mapped to [0, 31]
 */
struct Int5Block {
  static constexpr size_t BLOCK_SIZE = 256;

  // 256 weights * 5 bits / 8 = 160 bytes
  uint8_t data[BLOCK_SIZE * 5 / 8];
  float scale;

  inline float get(size_t idx) const {
    size_t bit_idx = idx * 5;
    size_t byte_idx = bit_idx / 8;
    size_t bit_offset = bit_idx % 8;

    // Read up to 2 bytes to extract 5 bits
    uint16_t word = (static_cast<uint16_t>(data[byte_idx]) << 8);
    if (byte_idx + 1 < sizeof(data)) {
      word |= data[byte_idx + 1];
    }
    uint8_t val = (word >> (11 - bit_offset)) & 0x1F;
    int8_t signed_val = static_cast<int8_t>(val) - 16; // Center at 0
    return signed_val * scale;
  }

  inline void set(size_t idx, uint8_t val) {
    size_t bit_idx = idx * 5;
    size_t byte_idx = bit_idx / 8;
    size_t bit_offset = bit_idx % 8;

    uint16_t mask = ~(0x001F << (11 - bit_offset));
    uint16_t insert = static_cast<uint16_t>(val & 0x1F) << (11 - bit_offset);

    uint16_t word = (static_cast<uint16_t>(data[byte_idx]) << 8);
    if (byte_idx + 1 < sizeof(data)) {
      word |= data[byte_idx + 1];
    }
    word = (word & mask) | insert;

    data[byte_idx] = (word >> 8) & 0xFF;
    if (byte_idx + 1 < sizeof(data)) {
      data[byte_idx + 1] = word & 0xFF;
    }
  }
};

/**
 * 6-bit quantized weights block
 *
 * Symmetric quantization: values in [-32, +31] mapped to [0, 63]
 */
struct Int6Block {
  static constexpr size_t BLOCK_SIZE = 256;

  // 256 weights * 6 bits / 8 = 192 bytes
  uint8_t data[BLOCK_SIZE * 6 / 8];
  float scale;

  inline float get(size_t idx) const {
    size_t bit_idx = idx * 6;
    size_t byte_idx = bit_idx / 8;
    size_t bit_offset = bit_idx % 8;

    // Read up to 2 bytes to extract 6 bits
    uint16_t word = (static_cast<uint16_t>(data[byte_idx]) << 8);
    if (byte_idx + 1 < sizeof(data)) {
      word |= data[byte_idx + 1];
    }
    uint8_t val = (word >> (10 - bit_offset)) & 0x3F;
    int8_t signed_val = static_cast<int8_t>(val) - 32; // Center at 0
    return signed_val * scale;
  }

  inline void set(size_t idx, uint8_t val) {
    size_t bit_idx = idx * 6;
    size_t byte_idx = bit_idx / 8;
    size_t bit_offset = bit_idx % 8;

    uint16_t mask = ~(0x003F << (10 - bit_offset));
    uint16_t insert = static_cast<uint16_t>(val & 0x3F) << (10 - bit_offset);

    uint16_t word = (static_cast<uint16_t>(data[byte_idx]) << 8);
    if (byte_idx + 1 < sizeof(data)) {
      word |= data[byte_idx + 1];
    }
    word = (word & mask) | insert;

    data[byte_idx] = (word >> 8) & 0xFF;
    if (byte_idx + 1 < sizeof(data)) {
      data[byte_idx + 1] = word & 0xFF;
    }
  }
};

/**
 * Binary (1-bit) quantized weights block
 *
 * Weights are {-1, +1}.
 * Stored as 1 bit per weight: 0 -> -1, 1 -> +1.
 * 256 weights -> 256 bits = 32 bytes.
 */
struct BinaryBlock {
  static constexpr size_t BLOCK_SIZE = 256; // Weights per block

  // 256 weights * 1 bit / 8 = 32 bytes
  uint8_t data[BLOCK_SIZE / 8];

  // Per-block scale
  float scale;

  // Get weight at index (returns dequantized value)
  inline float get(size_t idx) const {
    size_t byte_idx = idx / 8;
    size_t bit_idx = idx % 8;
    bool bit_set = (data[byte_idx] >> (7 - bit_idx)) & 1;
    // 0 -> -1, 1 -> +1
    float val = bit_set ? 1.0f : -1.0f;
    return val * scale;
  }

  // Set weight at index
  inline void set(size_t idx, bool bit_set) {
    size_t byte_idx = idx / 8;
    size_t bit_idx = idx % 8;
    if (bit_set) {
      data[byte_idx] |= (1 << (7 - bit_idx));
    } else {
      data[byte_idx] &= ~(1 << (7 - bit_idx));
    }
  }
};

/**
 * Int8 Block Quantization (for sensitive layers)
 *
 * Block Size: 256 weights
 * Structure:
 *   - float scale: Block scale factor
 *   - int8_t data[256]: 8-bit quantized weights
 * Total size: 4 + 256 = 260 bytes
 */
struct Int8Block {
  static constexpr size_t BLOCK_SIZE = 256;

  float scale;
  int8_t data[BLOCK_SIZE];

  // Quantize a block of 256 floats
  void set(const float *weights) {
    // Find max abs value
    float max_abs = 0.0f;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      max_abs = std::max(max_abs, std::abs(weights[i]));
    }

    scale = max_abs / 127.0f;
    if (scale == 0.0f) {
      std::memset(data, 0, BLOCK_SIZE);
      return;
    }

    float inv_scale = 1.0f / scale;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      // Clamp to [-127, 127]
      float val = weights[i] * inv_scale;
      val = std::max(-127.0f, std::min(127.0f, val));
      data[i] = static_cast<int8_t>(std::round(val));
    }
  }

  // Dequantize (slow, for verification)
  float get(int idx) const { return data[idx] * scale; }
};

/**
 * Packed tensor (quantized) or Raw tensor (FP32)
 */
struct PackedTensor {
  std::string name;
  std::vector<size_t> shape;
  std::vector<BitPlaneBlock> blocks;      // For ternary (1.58-bit)
  std::vector<Int2Block> int2_blocks;     // For 2-bit quantization
  std::vector<Int3Block> int3_blocks;     // For 3-bit quantization
  std::vector<Int4Block> int4_blocks;     // For 4-bit quantization
  std::vector<Int5Block> int5_blocks;     // For 5-bit quantization
  std::vector<Int6Block> int6_blocks;     // For 6-bit quantization
  std::vector<BinaryBlock> binary_blocks; // For binary (1-bit)
  std::vector<Int8Block> int8_blocks;     // For Int8 (8-bit)
  std::vector<float> raw_data;
  float global_scale = 1.0f;
  bool is_quantized = true;
  int quant_bits = 2; // 1=binary, 2=ternary, 3, 4, 5, 6, 8

  size_t num_elements() const {
    size_t n = 1;
    for (auto s : shape)
      n *= s;
    return n;
  }
};

// ============================================================================
// LUT Types
// ============================================================================

/**
 * Precomputed LUT for a group of activations
 *
 * For LUT_GROUP_SIZE=4 activations (a0, a1, a2, a3):
 * entries[i] = sum of activations where bit j of i is set
 *
 * Example:
 *   entries[0b0000] = 0
 *   entries[0b0001] = a3
 *   entries[0b0010] = a2
 *   entries[0b0011] = a2 + a3
 *   entries[0b1111] = a0 + a1 + a2 + a3
 */
struct LUTTable {
  alignas(SIMD_ALIGNMENT) float entries[LUT_ENTRIES];

  void precompute(const float *activations) {
    entries[0] = 0.0f;
    for (size_t i = 1; i < LUT_ENTRIES; ++i) {
      float sum = 0.0f;
      for (size_t j = 0; j < LUT_GROUP_SIZE; ++j) {
        if (i & (1 << (LUT_GROUP_SIZE - 1 - j))) {
          sum += activations[j];
        }
      }
      entries[i] = sum;
    }
  }
};

// ============================================================================
// Model Configuration
// ============================================================================

enum class ActivationType {
  RELU,
  GELU,
  SILU, // SwiGLU variant
  SWIGLU
};

enum class NormType { LAYER_NORM, RMS_NORM };

enum class LayerType { ATTENTION = 0, CONV = 1 };

struct ModelConfig {
  std::string model_type = "llama";

  // Architecture
  // Architecture
  size_t vocab_size = 32000;
  int bos_token_id = -1;
  int eos_token_id = -1;
  float rope_theta = 10000.0f; // Default to 10k, but Qwen2.5 uses 1M

  size_t hidden_size = 2048;
  size_t intermediate_size = 5632;
  size_t num_hidden_layers = 22;
  size_t num_attention_heads = 32;
  size_t num_key_value_heads = 4; // For GQA
  size_t max_position_embeddings = 2048;

  // Normalization
  NormType norm_type = NormType::RMS_NORM;
  float rms_norm_eps = 1e-5f;

  // Activation
  ActivationType activation = ActivationType::SILU;

  // Quantization
  float bits_per_weight = 1.58f;
  size_t group_size = LUT_GROUP_SIZE;

  // Rope
  // float rope_theta = 10000.0f; // Duplicate removed

  // Explicit head_dim (0 = inferred from hidden/heads)
  size_t head_dim_val = 0;

  // Llama 3 RoPE Scaling
  std::string rope_scaling_type = "";
  float rope_scaling_factor = 1.0f;
  float rope_scaling_low_freq_factor = 1.0f;
  float rope_scaling_high_freq_factor = 1.0f;
  size_t rope_scaling_orig_max_req = 8192;

  // Derived
  size_t head_dim() const {
    return head_dim_val > 0 ? head_dim_val
                            : (hidden_size / num_attention_heads);
  }
  size_t kv_head_dim() const { return head_dim(); }
};

// ============================================================================
// Runtime State
// ============================================================================

/**
 * KV Cache for autoregressive generation
 */
struct KVCache {
  size_t max_seq_len;
  size_t num_layers;
  size_t num_kv_heads;
  size_t head_dim;

  // Shape: [num_layers, 2, max_seq_len, num_kv_heads, head_dim]
  AlignedVector<float> cache;
  size_t current_pos = 0;

  void allocate(const ModelConfig &config, size_t max_len) {
    max_seq_len = max_len;
    num_layers = config.num_hidden_layers;
    num_kv_heads = config.num_key_value_heads;
    head_dim = config.head_dim();

    size_t total = num_layers * 2 * max_seq_len * num_kv_heads * head_dim;
    cache.resize(total, 0.0f);
  }

  float *key_cache(size_t layer) {
    return cache.data() + layer * 2 * max_seq_len * num_kv_heads * head_dim;
  }

  float *value_cache(size_t layer) {
    return cache.data() +
           (layer * 2 + 1) * max_seq_len * num_kv_heads * head_dim;
  }
};

/**
 * Generation parameters
 */
struct GenerationConfig {
  size_t max_new_tokens = 128;
  float temperature = 0.7f;
  float top_p = 0.9f;
  float min_p =
      0.0f; // Min-P sampling (0 = disabled, recommended 0.15 for LFM2)
  int top_k = 40;
  float repetition_penalty = 1.1f;
  uint64_t seed = 42;
  std::vector<int> stop_tokens;
};

// ============================================================================
// Utility Functions
// ============================================================================

inline size_t align_up(size_t n, size_t alignment) {
  return (n + alignment - 1) / alignment * alignment;
}

inline size_t div_ceil(size_t n, size_t d) { return (n + d - 1) / d; }

} // namespace lutmac
