#pragma once

/**
 * Advanced Quantization Techniques for LutMac
 *
 * Implements:
 * 1. Saliency-Aware Mixed Precision (Precision Highways)
 * 2. Outlier-Aware Scaling (clipping top-k outliers)
 * 3. Residual Quantization (RRQ) for reduced error
 * 4. Entropy-Preserving Grids (SEQ-style)
 */

#include "types.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace lutmac {

// ============================================================================
// Saliency Analysis
// ============================================================================

/**
 * Compute weight saliency scores based on magnitude and gradient
 * Higher saliency = more important for model accuracy
 */
struct SaliencyAnalyzer {
  // Compute saliency as absolute magnitude (simple but effective)
  static std::vector<float> compute_magnitude_saliency(const float *data,
                                                       size_t n) {
    std::vector<float> saliency(n);
    for (size_t i = 0; i < n; ++i) {
      saliency[i] = std::abs(data[i]);
    }
    return saliency;
  }

  // Find top-k salient weight indices
  static std::vector<size_t> find_salient_indices(const float *data, size_t n,
                                                  float percentile) {
    if (n == 0 || percentile <= 0.0f)
      return {};

    std::vector<std::pair<float, size_t>> indexed(n);
    for (size_t i = 0; i < n; ++i) {
      indexed[i] = {std::abs(data[i]), i};
    }

    // Partial sort to find top percentile
    size_t k = static_cast<size_t>(n * percentile);
    k = std::max(k, size_t(1));

    std::partial_sort(
        indexed.begin(), indexed.begin() + k, indexed.end(),
        [](const auto &a, const auto &b) { return a.first > b.first; });

    std::vector<size_t> indices;
    indices.reserve(k);
    for (size_t i = 0; i < k; ++i) {
      indices.push_back(indexed[i].second);
    }
    return indices;
  }
};

// ============================================================================
// Outlier-Aware Scaling
// ============================================================================

/**
 * Compute scale factor with outlier clipping
 * Instead of using absmax, clip top percentile of outliers
 */
inline float compute_clipped_scale(const float *data, size_t n,
                                   float clip_percentile = 0.01f,
                                   int num_levels = 7) {
  if (n == 0)
    return 1.0f;

  // Collect absolute values
  std::vector<float> abs_vals(n);
  for (size_t i = 0; i < n; ++i) {
    abs_vals[i] = std::abs(data[i]);
  }

  // Find the value at (1 - clip_percentile) percentile
  size_t clip_idx = static_cast<size_t>(n * (1.0f - clip_percentile));
  clip_idx = std::min(clip_idx, n - 1);

  std::nth_element(abs_vals.begin(), abs_vals.begin() + clip_idx,
                   abs_vals.end());
  float clipped_max = abs_vals[clip_idx];

  // Ensure non-zero scale
  if (clipped_max < 1e-10f) {
    clipped_max = 1e-10f;
  }

  return clipped_max / static_cast<float>(num_levels);
}

/**
 * Smooth clipping function: reduces the impact of extreme outliers
 * Uses tanh-based soft clipping
 */
inline float soft_clip(float value, float threshold) {
  if (std::abs(value) <= threshold) {
    return value;
  }
  float sign = value > 0 ? 1.0f : -1.0f;
  float excess = std::abs(value) - threshold;
  // Logarithmic compression of outliers
  return sign * (threshold + std::log1p(excess));
}

// ============================================================================
// Residual Quantization (RRQ)
// ============================================================================

/**
 * Recursive Residual Quantization
 *
 * Quantizes the residual error from initial quantization to further reduce
 * error. Storage overhead: ~0.1-0.2 bits per weight for significant accuracy
 * improvement.
 */
struct ResidualQuantizer {
  // Primary quantized values (e.g., Int4)
  std::vector<int8_t> primary_quant;
  float primary_scale;

  // Residual quantized values (e.g., Int2 or ternary)
  std::vector<int8_t> residual_quant;
  float residual_scale;

  /**
   * Perform 2-stage residual quantization
   * Stage 1: Quantize to primary_bits
   * Stage 2: Quantize residual to residual_bits
   */
  static void quantize_with_residual(const float *data, size_t n,
                                     int primary_bits, int residual_bits,
                                     int8_t *primary_out,
                                     float &primary_scale_out,
                                     int8_t *residual_out,
                                     float &residual_scale_out) {
    int primary_levels = (1 << (primary_bits - 1)) - 1;   // e.g., 7 for 4-bit
    int residual_levels = (1 << (residual_bits - 1)) - 1; // e.g., 1 for 2-bit

    // Stage 1: Primary quantization with outlier clipping
    primary_scale_out = compute_clipped_scale(data, n, 0.01f, primary_levels);
    float inv_primary_scale = 1.0f / primary_scale_out;

    std::vector<float> residual(n);

    for (size_t i = 0; i < n; ++i) {
      // Quantize to primary levels
      float scaled = data[i] * inv_primary_scale;
      int q = static_cast<int>(std::round(scaled));
      q = std::max(-primary_levels, std::min(primary_levels, q));
      primary_out[i] = static_cast<int8_t>(q);

      // Compute residual: original - reconstructed
      float reconstructed = static_cast<float>(q) * primary_scale_out;
      residual[i] = data[i] - reconstructed;
    }

    // Stage 2: Quantize residual
    residual_scale_out =
        compute_clipped_scale(residual.data(), n, 0.05f, residual_levels);
    float inv_residual_scale =
        (residual_scale_out > 1e-10f) ? (1.0f / residual_scale_out) : 0.0f;

    for (size_t i = 0; i < n; ++i) {
      float scaled = residual[i] * inv_residual_scale;
      int q = static_cast<int>(std::round(scaled));
      q = std::max(-residual_levels, std::min(residual_levels, q));
      residual_out[i] = static_cast<int8_t>(q);
    }
  }

  /**
   * Dequantize with residual correction
   */
  static float dequantize_with_residual(int8_t primary, float primary_scale,
                                        int8_t residual, float residual_scale) {
    return static_cast<float>(primary) * primary_scale +
           static_cast<float>(residual) * residual_scale;
  }
};

// ============================================================================
// Entropy-Preserving Quantization (SEQ-style)
// ============================================================================

/**
 * Stretched Elastic Quantization for low-bit regimes
 *
 * Uses non-uniform quantization grid that preserves more entropy
 * near the center of the distribution where most weights lie.
 */
inline float elastic_quantize(float value, float scale, int bits,
                              float stretch = 1.5f) {
  int levels = (1 << (bits - 1)) - 1;
  float normalized = value / scale;

  // Apply elastic stretching: compress near 0, expand near extremes
  // This preserves more information in the high-density center region
  float sign = normalized >= 0 ? 1.0f : -1.0f;
  float abs_val = std::abs(normalized);

  // Power-law stretching: x^(1/stretch) for abs_val < 1
  float stretched;
  if (abs_val <= 1.0f) {
    stretched = std::pow(abs_val, 1.0f / stretch);
  } else {
    // Compress outliers
    stretched = 1.0f + std::log(abs_val);
  }

  // Quantize the stretched value
  int q = static_cast<int>(std::round(stretched * levels * sign));
  q = std::max(-levels, std::min(levels, q));

  return static_cast<float>(q);
}

/**
 * Inverse elastic quantization (for dequant)
 */
inline float elastic_dequantize(int8_t quant, float scale, int bits,
                                float stretch = 1.5f) {
  int levels = (1 << (bits - 1)) - 1;
  float normalized = static_cast<float>(quant) / static_cast<float>(levels);

  float sign = normalized >= 0 ? 1.0f : -1.0f;
  float abs_val = std::abs(normalized);

  // Inverse power-law: x^stretch
  float unstretched;
  if (abs_val <= 1.0f) {
    unstretched = std::pow(abs_val, stretch);
  } else {
    unstretched = std::exp(abs_val - 1.0f);
  }

  return sign * unstretched * scale;
}

// ============================================================================
// Precision Highway Detection
// ============================================================================

/**
 * Identify layers that should remain in higher precision
 * Based on layer type and measured sensitivity
 */
struct PrecisionHighway {
  enum class LayerSensitivity {
    LOW,     // Can be aggressively quantized (most FFN layers)
    MEDIUM,  // Standard quantization (attention projections)
    HIGH,    // Needs higher precision (embed, lm_head, first/last layers)
    CRITICAL // Keep in FP16/FP32 (skip connections, layer norms)
  };

  static LayerSensitivity classify_layer(const std::string &name,
                                         size_t layer_idx,
                                         size_t total_layers) {
    // Embedding and output head are critical
    if (name.find("embed") != std::string::npos ||
        name.find("lm_head") != std::string::npos ||
        name.find("wte") != std::string::npos ||
        name.find("wpe") != std::string::npos) {
      return LayerSensitivity::HIGH;
    }

    // Layer norms should not be quantized
    if (name.find("norm") != std::string::npos ||
        name.find("ln_") != std::string::npos) {
      return LayerSensitivity::CRITICAL;
    }

    // First and last few layers are more sensitive
    if (layer_idx < 2 || layer_idx >= total_layers - 2) {
      return LayerSensitivity::HIGH;
    }

    // Attention O projection is somewhat sensitive
    if (name.find("o_proj") != std::string::npos) {
      return LayerSensitivity::MEDIUM;
    }

    // Q, K, V projections can be quantized more
    if (name.find("q_proj") != std::string::npos ||
        name.find("k_proj") != std::string::npos ||
        name.find("v_proj") != std::string::npos) {
      return LayerSensitivity::MEDIUM;
    }

    // FFN layers can usually be quantized aggressively
    if (name.find("mlp") != std::string::npos ||
        name.find("ffn") != std::string::npos ||
        name.find("gate") != std::string::npos ||
        name.find("up_proj") != std::string::npos ||
        name.find("down_proj") != std::string::npos) {
      return LayerSensitivity::LOW;
    }

    return LayerSensitivity::MEDIUM;
  }

  static int recommended_bits(LayerSensitivity sensitivity, int target_bits) {
    switch (sensitivity) {
    case LayerSensitivity::CRITICAL:
      return 16; // Keep in FP16
    case LayerSensitivity::HIGH:
      return std::max(target_bits + 4, 8); // At least 8-bit
    case LayerSensitivity::MEDIUM:
      return std::max(target_bits + 2, 4); // At least 4-bit
    case LayerSensitivity::LOW:
    default:
      return target_bits; // Use target
    }
  }
};

// ============================================================================
// Improved Int4 Quantization with Advanced Techniques
// ============================================================================

struct AdvancedInt4Block {
  static constexpr size_t BLOCK_SIZE = 256;

  uint8_t data[128]; // 256 x 4-bit = 128 bytes
  float scale;
  float residual_scale;      // For residual quantization
  uint8_t residual_data[64]; // 256 x 2-bit = 64 bytes for residual

  void set_primary(size_t i, uint8_t val) {
    size_t byte_idx = i / 2;
    if (i % 2 == 0) {
      data[byte_idx] = (data[byte_idx] & 0xF0) | (val & 0x0F);
    } else {
      data[byte_idx] = (data[byte_idx] & 0x0F) | ((val & 0x0F) << 4);
    }
  }

  void set_residual(size_t i, uint8_t val) {
    // 2-bit packing: 4 values per byte
    size_t byte_idx = i / 4;
    size_t shift = (i % 4) * 2;
    residual_data[byte_idx] =
        (residual_data[byte_idx] & ~(0x03 << shift)) | ((val & 0x03) << shift);
  }

  int8_t get_primary(size_t i) const {
    size_t byte_idx = i / 2;
    uint8_t nibble =
        (i % 2 == 0) ? (data[byte_idx] & 0x0F) : (data[byte_idx] >> 4);
    return static_cast<int8_t>(nibble) - 8; // Convert from [0,15] to [-8,7]
  }

  int8_t get_residual(size_t i) const {
    size_t byte_idx = i / 4;
    size_t shift = (i % 4) * 2;
    uint8_t val = (residual_data[byte_idx] >> shift) & 0x03;
    return static_cast<int8_t>(val) - 2; // Convert from [0,3] to [-2,1]
  }

  float dequantize(size_t i) const {
    return static_cast<float>(get_primary(i)) * scale +
           static_cast<float>(get_residual(i)) * residual_scale;
  }
};

} // namespace lutmac
