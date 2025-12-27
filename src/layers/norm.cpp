/**
 * LutMac: Normalization Layers
 *
 * RMSNorm and LayerNorm implementations.
 */

#include "lutmac/types.hpp"
#include <cmath>
#include <cstring>

namespace lutmac {

/**
 * RMSNorm: Root Mean Square Layer Normalization
 *
 * output = (x / sqrt(mean(x^2) + eps)) * weight
 */
void rms_norm(const float *input, const float *weight, float *output,
              size_t hidden_size, float eps = 1e-5f) {
  // Compute sum of squares
  float sum_sq = 0.0f;
  for (size_t i = 0; i < hidden_size; ++i) {
    sum_sq += input[i] * input[i];
  }

  // RMS
  float rms = std::sqrt(sum_sq / hidden_size + eps);
  float inv_rms = 1.0f / rms;

  // Normalize and scale
  for (size_t i = 0; i < hidden_size; ++i) {
    output[i] = input[i] * inv_rms * weight[i];
  }
}

/**
 * Gemma RMSNorm: Weights are offset by 1.0
 * output = normalized * (1.0 + weight)
 */
void gemma_rms_norm(const float *input, const float *weight, float *output,
                    size_t hidden_size, float eps = 1e-6f) {
  float sum_sq = 0.0f;
  for (size_t i = 0; i < hidden_size; ++i) {
    sum_sq += input[i] * input[i];
  }

  float rms = std::sqrt(sum_sq / hidden_size + eps);
  float inv_rms = 1.0f / rms;

  for (size_t i = 0; i < hidden_size; ++i) {
    output[i] = input[i] * inv_rms * (1.0f + weight[i]);
  }
}

/**
 * LayerNorm: Standard Layer Normalization
 *
 * output = ((x - mean) / sqrt(var + eps)) * weight + bias
 */
void layer_norm(const float *input, const float *weight, const float *bias,
                float *output, size_t hidden_size, float eps = 1e-5f) {
  // Compute mean
  float mean = 0.0f;
  for (size_t i = 0; i < hidden_size; ++i) {
    mean += input[i];
  }
  mean /= hidden_size;

  // Compute variance
  float var = 0.0f;
  for (size_t i = 0; i < hidden_size; ++i) {
    float diff = input[i] - mean;
    var += diff * diff;
  }
  var /= hidden_size;

  // Normalize
  float inv_std = 1.0f / std::sqrt(var + eps);
  for (size_t i = 0; i < hidden_size; ++i) {
    float norm = (input[i] - mean) * inv_std;
    output[i] = norm * weight[i] + (bias ? bias[i] : 0.0f);
  }
}

} // namespace lutmac
