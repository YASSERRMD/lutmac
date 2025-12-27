/**
 * LutMac: Embedding Layer
 *
 * Token and position embedding implementations.
 */

#include "lutmac/types.hpp"
#include <cmath>
#include <cstring>

namespace lutmac {

/**
 * Token embedding lookup
 *
 * @param embedding_table Embedding matrix [vocab_size, hidden_size]
 * @param token_ids Input token IDs
 * @param output Output embeddings
 * @param num_tokens Number of tokens
 * @param hidden_size Hidden dimension
 */
void embed_tokens(const float *embedding_table, const int *token_ids,
                  float *output, size_t num_tokens, size_t hidden_size) {
  for (size_t i = 0; i < num_tokens; ++i) {
    int token_id = token_ids[i];
    const float *embed = embedding_table + token_id * hidden_size;
    std::memcpy(output + i * hidden_size, embed, hidden_size * sizeof(float));
  }
}

/**
 * Rotary Position Embedding (RoPE) with Scaling support
 *
 * Supports Llama 3 style linear and frequency scaling.
 */
void apply_rope(float *q, float *k, size_t position, size_t num_heads,
                size_t num_kv_heads, size_t head_dim, float theta,
                float factor = 1.0f) {
  size_t half_dim = head_dim / 2;

  for (size_t h = 0; h < num_heads; ++h) {
    float *qh = q + h * head_dim;
    for (size_t d = 0; d < half_dim; ++d) {
      // Apply scaling factor to frequency if needed
      float freq = 1.0f / std::pow(theta, (2.0f * d) / head_dim);
      float angle = (position / factor) * freq;

      float cos_val = std::cos(angle);
      float sin_val = std::sin(angle);

      float q0 = qh[d];
      float q1 = qh[d + half_dim];

      qh[d] = q0 * cos_val - q1 * sin_val;
      qh[d + half_dim] = q0 * sin_val + q1 * cos_val;
    }
  }

  for (size_t h = 0; h < num_kv_heads; ++h) {
    float *kh = k + h * head_dim;
    for (size_t d = 0; d < half_dim; ++d) {
      float freq = 1.0f / std::pow(theta, (2.0f * d) / head_dim);
      float angle = (position / factor) * freq;

      float cos_val = std::cos(angle);
      float sin_val = std::sin(angle);

      float k0 = kh[d];
      float k1 = kh[d + half_dim];

      kh[d] = k0 * cos_val - k1 * sin_val;
      kh[d + half_dim] = k0 * sin_val + k1 * cos_val;
    }
  }
}

} // namespace lutmac
