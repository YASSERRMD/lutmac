/**
 * LutMac: Attention Layer
 *
 * Multi-head attention with LUT-based Q/K/V projections and GQA support.
 */

#include "lutmac/lut_gemm.hpp"
#include "lutmac/types.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace lutmac {

// Forward declarations
void apply_rope(float *q, float *k, size_t position, size_t num_heads,
                size_t num_kv_heads, size_t head_dim, float theta,
                float factor);

/**
 * Softmax over last dimension
 */
void softmax(float *data, size_t n) {
  // Find max for numerical stability
  float max_val = data[0];
  for (size_t i = 1; i < n; ++i) {
    max_val = std::max(max_val, data[i]);
  }

  // Exp and sum
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    data[i] = std::exp(data[i] - max_val);
    sum += data[i];
  }

  // Normalize
  float inv_sum = 1.0f / sum;
  for (size_t i = 0; i < n; ++i) {
    data[i] *= inv_sum;
  }
}

/**
 * Grouped Query Attention (GQA)
 *
 * Supports different numbers of query heads and key/value heads.
 */
struct AttentionLayer {
  // Weights (quantized)
  PackedTensor q_proj;
  PackedTensor k_proj;
  PackedTensor v_proj;
  PackedTensor o_proj;

  // Config
  size_t hidden_size;
  size_t num_heads;
  size_t num_kv_heads;
  size_t head_dim;
  float rope_theta;
  float rope_scaling_factor;

  // Scratch buffers
  AlignedVector<float> q_buf;
  AlignedVector<float> k_buf;
  AlignedVector<float> v_buf;
  AlignedVector<float> attn_weights;
  AlignedVector<float> attn_output;

  void allocate(const ModelConfig &config) {
    hidden_size = config.hidden_size;
    num_heads = config.num_attention_heads;
    num_kv_heads = config.num_key_value_heads;
    head_dim = config.head_dim();
    rope_theta = config.rope_theta;
    rope_scaling_factor = config.rope_scaling_factor;

    q_buf.resize(num_heads * head_dim);
    k_buf.resize(num_kv_heads * head_dim);
    v_buf.resize(num_kv_heads * head_dim);
    attn_weights.resize(num_heads * config.max_position_embeddings);
    attn_output.resize(hidden_size);
  }

  void forward(const float *hidden_states, KVCache &kv_cache, size_t layer_idx,
               size_t position, float *output) {
    // Q projection
    lut_linear(q_proj, nullptr, hidden_states, q_buf.data(), 1);

    // K projection
    lut_linear(k_proj, nullptr, hidden_states, k_buf.data(), 1);

    // V projection
    lut_linear(v_proj, nullptr, hidden_states, v_buf.data(), 1);

    // Apply RoPE
    apply_rope(q_buf.data(), k_buf.data(), position, num_heads, num_kv_heads,
               head_dim, rope_theta, rope_scaling_factor);

    // Store K and V in cache
    float *k_cache = kv_cache.key_cache(layer_idx);
    float *v_cache = kv_cache.value_cache(layer_idx);

    std::memcpy(k_cache + position * num_kv_heads * head_dim, k_buf.data(),
                num_kv_heads * head_dim * sizeof(float));
    std::memcpy(v_cache + position * num_kv_heads * head_dim, v_buf.data(),
                num_kv_heads * head_dim * sizeof(float));

    // Compute attention scores
    size_t seq_len = position + 1;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Group ratio for GQA
    size_t kv_group_size = num_heads / num_kv_heads;

    std::fill(attn_output.begin(), attn_output.end(), 0.0f);

    for (size_t h = 0; h < num_heads; ++h) {
      size_t kv_h = h / kv_group_size;
      const float *q_head = q_buf.data() + h * head_dim;

      // Compute attention scores for this head
      for (size_t t = 0; t < seq_len; ++t) {
        const float *k_t =
            k_cache + t * num_kv_heads * head_dim + kv_h * head_dim;

        float score = 0.0f;
        for (size_t d = 0; d < head_dim; ++d) {
          score += q_head[d] * k_t[d];
        }
        attn_weights[h * seq_len + t] = score * scale;
      }

      // Causal mask (already handled by seq_len)
      // Softmax
      softmax(attn_weights.data() + h * seq_len, seq_len);

      // Weighted sum of values
      for (size_t t = 0; t < seq_len; ++t) {
        const float *v_t =
            v_cache + t * num_kv_heads * head_dim + kv_h * head_dim;
        float weight = attn_weights[h * seq_len + t];

        for (size_t d = 0; d < head_dim; ++d) {
          attn_output[h * head_dim + d] += weight * v_t[d];
        }
      }
    }

    // O projection
    lut_linear(o_proj, nullptr, attn_output.data(), output, 1);
  }
};

} // namespace lutmac
