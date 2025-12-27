#pragma once

/**
 * LutMac: Model Abstraction
 *
 * Full transformer model with LUT-based layers.
 */

#include "lut_gemm.hpp"
#include "quantize.hpp"
#include "types.hpp"
#include <memory>
#include <unordered_map>

namespace lutmac {

// Forward declarations from layers
struct AttentionLayer;
struct FFNLayer;

/**
 * Transformer decoder layer
 */
struct TransformerLayer {
  size_t layer_idx;

  // Normalization weights
  AlignedVector<float> input_layernorm_weight;
  AlignedVector<float> post_attention_layernorm_weight;

  // Attention sublayer
  PackedTensor q_proj;
  PackedTensor k_proj;
  PackedTensor v_proj;
  PackedTensor o_proj;

  // Q/K/V biases (Qwen2.5 uses these)
  AlignedVector<float> q_bias;
  AlignedVector<float> k_bias;
  AlignedVector<float> v_bias;

  // FFN sublayer
  PackedTensor gate_proj;
  PackedTensor up_proj;
  PackedTensor down_proj;

  // Scratch buffers
  AlignedVector<float> attn_output;
  AlignedVector<float> ffn_output;
  AlignedVector<float> residual;
};

/**
 * Full transformer model
 */
class Model {
public:
  ModelConfig config;

  // Embedding
  // AlignedVector<float> embed_tokens; // [vocab_size, hidden_size]
  PackedTensor embed_tokens;

  // Layers
  std::vector<TransformerLayer> layers;

  // Final norm
  AlignedVector<float> norm_weight;

  // LM head (output projection)
  PackedTensor lm_head; // [vocab_size, hidden_size]

  // KV cache
  KVCache kv_cache;

  // Scratch buffers
  AlignedVector<float> hidden_states;
  AlignedVector<float> logits;

  // Generated tokens for repetition penalty
  std::vector<int> generated_tokens;

  Model() = default;

  /**
   * Allocate all buffers based on config
   */
  void allocate() {
    size_t h = config.hidden_size;
    size_t v = config.vocab_size;
    size_t n = config.num_hidden_layers;

    // embed_tokens is loaded from file, size set there
    // embed_tokens.resize(v * h);
    norm_weight.assign(h, 1.0f); // Default to identity (1.0) if not loaded
    hidden_states.resize(h);
    logits.resize(v);

    layers.resize(n);
    for (size_t i = 0; i < n; ++i) {
      TransformerLayer &layer = layers[i];
      layer.layer_idx = i;
      layer.input_layernorm_weight.assign(h, 1.0f); // Default to identity
      layer.post_attention_layernorm_weight.assign(h,
                                                   1.0f); // Default to identity
      layer.attn_output.resize(align_up(h, 256));
      layer.ffn_output.resize(align_up(h, 256));
      layer.residual.resize(h);

      // Q/K/V biases for Qwen2.5-style models
      layer.q_bias.resize(config.num_attention_heads * config.head_dim());
      layer.k_bias.resize(config.num_key_value_heads * config.head_dim());
      layer.v_bias.resize(config.num_key_value_heads * config.head_dim());
    }

    // Allocate KV cache
    kv_cache.allocate(config, config.max_position_embeddings);
  }

  /**
   * Forward pass for a single token
   */
  void forward(int token_id, size_t position);

  /**
   * Get output logits
   */
  const float *get_logits() const { return logits.data(); }

  /**
   * Sample next token
   */
  int sample(const GenerationConfig &gen_config);

  /**
   * Reset KV cache
   */
  void reset_cache() {
    kv_cache.current_pos = 0;
    std::fill(kv_cache.cache.begin(), kv_cache.cache.end(), 0.0f);
  }
};

/**
 * Load model from .lutmac file
 */
std::unique_ptr<Model> load_model(const std::string &path);

/**
 * Load model from safetensors and quantize on-the-fly
 */
std::unique_ptr<Model> load_and_quantize(const std::string &safetensors_path,
                                         const std::string &config_path);

} // namespace lutmac
