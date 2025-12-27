/**
 * LutMac: Model Implementation
 *
 * Forward pass and model loading.
 */

#include "lutmac/model.hpp"
#include "lutmac/format.hpp"
#include "lutmac/lut_gemm.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

namespace lutmac {

// External layer functions
// External layer functions
void rms_norm(const float *input, const float *weight, float *output,
              size_t hidden_size, float eps);
void gemma_rms_norm(const float *input, const float *weight, float *output,
                    size_t hidden_size, float eps);

void apply_rope(float *q, float *k, size_t position, size_t num_heads,
                size_t num_kv_heads, size_t head_dim, float theta,
                float factor);

void softmax(float *data, size_t n);

void embed_tokens(const float *embedding_table, const int *token_ids,
                  float *output, size_t num_tokens, size_t hidden_size);

// Activation functions
// Activation functions
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

inline float gelu(float x) {
  return 0.5f * x *
         (1.0f +
          std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

// Helper for padding
inline size_t pad_to_block(size_t n) { return (n + 255) / 256 * 256; }

void Model::forward(int token_id, size_t position) {
  const size_t h = config.hidden_size;

  const size_t num_heads = config.num_attention_heads;
  const size_t num_kv_heads = config.num_key_value_heads;
  const size_t head_dim = config.head_dim();
  const size_t intermediate = config.intermediate_size;

  // Token embedding
  // Dequantize embedding for current token
  // If raw: copy. If quantized: unpack.
  if (embed_tokens.is_quantized && embed_tokens.quant_bits == 4 &&
      !embed_tokens.int4_blocks.empty()) {
    // Int4 quantized embeddings - dequantize on-the-fly
    size_t blocks_per_row =
        (h + Int4Block::BLOCK_SIZE - 1) / Int4Block::BLOCK_SIZE;
    size_t row_start_block = token_id * blocks_per_row;

    for (size_t i = 0; i < h; ++i) {
      size_t block_offset = i / Int4Block::BLOCK_SIZE;
      size_t local_idx = i % Int4Block::BLOCK_SIZE;
      size_t block_idx = row_start_block + block_offset;

      if (block_idx < embed_tokens.int4_blocks.size()) {
        const auto &block = embed_tokens.int4_blocks[block_idx];

        // Extract 4-bit value
        size_t byte_idx = local_idx / 2;
        bool is_high = (local_idx % 2) == 0;
        uint8_t nibble = is_high ? (block.data[byte_idx] >> 4)
                                 : (block.data[byte_idx] & 0x0F);
        int8_t signed_val = static_cast<int8_t>(nibble) - 8;

        hidden_states[i] = signed_val * block.scale;
      } else {
        hidden_states[i] = 0.0f;
      }
    }
  }

  else if (embed_tokens.is_quantized && embed_tokens.quant_bits == 8 &&
           !embed_tokens.int8_blocks.empty()) {
    // Int8 quantized embeddings - dequantize
    size_t blocks_per_row =
        (h + Int8Block::BLOCK_SIZE - 1) / Int8Block::BLOCK_SIZE;
    size_t row_start_block = token_id * blocks_per_row;

    for (size_t i = 0; i < h; ++i) {
      size_t block_offset = i / Int8Block::BLOCK_SIZE;
      size_t local_idx = i % Int8Block::BLOCK_SIZE;
      size_t block_idx = row_start_block + block_offset;

      if (block_idx < embed_tokens.int8_blocks.size()) {
        const auto &block = embed_tokens.int8_blocks[block_idx];
        hidden_states[i] = block.data[local_idx] * block.scale;
      } else {
        hidden_states[i] = 0.0f;
      }
    }
  } else if (embed_tokens.is_quantized && embed_tokens.quant_bits == 1 &&
             !embed_tokens.binary_blocks.empty()) {
    // Binary (1-bit) quantized embeddings
    size_t blocks_per_row =
        (h + BinaryBlock::BLOCK_SIZE - 1) / BinaryBlock::BLOCK_SIZE;
    size_t row_start_block = token_id * blocks_per_row;

    for (size_t i = 0; i < h; ++i) {
      size_t block_offset = i / BinaryBlock::BLOCK_SIZE;
      size_t local_idx = i % BinaryBlock::BLOCK_SIZE;
      size_t block_idx = row_start_block + block_offset;

      if (block_idx < embed_tokens.binary_blocks.size()) {
        const auto &block = embed_tokens.binary_blocks[block_idx];

        // Extract 1-bit value
        size_t byte_idx = local_idx / 8;
        size_t bit_idx = local_idx % 8;
        bool bit = (block.data[byte_idx] >> bit_idx) & 1;

        // 1 -> scale, 0 -> -scale
        hidden_states[i] = bit ? block.scale : -block.scale;
      } else {
        hidden_states[i] = 0.0f;
      }
    }
  } else if (embed_tokens.is_quantized && embed_tokens.quant_bits == 2 &&
             !embed_tokens.int2_blocks.empty()) {
    // 2-bit quantized embeddings
    size_t blocks_per_row = (h + 255) / 256;
    size_t row_start_block = token_id * blocks_per_row;
    for (size_t i = 0; i < h; ++i) {
      size_t b_off = i / 256;
      size_t l_idx = i % 256;
      size_t b_idx = row_start_block + b_off;
      if (b_idx < embed_tokens.int2_blocks.size()) {
        hidden_states[i] = embed_tokens.int2_blocks[b_idx].get(l_idx);
      } else {
        hidden_states[i] = 0.0f;
      }
    }
  } else if (embed_tokens.is_quantized && embed_tokens.quant_bits == 3 &&
             !embed_tokens.int3_blocks.empty()) {
    // 3-bit quantized embeddings
    size_t blocks_per_row = (h + 255) / 256;
    size_t row_start_block = token_id * blocks_per_row;
    for (size_t i = 0; i < h; ++i) {
      size_t b_off = i / 256;
      size_t l_idx = i % 256;
      size_t b_idx = row_start_block + b_off;
      if (b_idx < embed_tokens.int3_blocks.size()) {
        hidden_states[i] = embed_tokens.int3_blocks[b_idx].get(l_idx);
      } else {
        hidden_states[i] = 0.0f;
      }
    }
  } else if (embed_tokens.is_quantized) {
    size_t start_idx = token_id * h;

    if (embed_tokens.quant_bits == 8 && !embed_tokens.int8_blocks.empty()) {
      // Int8 Embeddings
      for (size_t i = 0; i < h; ++i) {
        size_t global_idx = start_idx + i;
        size_t b_idx = global_idx / Int8Block::BLOCK_SIZE;
        size_t l_idx = global_idx % Int8Block::BLOCK_SIZE;
        if (b_idx < embed_tokens.int8_blocks.size()) {
          hidden_states[i] = embed_tokens.int8_blocks[b_idx].get(l_idx);
        } else {
          hidden_states[i] = 0.0f;
        }
      }
    } else if (embed_tokens.quant_bits == 4 &&
               !embed_tokens.int4_blocks.empty()) {
      // Int4 Embeddings
      for (size_t i = 0; i < h; ++i) {
        size_t global_idx = start_idx + i;
        size_t b_idx = global_idx / Int4Block::BLOCK_SIZE;
        size_t l_idx = global_idx % Int4Block::BLOCK_SIZE;
        if (b_idx < embed_tokens.int4_blocks.size()) {
          hidden_states[i] = embed_tokens.int4_blocks[b_idx].get(l_idx);
        } else {
          hidden_states[i] = 0.0f;
        }
      }
    } else if (embed_tokens.quant_bits == 2 &&
               !embed_tokens.int2_blocks.empty()) {
      // Int2 Embeddings
      for (size_t i = 0; i < h; ++i) {
        size_t global_idx = start_idx + i;
        size_t b_idx = global_idx / Int2Block::BLOCK_SIZE;
        size_t l_idx = global_idx % Int2Block::BLOCK_SIZE;
        if (b_idx < embed_tokens.int2_blocks.size()) {
          hidden_states[i] = embed_tokens.int2_blocks[b_idx].get(l_idx);
        } else {
          hidden_states[i] = 0.0f;
        }
      }
    } else if (embed_tokens.quant_bits == 1 &&
               !embed_tokens.binary_blocks.empty()) {
      // Binary Embeddings
      for (size_t i = 0; i < h; ++i) {
        size_t global_idx = start_idx + i;
        size_t b_idx = global_idx / BinaryBlock::BLOCK_SIZE;
        size_t l_idx = global_idx % BinaryBlock::BLOCK_SIZE;
        if (b_idx < embed_tokens.binary_blocks.size()) {
          hidden_states[i] = embed_tokens.binary_blocks[b_idx].get(l_idx);
        } else {
          hidden_states[i] = 0.0f;
        }
      }
    } else {
      // Default: Ternary (BitPlaneBlock)
      // Structure: [vocab_size, hidden_size]
      for (size_t i = 0; i < h; ++i) {
        size_t global_idx = start_idx + i;
        size_t block_idx = global_idx / BitPlaneBlock::BLOCK_SIZE;
        size_t local_idx = global_idx % BitPlaneBlock::BLOCK_SIZE;

        if (block_idx < embed_tokens.blocks.size()) {
          const auto &block = embed_tokens.blocks[block_idx];
          // Extract ternary value (Assuming BitPlaneBlock logic is inline)
          // We can't verify if BitPlaneBlock has get() helper easily (it wasn't
          // in previous view) So we copy the existing extraction logic.
          int byte_idx = local_idx / 8;
          int bit_idx = 7 - (local_idx % 8);

          bool sign = (block.sign_plane[byte_idx] >> bit_idx) & 1;
          bool zero = (block.zero_plane[byte_idx] >> bit_idx) & 1;

          float val = 0.0f;
          if (!zero) {
            val = sign ? -1.0f : 1.0f;
            val *= block.scale;
          }
          hidden_states[i] = val;
        } else {
          hidden_states[i] = 0.0f;
        }
      }
    }
  } else {
    // Raw FP32
    // embed_tokens.raw_data is [vocab, h].
    // We copy row `token_id` to `hidden_states`.
    const float *embed_ptr = embed_tokens.raw_data.data() + token_id * h;
    std::memcpy(hidden_states.data(), embed_ptr, h * sizeof(float));
  }

  // Scale embeddings for Gemma
  if (config.model_type == "gemma") {
    float scale = std::sqrt(static_cast<float>(config.hidden_size));
    for (size_t i = 0; i < h; ++i) {
      hidden_states[i] *= scale;
    }
  }

  // Scaling is now handled via rope_scaling_factor in config.json

  if (config.hidden_size == 4096) {
    // Llama 2/3 usually 7B has 4096 hidden. Consider adjusting theta.
  }

  // Debug: Start
  // std::cerr << "DEBUG: Start Forward token=" << token_id << " pos=" <<
  // position << "\n";

  // Process each layer
  // Pre-allocate all scratch buffers OUTSIDE the loop to avoid malloc overhead
  AlignedVector<float> normed(pad_to_block(h));
  AlignedVector<float> q_buf(pad_to_block(num_heads * head_dim));
  AlignedVector<float> k_buf(pad_to_block(num_kv_heads * head_dim));
  AlignedVector<float> v_buf(pad_to_block(num_kv_heads * head_dim));
  // attn_output must hold concatenated head outputs (can be > hidden_size)
  size_t attn_dim = num_heads * head_dim;
  AlignedVector<float> attn_output(attn_dim, 0.0f);
  AlignedVector<float> attn_weights(config.max_position_embeddings);
  AlignedVector<float> attn_output_padded(pad_to_block(attn_dim));
  AlignedVector<float> gate_buf(pad_to_block(intermediate));
  AlignedVector<float> up_buf(pad_to_block(intermediate));
  AlignedVector<float> inter_buf(pad_to_block(intermediate));

  // Debug: Check hidden states after embedding (commented out)
  // static bool embed_debug = false;
  // if (!embed_debug) { ... }

  for (size_t i = 0; i < config.num_hidden_layers; ++i) {
    TransformerLayer &layer = layers[i];

    // 1. Input Norm
    if (config.model_type == "gemma") {
      gemma_rms_norm(hidden_states.data(), layer.input_layernorm_weight.data(),
                     normed.data(), h, config.rms_norm_eps);
    } else {
      rms_norm(hidden_states.data(), layer.input_layernorm_weight.data(),
               normed.data(), h, config.rms_norm_eps);
    }
    std::fill(normed.begin() + h, normed.end(), 0.0f);

    // --- Attention Layer ---

    // QKV Projections
    lut_linear(layer.q_proj, nullptr, normed.data(), q_buf.data(), 1);
    lut_linear(layer.k_proj, nullptr, normed.data(), k_buf.data(), 1);
    lut_linear(layer.v_proj, nullptr, normed.data(), v_buf.data(), 1);

    // Add biases if present (Qwen2.5)
    if (!layer.q_bias.empty()) {
      for (size_t j = 0; j < num_heads * head_dim; ++j)
        q_buf[j] += layer.q_bias[j];
    }
    if (!layer.k_bias.empty()) {
      for (size_t j = 0; j < num_kv_heads * head_dim; ++j)
        k_buf[j] += layer.k_bias[j];
    }
    if (!layer.v_bias.empty()) {
      for (size_t j = 0; j < num_kv_heads * head_dim; ++j)
        v_buf[j] += layer.v_bias[j];
    }

    // RoPE
    apply_rope(q_buf.data(), k_buf.data(), position, num_heads, num_kv_heads,
               head_dim, config.rope_theta, config.rope_scaling_factor);

    // KV Cache Update
    float *k_cache = kv_cache.key_cache(layer.layer_idx);
    float *v_cache = kv_cache.value_cache(layer.layer_idx);
    std::memcpy(k_cache + position * num_kv_heads * head_dim, k_buf.data(),
                num_kv_heads * head_dim * sizeof(float));
    std::memcpy(v_cache + position * num_kv_heads * head_dim, v_buf.data(),
                num_kv_heads * head_dim * sizeof(float));

    // Compute Attention (Standard)
    size_t seq_len = position + 1;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    size_t kv_group_size = num_heads / num_kv_heads;

    std::fill(attn_output.begin(), attn_output.end(), 0.0f);

    for (size_t head = 0; head < num_heads; ++head) {
      size_t kv_head = head / kv_group_size;
      const float *q_head = q_buf.data() + head * head_dim;

      // Score
      for (size_t t = 0; t < seq_len; ++t) {
        const float *k_t =
            k_cache + t * num_kv_heads * head_dim + kv_head * head_dim;
        float score = 0.0f;
        // Vectorize this dot product? Compiler should handle OD=64/128
        for (size_t d = 0; d < head_dim; ++d) {
          score += q_head[d] * k_t[d];
        }
        attn_weights[t] = score * scale;
      }

      softmax(attn_weights.data(), seq_len);

      // Value agg
      float *out_head = attn_output.data() + head * head_dim;
      for (size_t t = 0; t < seq_len; ++t) {
        const float *v_t =
            v_cache + t * num_kv_heads * head_dim + kv_head * head_dim;
        float weight = attn_weights[t];
        for (size_t d = 0; d < head_dim; ++d) {
          out_head[d] += weight * v_t[d];
        }
      }
    }

    // Output Projection
    std::memcpy(attn_output_padded.data(), attn_output.data(),
                attn_dim * sizeof(float));
    std::fill(attn_output_padded.begin() + attn_dim, attn_output_padded.end(),
              0.0f);

    lut_linear(layer.o_proj, nullptr, attn_output_padded.data(),
               layer.attn_output.data(), 1);

    // Residual Connection 1 (Mixer -> Residual)
    // Residual Connection 1
    for (size_t j = 0; j < h; ++j) {
      hidden_states[j] += layer.attn_output[j];
    }

    // Post-Attention / Pre-FFN Norm
    if (config.model_type == "gemma") {
      gemma_rms_norm(hidden_states.data(),
                     layer.post_attention_layernorm_weight.data(),
                     normed.data(), h, config.rms_norm_eps);
    } else {
      rms_norm(hidden_states.data(),
               layer.post_attention_layernorm_weight.data(), normed.data(), h,
               config.rms_norm_eps);
    }
    std::fill(normed.begin() + h, normed.end(), 0.0f);

    // FFN (Common for both types in LFM2)
    lut_linear(layer.gate_proj, nullptr, normed.data(), gate_buf.data(), 1);
    lut_linear(layer.up_proj, nullptr, normed.data(), up_buf.data(), 1);

    // Debug: Check if FFN weights are loaded (commented out)
    // static bool ffn_debug = false;
    // if (!ffn_debug && i == 0) { ... }

    if (config.activation == ActivationType::GELU) {
      for (size_t j = 0; j < intermediate; ++j) {
        inter_buf[j] = gelu(gate_buf[j]) * up_buf[j];
      }
    } else { // SiLU (Default)
      for (size_t j = 0; j < intermediate; ++j) {
        inter_buf[j] = silu(gate_buf[j]) * up_buf[j];
      }
    }

    lut_linear(layer.down_proj, nullptr, inter_buf.data(),
               layer.ffn_output.data(), 1);

    // Residual Connection 2 (FFN -> Residual)
    // Residual Connection 2 (FFN -> Residual)
    for (size_t j = 0; j < h; ++j) {
      hidden_states[j] += layer.ffn_output[j];
    }

    // Debug: Print hidden states after some layers (commented out)
    // static bool layer_debug = false;
    // if (!layer_debug && i == 0) { ... }
    // static bool last_layer_debug = false;
    // if (!last_layer_debug && i == config.num_hidden_layers - 1) { ... }
  }

  // Final norm
  AlignedVector<float> final_normed(pad_to_block(h));
  if (config.model_type == "gemma") {
    gemma_rms_norm(hidden_states.data(), norm_weight.data(),
                   final_normed.data(), h, config.rms_norm_eps);
  } else {
    rms_norm(hidden_states.data(), norm_weight.data(), final_normed.data(), h,
             config.rms_norm_eps);
  }
  std::fill(final_normed.begin() + h, final_normed.end(), 0.0f);

  // LM head
  lut_linear(lm_head, nullptr, final_normed.data(), logits.data(), 1);

  // Debug: Print first 5 logits after one forward pass (commented out)
  // static bool logit_debug_once = false;
  // if (!logit_debug_once) { ... }
  // // fprintf(stderr, "DEBUG: Head Done\n");
}

int Model::sample(const GenerationConfig &gen_config) {
  const size_t v = config.vocab_size;

  // Apply repetition penalty (before temperature)
  if (gen_config.repetition_penalty != 1.0f && !generated_tokens.empty()) {
    for (int token : generated_tokens) {
      if (token >= 0 && static_cast<size_t>(token) < v) {
        if (logits[token] > 0) {
          logits[token] /= gen_config.repetition_penalty;
        } else {
          logits[token] *= gen_config.repetition_penalty;
        }
      }
    }
  }

  // Apply temperature
  if (gen_config.temperature != 1.0f && gen_config.temperature > 0.0f) {
    float inv_temp = 1.0f / gen_config.temperature;
    for (size_t i = 0; i < v; ++i) {
      logits[i] *= inv_temp;
    }
  }

  // Greedy sampling for now
  if (gen_config.temperature <= 0.0f || gen_config.top_k == 1) {
    int best = 0;
    float best_val = logits[0];
    for (size_t i = 1; i < v; ++i) {
      if (logits[i] > best_val) {
        best_val = logits[i];
        best = static_cast<int>(i);
      }
    }
    return best;
  }

  // Softmax
  float max_logit = *std::max_element(logits.begin(), logits.end());
  float sum = 0.0f;
  for (size_t i = 0; i < v; ++i) {
    logits[i] = std::exp(logits[i] - max_logit);
    sum += logits[i];
  }
  for (size_t i = 0; i < v; ++i) {
    logits[i] /= sum;
  }

  // Min-P sampling: Remove tokens with prob < min_p * max_prob
  if (gen_config.min_p > 0.0f) {
    float max_prob = *std::max_element(logits.begin(), logits.end());
    float threshold = gen_config.min_p * max_prob;
    for (size_t i = 0; i < v; ++i) {
      if (logits[i] < threshold) {
        logits[i] = 0.0f;
      }
    }
    // Renormalize after min_p filtering
    sum = 0.0f;
    for (size_t i = 0; i < v; ++i) {
      sum += logits[i];
    }
    if (sum > 0.0f) {
      for (size_t i = 0; i < v; ++i) {
        logits[i] /= sum;
      }
    }
  }

  // Top-p sampling
  if (gen_config.top_p < 1.0f) {
    // Sort indices by probability
    std::vector<std::pair<float, int>> probs(v);
    for (size_t i = 0; i < v; ++i) {
      probs[i] = {logits[i], static_cast<int>(i)};
    }
    std::sort(probs.begin(), probs.end(), std::greater<>());

    // Find cutoff
    float cumsum = 0.0f;
    size_t cutoff = v;
    for (size_t i = 0; i < v; ++i) {
      cumsum += probs[i].first;
      if (cumsum >= gen_config.top_p) {
        cutoff = i + 1;
        break;
      }
    }

    // Zero out low probability tokens
    for (size_t i = cutoff; i < v; ++i) {
      logits[probs[i].second] = 0.0f;
    }

    // Renormalize
    sum = 0.0f;
    for (size_t i = 0; i < v; ++i) {
      sum += logits[i];
    }
    for (size_t i = 0; i < v; ++i) {
      logits[i] /= sum;
    }
  }

  // Sample
  static std::mt19937 rng(gen_config.seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float r = dist(rng);

  float cumsum = 0.0f;
  for (size_t i = 0; i < v; ++i) {
    cumsum += logits[i];
    if (r < cumsum) {
      return static_cast<int>(i);
    }
  }

  return static_cast<int>(v - 1);
}

// Load model from file
std::unique_ptr<Model> load_model(const std::string &path) {
  ModelConfig config;
  std::vector<PackedTensor> tensors;

  if (!load_lutmac(path, config, tensors)) {
    return nullptr;
  }

  auto model = std::make_unique<Model>();
  model->config = config;
  model->allocate();

  // Map tensors
  for (auto &t : tensors) {
    // Tensor loading (verbose debug removed for clean output)

    if (t.name == "embed_tokens" || t.name == "token_embeddings.weight" ||
        t.name == "model.embed_tokens.weight") {
      // Embeddings can now be quantized or raw
      model->embed_tokens = std::move(t);
    } else if (t.name == "norm.weight" || t.name == "model.norm.weight") {
      if (!t.is_quantized) {
        if (t.raw_data.size() == model->norm_weight.size()) {
          std::copy(t.raw_data.begin(), t.raw_data.end(),
                    model->norm_weight.begin());
        }
      }

    } else if (t.name == "lm_head.weight" || t.name == "output.weight") {
      model->lm_head = std::move(t);
    } else if (t.name.find("layers.") != std::string::npos) {
      // Find layer index
      size_t start = t.name.find("layers.") + 7;
      size_t end = t.name.find(".", start);
      if (end != std::string::npos) {
        int layer_idx = std::stoi(t.name.substr(start, end - start));
        if (layer_idx >= 0 && layer_idx < model->layers.size()) {
          auto &layer = model->layers[layer_idx];

          if (t.name.find("input_layernorm.weight") != std::string::npos) {
            if (!t.is_quantized &&
                t.raw_data.size() == layer.input_layernorm_weight.size())
              std::copy(t.raw_data.begin(), t.raw_data.end(),
                        layer.input_layernorm_weight.begin());
          } else if (t.name.find("post_attention_layernorm.weight") !=
                     std::string::npos) {
            if (!t.is_quantized &&
                t.raw_data.size() ==
                    layer.post_attention_layernorm_weight.size())
              std::copy(t.raw_data.begin(), t.raw_data.end(),
                        layer.post_attention_layernorm_weight.begin());
          } else if (t.name.find("self_attn.q_proj.weight") !=
                     std::string::npos) {
            layer.q_proj = std::move(t);
          } else if (t.name.find("self_attn.k_proj.weight") !=
                     std::string::npos) {
            layer.k_proj = std::move(t);
          } else if (t.name.find("self_attn.v_proj.weight") !=
                     std::string::npos) {
            layer.v_proj = std::move(t);
          } else if (t.name.find("self_attn.o_proj.weight") !=
                     std::string::npos) {
            layer.o_proj = std::move(t);
          } else if (t.name.find("self_attn.q_proj.bias") !=
                     std::string::npos) {
            if (!t.is_quantized && t.raw_data.size() == layer.q_bias.size())
              std::copy(t.raw_data.begin(), t.raw_data.end(),
                        layer.q_bias.begin());
          } else if (t.name.find("self_attn.k_proj.bias") !=
                     std::string::npos) {
            if (!t.is_quantized && t.raw_data.size() == layer.k_bias.size())
              std::copy(t.raw_data.begin(), t.raw_data.end(),
                        layer.k_bias.begin());
          } else if (t.name.find("self_attn.v_proj.bias") !=
                     std::string::npos) {
            if (!t.is_quantized && t.raw_data.size() == layer.v_bias.size())
              std::copy(t.raw_data.begin(), t.raw_data.end(),
                        layer.v_bias.begin());
          } else if (t.name.find("mlp.gate_proj.weight") != std::string::npos) {
            layer.gate_proj = std::move(t);
          } else if (t.name.find("mlp.up_proj.weight") != std::string::npos) {
            layer.up_proj = std::move(t);
          } else if (t.name.find("mlp.down_proj.weight") != std::string::npos) {
            layer.down_proj = std::move(t);
          }
        }
      }
    }
  }

  // Verify critical weights
  if (model->lm_head.raw_data.empty() && model->lm_head.blocks.empty() &&
      model->lm_head.int4_blocks.empty() &&
      model->lm_head.int8_blocks.empty() &&
      model->lm_head.int2_blocks.empty() &&
      model->lm_head.int3_blocks.empty() &&
      model->lm_head.binary_blocks.empty() &&
      model->embed_tokens.num_elements() > 0) {
    std::cout << "Note: lm_head not found, using tied embed_tokens"
              << std::endl;
    // Deep copy (since PackedTensor owns vectors)
    model->lm_head.name = "lm_head";
    model->lm_head.shape = model->embed_tokens.shape;
    model->lm_head.is_quantized = model->embed_tokens.is_quantized;
    model->lm_head.quant_bits = model->embed_tokens.quant_bits;
    model->lm_head.raw_data = model->embed_tokens.raw_data;
    model->lm_head.blocks = model->embed_tokens.blocks;
    model->lm_head.int2_blocks = model->embed_tokens.int2_blocks;
    model->lm_head.int3_blocks = model->embed_tokens.int3_blocks;
    model->lm_head.int4_blocks = model->embed_tokens.int4_blocks;
    model->lm_head.int8_blocks = model->embed_tokens.int8_blocks;
    model->lm_head.binary_blocks = model->embed_tokens.binary_blocks;
    model->lm_head.global_scale = model->embed_tokens.global_scale;
  }

  if (model->embed_tokens.num_elements() == 0) {
    std::cerr << "Warning: Embeddings look empty!\n";
  }
  if (model->norm_weight[0] == 0.0f) {
    std::cerr << "Warning: Final norm weight looks empty/zero!\n";
  }

  return model;
}

} // namespace lutmac
