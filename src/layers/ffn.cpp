/**
 * LutMac: Feed-Forward Network Layer
 *
 * SwiGLU and GELU activation implementations with LUT-based linear layers.
 */

#include "lutmac/lut_gemm.hpp"
#include "lutmac/types.hpp"
#include <cmath>

namespace lutmac {

/**
 * GELU activation function
 */
inline float gelu(float x) {
  // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  constexpr float sqrt_2_over_pi = 0.7978845608f;
  float x3 = x * x * x;
  float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

/**
 * SiLU (Swish) activation function
 */
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

/**
 * Feed-Forward Network with SwiGLU activation
 *
 * output = down_proj(silu(gate_proj(x)) * up_proj(x))
 */
struct FFNLayer {
  // Weights (quantized)
  PackedTensor gate_proj; // [intermediate_size, hidden_size]
  PackedTensor up_proj;   // [intermediate_size, hidden_size]
  PackedTensor down_proj; // [hidden_size, intermediate_size]

  // Config
  size_t hidden_size;
  size_t intermediate_size;
  ActivationType activation;

  // Scratch buffers
  AlignedVector<float> gate_buf;
  AlignedVector<float> up_buf;
  AlignedVector<float> intermediate_buf;

  void allocate(const ModelConfig &config) {
    hidden_size = config.hidden_size;
    intermediate_size = config.intermediate_size;
    activation = config.activation;

    gate_buf.resize(intermediate_size);
    up_buf.resize(intermediate_size);
    intermediate_buf.resize(intermediate_size);
  }

  void forward(const float *hidden_states, float *output) {
    // Gate projection
    lut_linear(gate_proj, nullptr, hidden_states, gate_buf.data(), 1);

    // Up projection
    lut_linear(up_proj, nullptr, hidden_states, up_buf.data(), 1);

    // Apply activation and element-wise multiply
    switch (activation) {
    case ActivationType::SILU:
    case ActivationType::SWIGLU:
      for (size_t i = 0; i < intermediate_size; ++i) {
        intermediate_buf[i] = silu(gate_buf[i]) * up_buf[i];
      }
      break;

    case ActivationType::GELU:
      for (size_t i = 0; i < intermediate_size; ++i) {
        intermediate_buf[i] = gelu(gate_buf[i]) * up_buf[i];
      }
      break;

    case ActivationType::RELU:
      for (size_t i = 0; i < intermediate_size; ++i) {
        float g = gate_buf[i] > 0.0f ? gate_buf[i] : 0.0f;
        intermediate_buf[i] = g * up_buf[i];
      }
      break;
    }

    // Down projection
    lut_linear(down_proj, nullptr, intermediate_buf.data(), output, 1);
  }
};

} // namespace lutmac
