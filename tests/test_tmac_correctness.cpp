#include "lutmac/lut_gemm.hpp"
#include "lutmac/quantize.hpp"
#include "lutmac/types.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace lutmac;

void test_ternary_correctness() {
  std::cout << "Testing Ternary T-MAC Correctness..." << std::endl;
  const size_t M = 16;
  const size_t K = 256;

  std::vector<float> weights(M * K);
  std::vector<float> activations(K);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (auto &w : weights)
    w = dist(gen);
  for (auto &x : activations)
    x = dist(gen);

  // Quantize using our tools
  PackedTensor packed = quantize_tensor(weights.data(), {M, K});

  // Reference output from dequantized weights
  std::vector<float> ref_output(M, 0.0f);
  for (size_t r = 0; r < M; ++r) {
    for (size_t c = 0; c < K; ++c) {
      size_t b_idx = (r * (K / 256)) + (c / 256);
      size_t l_idx = c % 256;
      const auto &block = packed.blocks[b_idx];

      ref_output[r] += block.get(l_idx) * activations[c];
    }
  }

  std::vector<float> test_output(M, 0.0f);
  lut_gemv(packed.blocks.data(), packed.blocks.size(), activations.data(),
           test_output.data(), M, K);

  // Compare
  float max_err = 0.0f;
  for (size_t i = 0; i < M; ++i) {
    float err = std::abs(test_output[i] - ref_output[i]);
    max_err = std::max(max_err, err);
  }

  std::cout << "  Max Error (Ternary): " << max_err << std::endl;
  assert(max_err < 1e-4);
}

void test_int2_correctness() {
  std::cout << "Testing Int2 T-MAC Correctness..." << std::endl;
  const size_t M = 16;
  const size_t K = 256;

  std::vector<float> weights(M * K);
  std::vector<float> activations(K);
  std::mt19937 gen(43);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &w : weights)
    w = dist(gen);
  for (auto &x : activations)
    x = dist(gen);

  PackedTensor packed = quantize_tensor_int2(weights.data(), {M, K});

  std::vector<float> ref_output(M, 0.0f);
  for (size_t r = 0; r < M; ++r) {
    for (size_t c = 0; c < K; ++c) {
      size_t b_idx = (r * (K / 256)) + (c / 256);
      float val = packed.int2_blocks[b_idx].get(c % 256);
      ref_output[r] += val * activations[c];
    }
  }

  std::vector<float> test_output(M, 0.0f);
  int2_gemv(packed.int2_blocks.data(), packed.int2_blocks.size(),
            activations.data(), test_output.data(), M, K);

  float max_err = 0.0f;
  for (size_t i = 0; i < M; ++i) {
    float err = std::abs(test_output[i] - ref_output[i]);
    max_err = std::max(max_err, err);
  }
  std::cout << "  Max Error (Int2): " << max_err << std::endl;
  assert(max_err < 1e-3);
}

void test_int3_correctness() {
  std::cout << "Testing Int3 T-MAC Correctness..." << std::endl;
  const size_t M = 16;
  const size_t K = 256;

  std::vector<float> weights(M * K);
  std::vector<float> activations(K);
  std::mt19937 gen(44);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &w : weights)
    w = dist(gen);
  for (auto &x : activations)
    x = dist(gen);

  PackedTensor packed = quantize_tensor_int3(weights.data(), {M, K});

  std::vector<float> ref_output(M, 0.0f);
  for (size_t r = 0; r < M; ++r) {
    for (size_t c = 0; c < K; ++c) {
      size_t b_idx = (r * (K / 256)) + (c / 256);
      float val = packed.int3_blocks[b_idx].get(c % 256);
      ref_output[r] += val * activations[c];
    }
  }

  std::vector<float> test_output(M, 0.0f);
  int3_gemv(packed.int3_blocks.data(), packed.int3_blocks.size(),
            activations.data(), test_output.data(), M, K);

  float max_err = 0.0f;
  for (size_t i = 0; i < M; ++i) {
    float err = std::abs(test_output[i] - ref_output[i]);
    max_err = std::max(max_err, err);
  }
  std::cout << "  Max Error (Int3): " << max_err << std::endl;
  if (max_err >= 1e-3) {
    std::cerr << "Int3 Test Failed! Max error: " << max_err << std::endl;
  }
  assert(max_err < 1e-3);
}

int main() {
  try {
    test_ternary_correctness();
    test_int2_correctness();
    test_int3_correctness();
    std::cout << "All Correctness Tests Passed!" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Test failed: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
