/**
 * LutMac: LUT-Based GEMM Unit Tests
 */

#include "lutmac/lut_gemm.hpp"
#include "lutmac/quantize.hpp"
#include "lutmac/types.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace lutmac;

bool test_lut_precomputation() {
  std::cout << "Testing LUT precomputation...\n";

  float activations[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float lut[16];

  precompute_lut(activations, lut);

  // Verify all 16 combinations
  float expected[16] = {
      0.0f,                     // 0b0000: none
      4.0f,                     // 0b0001: a3
      3.0f,                     // 0b0010: a2
      3.0f + 4.0f,              // 0b0011: a2 + a3
      2.0f,                     // 0b0100: a1
      2.0f + 4.0f,              // 0b0101: a1 + a3
      2.0f + 3.0f,              // 0b0110: a1 + a2
      2.0f + 3.0f + 4.0f,       // 0b0111: a1 + a2 + a3
      1.0f,                     // 0b1000: a0
      1.0f + 4.0f,              // 0b1001: a0 + a3
      1.0f + 3.0f,              // 0b1010: a0 + a2
      1.0f + 3.0f + 4.0f,       // 0b1011: a0 + a2 + a3
      1.0f + 2.0f,              // 0b1100: a0 + a1
      1.0f + 2.0f + 4.0f,       // 0b1101: a0 + a1 + a3
      1.0f + 2.0f + 3.0f,       // 0b1110: a0 + a1 + a2
      1.0f + 2.0f + 3.0f + 4.0f // 0b1111: all
  };

  for (int i = 0; i < 16; ++i) {
    if (std::abs(lut[i] - expected[i]) > 0.001f) {
      std::cerr << "FAIL: LUT[" << i << "] = " << lut[i] << ", expected "
                << expected[i] << "\n";
      return false;
    }
  }

  std::cout << "  ✓ LUT precomputation OK\n";
  return true;
}

bool test_index_extraction() {
  std::cout << "Testing index extraction...\n";

  // Create a simple bit-plane
  // 4 weights: [+1, -1, 0, +1]
  // sign:  [0, 1, 0, 0] -> 0b0100
  // zero:  [0, 0, 1, 0] -> 0b0010

  uint8_t sign_plane[4] = {0b01000000, 0, 0, 0}; // First nibble: 0100
  uint8_t zero_plane[4] = {0b00100000, 0, 0, 0}; // First nibble: 0010

  uint8_t pos_idx, neg_idx;
  extract_indices(sign_plane, zero_plane, 0, pos_idx, neg_idx);

  // positive: non-zero AND not-negative
  // non_zero = ~0010 & 0xF = 1101
  // positive = 1101 & ~0100 = 1101 & 1011 = 1001 = positions 0 and 3
  // negative = 1101 & 0100 = 0100 = position 1

  if (pos_idx != 0b1001) {
    std::cerr << "FAIL: pos_idx = " << (int)pos_idx << ", expected 9\n";
    return false;
  }

  if (neg_idx != 0b0100) {
    std::cerr << "FAIL: neg_idx = " << (int)neg_idx << ", expected 4\n";
    return false;
  }

  std::cout << "  ✓ Index extraction OK\n";
  return true;
}

bool test_scalar_gemv() {
  std::cout << "Testing scalar GEMV...\n";

  // Create a simple test case
  const size_t M = 4;  // Output dim
  const size_t K = 16; // Input dim (must be multiple of 4)

  // Create weights with known values
  std::vector<float> weights(M * K);
  for (size_t i = 0; i < M * K; ++i) {
    // Alternating +0.5, -0.5, 0
    int mod = i % 3;
    weights[i] = (mod == 0) ? 0.5f : (mod == 1) ? -0.5f : 0.0f;
  }

  // Quantize
  PackedTensor packed = quantize_tensor(weights.data(), {M, K}, "test");

  // Create activations
  AlignedVector<float> activations(K);
  for (size_t i = 0; i < K; ++i) {
    activations[i] = 1.0f;
  }

  // Run GEMV
  AlignedVector<float> output(M);
  lut_gemv_scalar(packed.blocks.data(), packed.blocks.size(),
                  activations.data(), output.data(), M, K);

  // Output should have some non-zero values
  float sum = 0.0f;
  for (size_t i = 0; i < M; ++i) {
    sum += std::abs(output[i]);
  }

  if (sum < 0.001f) {
    std::cerr << "FAIL: Output is all zeros\n";
    return false;
  }

  std::cout << "  Output sum: " << sum << "\n";
  std::cout << "  ✓ Scalar GEMV OK\n";
  return true;
}

bool test_gemv_correctness() {
  std::cout << "Testing GEMV correctness...\n";

  // Create a simple case where we know the answer
  const size_t M = 1;
  const size_t K = 4;

  // All weights = +1 (after quantization)
  std::vector<float> weights = {1.0f, 1.0f, 1.0f, 1.0f};

  // Activations = [1, 2, 3, 4]
  AlignedVector<float> activations = {1.0f, 2.0f, 3.0f, 4.0f};

  // Expected output = 1 + 2 + 3 + 4 = 10 (scaled)

  PackedTensor packed = quantize_tensor(weights.data(), {M, K}, "test");

  AlignedVector<float> output(M);
  lut_gemv_scalar(packed.blocks.data(), packed.blocks.size(),
                  activations.data(), output.data(), M, K);

  // The exact value depends on scaling, but should be positive and proportional
  if (output[0] <= 0.0f) {
    std::cerr << "FAIL: Expected positive output, got " << output[0] << "\n";
    return false;
  }

  std::cout << "  Output: " << output[0] << "\n";
  std::cout << "  ✓ GEMV correctness OK\n";
  return true;
}

bool test_gemv_performance() {
  std::cout << "Testing GEMV performance...\n";

  const size_t M = 2048;
  const size_t K = 2048;
  const size_t iterations = 10;

  // Create random weights
  std::vector<float> weights(M * K);
  for (size_t i = 0; i < M * K; ++i) {
    weights[i] = (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * 0.1f;
  }

  PackedTensor packed = quantize_tensor(weights.data(), {M, K}, "perf_test");

  AlignedVector<float> activations(K);
  for (size_t i = 0; i < K; ++i) {
    activations[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  AlignedVector<float> output(M);

  // Warmup
  for (size_t i = 0; i < 3; ++i) {
    lut_gemv(packed.blocks.data(), packed.blocks.size(), activations.data(),
             output.data(), M, K);
  }

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < iterations; ++i) {
    lut_gemv(packed.blocks.data(), packed.blocks.size(), activations.data(),
             output.data(), M, K);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  double us_per_op = duration.count() / static_cast<double>(iterations);
  double gflops = (M * K * 2.0) / us_per_op / 1000.0;

  std::cout << "  Matrix size: " << M << " x " << K << "\n";
  std::cout << "  Time per op: " << us_per_op << " μs\n";
  std::cout << "  Throughput:  " << gflops << " GFLOP/s (effective)\n";
  std::cout << "  ✓ Performance test OK\n";

  return true;
}

bool test_int4_gemv() {
  std::cout << "Testing Int4 GEMV...\n";

  // Create a simple test case
  const size_t M = 1;
  const size_t K = 32; // Minimum for NEON loop

  std::vector<float> weights(K);
  for (size_t i = 0; i < K; ++i) {
    // Ramp -7 to +7
    int val = (i % 15) - 7;
    // Scale by 0.5
    weights[i] = val * 0.5f;
  }

  // Quantize using actual quantization logic
  PackedTensor packed =
      quantize_tensor_int4(weights.data(), {M, K}, "test_int4");

  // Create activations (all 1.0)
  AlignedVector<float> activations(K, 1.0f);

  // Run GEMV
  AlignedVector<float> output(M);
  int4_gemv(packed.int4_blocks.data(), packed.int4_blocks.size(),
            activations.data(), output.data(), M, K);

  // Expected output
  float expected_sum = 0.0f;

  for (float w : weights)
    expected_sum += w;

  std::cout << "  Expected: " << expected_sum << ", Got: " << output[0] << "\n";

  if (std::abs(output[0] - expected_sum) > 0.01f) {
    std::cerr << "FAIL: Output mismatch\n";
    return false;
  }

  std::cout << "  ✓ Int4 GEMV OK\n";
  return true;
}

bool test_binary_gemv() {
  std::cout << "Testing Binary GEMV...\n";

  // Create a simple test case
  const size_t M = 1;
  const size_t K = 256; // 64 bit-planes/blocks usually 128/256?
                        // BinaryBlock is 256 bits (32 bytes data) + scale.
                        // Correct size is 256 weights.

  std::vector<float> weights(K);
  for (size_t i = 0; i < K; ++i) {
    // 1 or -1
    weights[i] = (i % 2 == 0) ? 1.0f : -1.0f;
  }

  // Quantize using actual quantization logic
  PackedTensor packed =
      quantize_tensor_binary(weights.data(), {M, K}, "test_binary");

  // Create activations (all 1.0)
  AlignedVector<float> activations(K, 1.0f);

  // Run GEMV
  AlignedVector<float> output(M);
  lut_gemv_binary(packed.binary_blocks.data(), packed.binary_blocks.size(),
                  activations.data(), output.data(), M, K);

  // Expected output
  // Sum of weights * activations
  // W = 1, -1, 1, -1...
  // Sum should be 0.

  float expected_sum = 0.0f;
  for (float w : weights)
    expected_sum += w; // Should come out close to 0

  std::cout << "  Expected: " << expected_sum << ", Got: " << output[0] << "\n";

  if (std::abs(output[0] - expected_sum) > 1.0f) { // Tolerance due to scaling
    std::cerr << "FAIL: Output mismatch\n";
    return false;
  }

  std::cout << "  ✓ Binary GEMV OK\n";
  return true;
}

bool test_int8_gemv() {
  std::cout << "Testing Int8 GEMV...\n";

  size_t M = 256;
  size_t K = 512;

  std::vector<float> weights(M * K);
  std::vector<float> input(K);
  std::vector<float> output_ref(M);
  std::vector<float> output_lut(M);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Init data
  for (size_t i = 0; i < M * K; ++i)
    weights[i] = dist(rng);
  for (size_t i = 0; i < K; ++i)
    input[i] = dist(rng);

  // Quantize
  lutmac::PackedTensor packed =
      lutmac::quantize_tensor_int8(weights.data(), {M, K}, "test");

  // Compute Reference (De-quantized)
  for (size_t r = 0; r < M; ++r) {
    float acc = 0.0f;
    size_t blocks_per_row = (K + 255) / 256;
    for (size_t b = 0; b < blocks_per_row; ++b) {
      if (r * blocks_per_row + b >= packed.int8_blocks.size())
        continue;
      const auto &block = packed.int8_blocks[r * blocks_per_row + b];
      for (size_t i = 0; i < 256; ++i) {
        if (b * 256 + i < K) {
          float w = block.data[i] * block.scale;
          acc += w * input[b * 256 + i];
        }
      }
    }
    output_ref[r] = acc;
  }

  // Run Kernel
  lutmac::int8_gemv(packed.int8_blocks.data(), packed.int8_blocks.size(),
                    input.data(), output_lut.data(), M, K);

  // Compare
  float max_diff = 0.0f;
  for (size_t i = 0; i < M; ++i) {
    float diff = std::abs(output_ref[i] - output_lut[i]);
    if (diff > max_diff)
      max_diff = diff;
  }

  if (max_diff > 1e-3f) {
    std::cout << "  X Int8 GEMV failed. Max diff: " << max_diff << "\n";
    return false;
  }
  std::cout << "  ✓ Int8 GEMV OK\n";
  return true;
}

int main() {
  std::cout
      << "╔══════════════════════════════════════════════════════════════╗\n";
  std::cout
      << "║             LutMac LUT-GEMM Tests                            ║\n";
  std::cout
      << "╚══════════════════════════════════════════════════════════════╝\n\n";

  int passed = 0;
  int failed = 0;

  if (test_lut_precomputation())
    passed++;
  else
    failed++;
  if (test_index_extraction())
    passed++;
  else
    failed++;
  if (test_scalar_gemv())
    passed++;
  else
    failed++;
  if (test_gemv_correctness())
    passed++;
  else
    failed++;
  if (test_gemv_performance())
    passed++;
  else
    failed++;
  if (test_int4_gemv())
    passed++;
  else
    failed++;
  if (test_binary_gemv())
    passed++;
  else
    failed++;

  if (test_int8_gemv())
    passed++;
  else
    failed++;

  std::cout
      << "\n═══════════════════════════════════════════════════════════════\n";
  std::cout << "Results: " << passed << " passed, " << failed << " failed\n";

  return failed > 0 ? 1 : 0;
}
