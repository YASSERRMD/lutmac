/**
 * LutMac: Quantization Unit Tests
 */

#include "lutmac/quantize.hpp"
#include "lutmac/types.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace lutmac;

bool test_ternary_quantization() {
  std::cout << "Testing ternary quantization...\n";

  // Test values
  float values[] = {0.5f, -0.3f, 0.01f, -0.8f, 0.0f};
  float threshold = 0.1f;

  TernaryValue expected[] = {
      TernaryValue::POS_ONE, // 0.5 > 0.1
      TernaryValue::NEG_ONE, // -0.3 < -0.1
      TernaryValue::ZERO,    // |0.01| < 0.1
      TernaryValue::NEG_ONE, // -0.8 < -0.1
      TernaryValue::ZERO     // 0.0 < 0.1
  };

  for (int i = 0; i < 5; ++i) {
    TernaryValue result = quantize_ternary(values[i], threshold);
    if (result != expected[i]) {
      std::cerr << "FAIL: quantize_ternary(" << values[i] << ", " << threshold
                << ")\n";
      std::cerr << "  Expected: " << static_cast<int>(expected[i]) << "\n";
      std::cerr << "  Got:      " << static_cast<int>(result) << "\n";
      return false;
    }
  }

  std::cout << "  ✓ Ternary quantization OK\n";
  return true;
}

bool test_bitplane_packing() {
  std::cout << "Testing bit-plane packing...\n";

  // Create ternary values
  TernaryValue ternary[16] = {
      TernaryValue::POS_ONE, // +1
      TernaryValue::NEG_ONE, // -1
      TernaryValue::ZERO,    // 0
      TernaryValue::POS_ONE, // +1
      TernaryValue::NEG_ONE, // -1
      TernaryValue::ZERO,    // 0
      TernaryValue::POS_ONE, // +1
      TernaryValue::NEG_ONE, // -1
      TernaryValue::ZERO,    // 0
      TernaryValue::POS_ONE, // +1
      TernaryValue::NEG_ONE, // -1
      TernaryValue::ZERO,    // 0
      TernaryValue::POS_ONE, // +1
      TernaryValue::NEG_ONE, // -1
      TernaryValue::ZERO,    // 0
      TernaryValue::POS_ONE  // +1
  };

  uint8_t sign_plane[2];
  uint8_t zero_plane[2];

  pack_to_bitplanes(ternary, 16, sign_plane, zero_plane);

  // Unpack and verify
  TernaryValue unpacked[16];
  unpack_from_bitplanes(sign_plane, zero_plane, 16, unpacked);

  for (int i = 0; i < 16; ++i) {
    if (unpacked[i] != ternary[i]) {
      std::cerr << "FAIL: Bit-plane round-trip at index " << i << "\n";
      std::cerr << "  Expected: " << static_cast<int>(ternary[i]) << "\n";
      std::cerr << "  Got:      " << static_cast<int>(unpacked[i]) << "\n";
      return false;
    }
  }

  std::cout << "  ✓ Bit-plane packing OK\n";
  return true;
}

bool test_tensor_quantization() {
  std::cout << "Testing full tensor quantization...\n";

  // Create random-ish test data
  const size_t n = 512;
  std::vector<float> data(n);
  for (size_t i = 0; i < n; ++i) {
    // Varied values including positive, negative, and near-zero
    data[i] = std::sin(static_cast<float>(i) * 0.1f) * 0.5f;
  }

  // Quantize
  PackedTensor packed = quantize_tensor(data.data(), {n}, "test");

  // Verify structure
  if (packed.shape.size() != 1 || packed.shape[0] != n) {
    std::cerr << "FAIL: Wrong shape\n";
    return false;
  }

  if (packed.blocks.empty()) {
    std::cerr << "FAIL: No blocks created\n";
    return false;
  }

  // Dequantize and check reconstruction error
  std::vector<float> reconstructed = dequantize_tensor(packed);

  if (reconstructed.size() != n) {
    std::cerr << "FAIL: Wrong reconstructed size\n";
    return false;
  }

  // Calculate reconstruction error
  float mse = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float diff = data[i] - reconstructed[i];
    mse += diff * diff;
  }
  mse /= n;

  std::cout << "  Reconstruction MSE: " << mse << "\n";

  // For ternary quantization, expect some error but not too much
  if (mse > 1.0f) {
    std::cerr << "FAIL: MSE too high\n";
    return false;
  }

  std::cout << "  ✓ Tensor quantization OK\n";
  return true;
}

bool test_statistics() {
  std::cout << "Testing statistics computation...\n";

  float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  QuantizationStats stats;
  stats.compute(data, 5);

  // Mean should be 3.0
  if (std::abs(stats.mean - 3.0f) > 0.001f) {
    std::cerr << "FAIL: Wrong mean. Expected 3.0, got " << stats.mean << "\n";
    return false;
  }

  // Std dev should be sqrt(2)
  if (std::abs(stats.std_dev - std::sqrt(2.0f)) > 0.001f) {
    std::cerr << "FAIL: Wrong std dev. Expected " << std::sqrt(2.0f) << ", got "
              << stats.std_dev << "\n";
    return false;
  }

  std::cout << "  ✓ Statistics OK\n";
  return true;
}

int main() {
  std::cout
      << "╔══════════════════════════════════════════════════════════════╗\n";
  std::cout
      << "║             LutMac Quantization Tests                        ║\n";
  std::cout
      << "╚══════════════════════════════════════════════════════════════╝\n\n";

  int passed = 0;
  int failed = 0;

  if (test_ternary_quantization())
    passed++;
  else
    failed++;
  if (test_bitplane_packing())
    passed++;
  else
    failed++;
  if (test_tensor_quantization())
    passed++;
  else
    failed++;
  if (test_statistics())
    passed++;
  else
    failed++;

  std::cout
      << "\n═══════════════════════════════════════════════════════════════\n";
  std::cout << "Results: " << passed << " passed, " << failed << " failed\n";

  return failed > 0 ? 1 : 0;
}
