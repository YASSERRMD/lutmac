/**
 * LutMac Benchmark Tool
 *
 * Performance benchmarking for LUT-based GEMM operations.
 *
 * Usage:
 *   lutmac-bench --model model.lutmac --tokens 100
 */

#include "lutmac/lut_gemm.hpp"
#include "lutmac/quantize.hpp"
#include "lutmac/types.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace lutmac;

void print_usage(const char *prog) {
  std::cout << "LutMac Benchmark Tool v" << LUTMAC_FORMAT_VERSION << "\n\n";
  std::cout << "Usage: " << prog << " [options]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --model, -m <path>   Path to .lutmac model (optional)\n";
  std::cout
      << "  --tokens, -n <n>     Number of tokens to simulate (default: 100)\n";
  std::cout << "  --hidden-size <n>    Hidden dimension (default: 2048)\n";
  std::cout << "  --iterations <n>     Benchmark iterations (default: 10)\n";
  std::cout << "  --warmup <n>         Warmup iterations (default: 3)\n";
  std::cout << "  --help, -h           Show this help\n";
}

struct BenchArgs {
  std::string model_path;
  size_t tokens = 100;
  size_t hidden_size = 2048;
  size_t iterations = 10;
  size_t warmup = 3;
};

bool parse_args(int argc, char **argv, BenchArgs &args) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return false;
    } else if (arg == "--model" || arg == "-m") {
      if (i + 1 < argc)
        args.model_path = argv[++i];
    } else if (arg == "--tokens" || arg == "-n") {
      if (i + 1 < argc)
        args.tokens = std::stoul(argv[++i]);
    } else if (arg == "--hidden-size") {
      if (i + 1 < argc)
        args.hidden_size = std::stoul(argv[++i]);
    } else if (arg == "--iterations") {
      if (i + 1 < argc)
        args.iterations = std::stoul(argv[++i]);
    } else if (arg == "--warmup") {
      if (i + 1 < argc)
        args.warmup = std::stoul(argv[++i]);
    }
  }

  return true;
}

double benchmark_lut_gemv(const BitPlaneBlock *blocks, size_t num_blocks,
                          const float *activations, float *output, size_t M,
                          size_t K, size_t iterations) {
  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < iterations; ++i) {
    lut_gemv(blocks, num_blocks, activations, output, M, K);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  return duration.count() / static_cast<double>(iterations);
}

int main(int argc, char **argv) {
  BenchArgs args;

  if (!parse_args(argc, argv, args)) {
    return 0;
  }

  std::cout
      << "╔══════════════════════════════════════════════════════════════╗\n";
  std::cout
      << "║               LutMac Benchmark Tool v1.0                     ║\n";
  std::cout
      << "║     Bit-Serial LUT Engine for Ultra-Low-Bit Inference        ║\n";
  std::cout
      << "╚══════════════════════════════════════════════════════════════╝\n\n";

  std::cout << "Configuration:\n";
  std::cout << "  Hidden size:  " << args.hidden_size << "\n";
  std::cout << "  Tokens:       " << args.tokens << "\n";
  std::cout << "  Iterations:   " << args.iterations << "\n";
  std::cout << "  Warmup:       " << args.warmup << "\n\n";

  // Detect SIMD support
  std::cout << "SIMD Support:\n";
#if defined(LUTMAC_HAS_AVX512) || defined(LUTMAC_AVX512)
  std::cout << "  [✓] AVX-512\n";
#else
  std::cout << "  [ ] AVX-512\n";
#endif
#if defined(LUTMAC_HAS_AVX2) || defined(LUTMAC_AVX2)
  std::cout << "  [✓] AVX2\n";
#else
  std::cout << "  [ ] AVX2\n";
#endif
#if defined(LUTMAC_HAS_NEON) || defined(LUTMAC_NEON)
  std::cout << "  [✓] ARM NEON\n";
#else
  std::cout << "  [ ] ARM NEON\n";
#endif
  std::cout << "\n";

  // Create test tensors
  size_t M = args.hidden_size; // Output dim
  size_t K = args.hidden_size; // Input dim

  // Generate random weights
  std::vector<float> weights(M * K);
  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] = (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * 0.02f;
  }

  // Quantize
  PackedTensor packed = quantize_tensor(weights.data(), {M, K}, "test");

  // Create activations
  AlignedVector<float> activations(K);
  for (size_t i = 0; i < K; ++i) {
    activations[i] = (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
  }

  AlignedVector<float> output(M);

  // Warmup
  std::cout << "Warming up...\n";
  for (size_t i = 0; i < args.warmup; ++i) {
    lut_gemv(packed.blocks.data(), packed.blocks.size(), activations.data(),
             output.data(), M, K);
  }

  // Benchmark LUT GEMV
  std::cout << "Running benchmark...\n\n";

  double gemv_time_us = benchmark_lut_gemv(
      packed.blocks.data(), packed.blocks.size(), activations.data(),
      output.data(), M, K, args.iterations);

  // Calculate metrics
  size_t total_ops = M * K * 2; // multiply + add per weight
  double gflops = total_ops / gemv_time_us / 1000.0; // GFLOP/s

  size_t weight_bytes = packed.blocks.size() * sizeof(BitPlaneBlock);
  double bandwidth_gb_s = weight_bytes / gemv_time_us / 1000.0;

  // Token throughput estimation
  // Assume 7 GEMV ops per token (Q, K, V, O, gate, up, down)
  double token_time_us = gemv_time_us * 7;
  double tokens_per_sec = 1000000.0 / token_time_us;

  std::cout
      << "═══════════════════════════════════════════════════════════════\n";
  std::cout
      << "                         Results                               \n";
  std::cout
      << "═══════════════════════════════════════════════════════════════\n\n";

  std::cout << std::fixed << std::setprecision(2);

  std::cout << "LUT-Based GEMV (" << M << " x " << K << "):\n";
  std::cout << "  Time per op:      " << gemv_time_us << " μs\n";
  std::cout << "  Throughput:       " << gflops << " GFLOP/s (effective)\n";
  std::cout << "  Memory bandwidth: " << bandwidth_gb_s << " GB/s\n\n";

  std::cout << "Estimated Token Throughput:\n";
  std::cout << "  Time per token:   " << (token_time_us / 1000.0) << " ms\n";
  std::cout << "  Tokens/second:    " << tokens_per_sec << "\n\n";

  std::cout << "Memory Efficiency:\n";
  std::cout << "  Weight size:      " << (weight_bytes / 1024.0) << " KB\n";
  std::cout << "  Bits per weight:  1.58\n";
  std::cout << "  Compression:      " << (32.0 / 1.58) << "x vs FP32\n\n";

  // Verification
  std::cout << "Verification:\n";
  float sum = 0.0f;
  for (size_t i = 0; i < M; ++i) {
    sum += output[i] * output[i];
  }
  std::cout << "  Output L2 norm:   " << std::sqrt(sum) << "\n";
  std::cout << "  Status:           "
            << (std::isfinite(sum) ? "✓ OK" : "✗ FAILED") << "\n";

  return 0;
}
