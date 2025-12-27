#pragma once

/**
 * LutMac: File Format
 *
 * Custom .lutmac binary format for packed ternary weights.
 *
 * Format:
 *   [Magic: 8 bytes "LUTMAC01"]
 *   [Version: 4 bytes]
 *   [Header Size: 4 bytes]
 *   [JSON Metadata: variable]
 *   [Padding to 256-byte alignment]
 *   [Tensor Data: bit-plane packed]
 */

#include "types.hpp"
#include <fstream>
#include <string>
#include <vector>

namespace lutmac {

constexpr size_t FORMAT_ALIGNMENT = 256;

/**
 * File header structure
 */
struct LutmacHeader {
  char magic[8];
  uint32_t version;
  uint32_t header_size;
  // Followed by JSON metadata
};

/**
 * Tensor metadata in JSON
 */
struct TensorMeta {
  std::string name;
  std::vector<size_t> shape;
  size_t offset; // Byte offset in data section
  size_t num_blocks;
  float global_scale;
};

/**
 * Save model to .lutmac format
 */
bool save_lutmac(const std::string &path, const ModelConfig &config,
                 const std::vector<PackedTensor> &tensors);

/**
 * Load model from .lutmac format
 */
bool load_lutmac(const std::string &path, ModelConfig &config,
                 std::vector<PackedTensor> &tensors, bool header_only = false);

/**
 * Validate .lutmac file
 */
bool validate_lutmac(const std::string &path);

/**
 * Get model info without loading weights
 */
struct ModelInfo {
  ModelConfig config;
  size_t file_size;
  size_t num_tensors;
  size_t total_parameters;
  float bits_per_weight;
};

ModelInfo get_model_info(const std::string &path);

} // namespace lutmac
