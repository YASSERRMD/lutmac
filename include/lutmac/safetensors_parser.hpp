#pragma once

/**
 * LutMac: Safetensors Parser Wrapper
 *
 * Wraps the safetensors-cpp library for easy tensor extraction.
 */

#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors.hh"

#include "lutmac/types.hpp"
#include <cstring>
#include <string>
#include <vector>

namespace lutmac {

/**
 * Loaded tensor with float data
 */
struct LoadedTensor {
  std::string name;
  std::vector<size_t> shape;
  std::vector<float> data;
  safetensors::dtype original_dtype;
};

/**
 * Load all tensors from a safetensors file and convert to float32
 */
inline bool load_safetensors(const std::string &path,
                             std::vector<LoadedTensor> &tensors,
                             std::string &error) {
  safetensors::safetensors_t st;
  std::string warn;

  if (!safetensors::mmap_from_file(path, &st, &warn, &error)) {
    return false;
  }

  // Iterate through all tensors
  for (const auto &name : st.tensors.keys()) {
    safetensors::tensor_t tensor;
    if (!st.tensors.at(name, &tensor)) {
      error = "Failed to get tensor: " + name;
      return false;
    }

    LoadedTensor loaded;
    loaded.name = name;
    loaded.shape = tensor.shape;
    loaded.original_dtype = tensor.dtype;

    // Calculate number of elements
    size_t num_elements = 1;
    for (auto s : tensor.shape) {
      num_elements *= s;
    }

    // Get data pointer
    const uint8_t *data_ptr = st.databuffer_addr + tensor.data_offsets[0];
    size_t data_size = tensor.data_offsets[1] - tensor.data_offsets[0];

    // Convert to float32
    loaded.data.resize(num_elements);

    switch (tensor.dtype) {
    case safetensors::kFLOAT32: {
      std::memcpy(loaded.data.data(), data_ptr, num_elements * sizeof(float));
      break;
    }
    case safetensors::kFLOAT16: {
      const uint16_t *fp16_data = reinterpret_cast<const uint16_t *>(data_ptr);
      for (size_t i = 0; i < num_elements; ++i) {
        loaded.data[i] = safetensors::fp16_to_float(fp16_data[i]);
      }
      break;
    }
    case safetensors::kBFLOAT16: {
      const uint16_t *bf16_data = reinterpret_cast<const uint16_t *>(data_ptr);
      for (size_t i = 0; i < num_elements; ++i) {
        loaded.data[i] = safetensors::bfloat16_to_float(bf16_data[i]);
      }
      break;
    }
    default:
      error = "Unsupported dtype for tensor: " + name;
      return false;
    }

    tensors.push_back(std::move(loaded));
  }

  return true;
}

/**
 * Get dtype name as string
 */
inline std::string dtype_to_string(safetensors::dtype dtype) {
  switch (dtype) {
  case safetensors::kBOOL:
    return "bool";
  case safetensors::kUINT8:
    return "uint8";
  case safetensors::kINT8:
    return "int8";
  case safetensors::kINT16:
    return "int16";
  case safetensors::kUINT16:
    return "uint16";
  case safetensors::kFLOAT16:
    return "float16";
  case safetensors::kBFLOAT16:
    return "bfloat16";
  case safetensors::kINT32:
    return "int32";
  case safetensors::kUINT32:
    return "uint32";
  case safetensors::kFLOAT32:
    return "float32";
  case safetensors::kFLOAT64:
    return "float64";
  case safetensors::kINT64:
    return "int64";
  case safetensors::kUINT64:
    return "uint64";
  default:
    return "unknown";
  }
}

} // namespace lutmac
