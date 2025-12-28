/**
 * LutMac: Format Implementation
 *
 * Save/load .lutmac files with support for mixed precision (quantized + FP32).
 */

#include "lutmac/format.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

namespace lutmac {

// Simple JSON serialization/deserialization helpers
namespace json {

std::string escape(const std::string &s) {
  std::string result;
  for (char c : s) {
    if (c == '"')
      result += "\\\"";
    else if (c == '\\')
      result += "\\\\";
    else
      result += c;
  }
  return result;
}

std::string serialize_config(const ModelConfig &config) {
  std::ostringstream ss;
  ss << "{\n";
  ss << "  \"model_type\": \"" << escape(config.model_type) << "\",\n";
  ss << "  \"vocab_size\": " << config.vocab_size << ",\n";
  ss << "  \"hidden_size\": " << config.hidden_size << ",\n";
  ss << "  \"intermediate_size\": " << config.intermediate_size << ",\n";
  ss << "  \"num_hidden_layers\": " << config.num_hidden_layers << ",\n";
  ss << "  \"num_attention_heads\": " << config.num_attention_heads << ",\n";
  ss << "  \"num_key_value_heads\": " << config.num_key_value_heads << ",\n";
  ss << "  \"max_position_embeddings\": " << config.max_position_embeddings
     << ",\n";
  ss << "  \"rms_norm_eps\": " << config.rms_norm_eps << ",\n";
  ss << "  \"rope_theta\": " << config.rope_theta << ",\n";
  ss << "  \"head_dim\": " << config.head_dim_val << ",\n";
  ss << "  \"bos_token_id\": " << config.bos_token_id << ",\n";
  ss << "  \"eos_token_id\": " << config.eos_token_id << ",\n";
  ss << "  \"activation\": " << static_cast<int>(config.activation) << ",\n";
  ss << "  \"norm_type\": " << static_cast<int>(config.norm_type) << ",\n";
  ss << "  \"rope_scaling_factor\": " << config.rope_scaling_factor << ",\n";

  ss << "  \"bits_per_weight\": " << config.bits_per_weight << "\n";
  ss << "}";
  return ss.str();
}

// Helper to extract string value from JSON
std::string get_string(const std::string &json, const std::string &key) {
  size_t key_pos = json.find("\"" + key + "\"");
  if (key_pos == std::string::npos)
    return "";

  size_t val_start = json.find("\"", key_pos + key.length() + 2); // after ":"
  if (val_start == std::string::npos)
    return "";
  val_start++;

  size_t val_end = json.find("\"", val_start);
  if (val_end == std::string::npos)
    return "";

  return json.substr(val_start, val_end - val_start);
}

// Helper to extract int value
long get_int(const std::string &json, const std::string &key) {
  size_t key_pos = json.find("\"" + key + "\"");
  if (key_pos == std::string::npos)
    return 0;

  size_t colon = json.find(":", key_pos);
  size_t val_start = json.find_first_not_of(" \t\n\r", colon + 1);
  size_t val_end = json.find_first_of(",}\n\r", val_start);

  std::string s = json.substr(val_start, val_end - val_start);
  try {
    long result = std::stoll(s);
    return result;
  } catch (...) {
    return 0;
  }
}

// Helper to extract float value
float get_float(const std::string &json, const std::string &key) {
  size_t key_pos = json.find("\"" + key + "\"");
  if (key_pos == std::string::npos)
    return 0.0f;

  size_t colon = json.find(":", key_pos);
  size_t val_start = json.find_first_not_of(" \t\n\r", colon + 1);
  size_t val_end = json.find_first_of(",}\n\r", val_start);

  std::string s = json.substr(val_start, val_end - val_start);
  try {
    return std::stof(s);
  } catch (...) {
    return 0.0f;
  }
  return 0.0f;
}

// Helper to extract integer array
std::vector<int> get_int_array(const std::string &json,
                               const std::string &key) {
  std::vector<int> result;
  size_t key_pos = json.find("\"" + key + "\"");
  if (key_pos == std::string::npos)
    return result;

  size_t colon = json.find(":", key_pos);
  size_t start_bracket = json.find("[", colon);
  if (start_bracket == std::string::npos)
    return result;

  size_t end_bracket = json.find("]", start_bracket);
  if (end_bracket == std::string::npos)
    return result;

  std::string content =
      json.substr(start_bracket + 1, end_bracket - start_bracket - 1);
  std::istringstream ss(content);
  std::string segment;
  while (std::getline(ss, segment, ',')) {
    try {
      result.push_back(std::stoi(segment));
    } catch (...) {
    }
  }
  return result;
}

} // namespace json

bool save_lutmac(const std::string &path, const ModelConfig &config,
                 const std::vector<PackedTensor> &tensors) {
  std::ofstream file(path, std::ios::binary);
  if (!file)
    return false;

  // Write magic
  file.write(LUTMAC_MAGIC, 8);

  // Write version
  uint32_t version = LUTMAC_FORMAT_VERSION;
  file.write(reinterpret_cast<char *>(&version), 4);

  // Serialize metadata
  std::ostringstream meta;
  meta << "{\n";
  meta << "  \"config\": " << json::serialize_config(config) << ",\n";
  meta << "  \"tensors\": [\n";

  size_t offset = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto &t = tensors[i];
    meta << "    {\n";
    meta << "      \"name\": \"" << json::escape(t.name) << "\",\n";
    meta << "      \"shape\": [";
    for (size_t j = 0; j < t.shape.size(); ++j) {
      if (j > 0)
        meta << ", ";
      meta << t.shape[j];
    }
    meta << "],\n";
    meta << "      \"offset\": " << offset << ",\n";

    size_t stored_bytes = 0;
    if (t.is_quantized) {
      if (t.quant_bits == 8 && !t.int8_blocks.empty()) {
        meta << "      \"type\": \"int8\",\n";
        meta << "      \"num_blocks\": " << t.int8_blocks.size() << ",\n";
        meta << "      \"quant_bits\": 8\n";
        stored_bytes = t.int8_blocks.size() * sizeof(Int8Block);
      } else if (t.quant_bits == 6 && !t.int6_blocks.empty()) {
        meta << "      \"type\": \"int6\",\n";
        meta << "      \"num_blocks\": " << t.int6_blocks.size() << ",\n";
        meta << "      \"quant_bits\": 6\n";
        stored_bytes = t.int6_blocks.size() * sizeof(Int6Block);
      } else if (t.quant_bits == 5 && !t.int5_blocks.empty()) {
        meta << "      \"type\": \"int5\",\n";
        meta << "      \"num_blocks\": " << t.int5_blocks.size() << ",\n";
        meta << "      \"quant_bits\": 5\n";
        stored_bytes = t.int5_blocks.size() * sizeof(Int5Block);
      } else if (t.quant_bits == 4 && !t.int4_blocks.empty()) {
        meta << "      \"type\": \"int4\",\n";
        meta << "      \"num_blocks\": " << t.int4_blocks.size() << ",\n";
        meta << "      \"quant_bits\": 4\n";
        stored_bytes = t.int4_blocks.size() * sizeof(Int4Block);
      } else if (t.quant_bits == 3 && !t.int3_blocks.empty()) {
        meta << "      \"type\": \"int3\",\n";
        meta << "      \"num_blocks\": " << t.int3_blocks.size() << ",\n";
        meta << "      \"quant_bits\": 3\n";
        stored_bytes = t.int3_blocks.size() * sizeof(Int3Block);
      } else if (t.quant_bits == 2 && !t.int2_blocks.empty()) {
        meta << "      \"type\": \"int2\",\n";
        meta << "      \"num_blocks\": " << t.int2_blocks.size() << ",\n";
        meta << "      \"quant_bits\": 2\n";
        stored_bytes = t.int2_blocks.size() * sizeof(Int2Block);
      } else if (t.quant_bits == 1 && !t.binary_blocks.empty()) {
        meta << "      \"type\": \"binary\",\n";
        meta << "      \"num_blocks\": " << t.binary_blocks.size() << ",\n";
        meta << "      \"quant_bits\": 1\n";
        stored_bytes = t.binary_blocks.size() * sizeof(BinaryBlock);
      } else if (!t.blocks.empty()) {
        meta << "      \"type\": \"ternary\",\n";
        meta << "      \"num_blocks\": " << t.blocks.size() << ",\n";
        meta << "      \"quant_bits\": 1.58\n";
        stored_bytes = t.blocks.size() * sizeof(BitPlaneBlock);
      }
    } else {
      meta << "      \"type\": \"raw\",\n";
      meta << "      \"num_blocks\": 0\n";
      stored_bytes = t.raw_data.size() * sizeof(float);
    }

    meta << "    }";
    if (i < tensors.size() - 1)
      meta << ",";
    meta << "\n";

    offset += stored_bytes;
  }

  meta << "  ]\n";
  meta << "}";

  std::string metadata = meta.str();

  // Write header size
  uint32_t header_size = static_cast<uint32_t>(metadata.size());
  file.write(reinterpret_cast<char *>(&header_size), 4);

  // Write metadata
  file.write(metadata.c_str(), metadata.size());

  // Pad to alignment
  size_t current_pos = 16 + metadata.size();
  size_t padding =
      (FORMAT_ALIGNMENT - (current_pos % FORMAT_ALIGNMENT)) % FORMAT_ALIGNMENT;
  std::vector<char> pad(padding, 0);
  file.write(pad.data(), padding);

  // Write tensor data
  for (const auto &t : tensors) {
    if (t.is_quantized) {
      if (t.quant_bits == 8 && !t.int8_blocks.empty()) {
        file.write(reinterpret_cast<const char *>(t.int8_blocks.data()),
                   t.int8_blocks.size() * sizeof(Int8Block));
      } else if (t.quant_bits == 6 && !t.int6_blocks.empty()) {
        file.write(reinterpret_cast<const char *>(t.int6_blocks.data()),
                   t.int6_blocks.size() * sizeof(Int6Block));
      } else if (t.quant_bits == 5 && !t.int5_blocks.empty()) {
        file.write(reinterpret_cast<const char *>(t.int5_blocks.data()),
                   t.int5_blocks.size() * sizeof(Int5Block));
      } else if (t.quant_bits == 4 && !t.int4_blocks.empty()) {
        file.write(reinterpret_cast<const char *>(t.int4_blocks.data()),
                   t.int4_blocks.size() * sizeof(Int4Block));
      } else if (t.quant_bits == 3 && !t.int3_blocks.empty()) {
        file.write(reinterpret_cast<const char *>(t.int3_blocks.data()),
                   t.int3_blocks.size() * sizeof(Int3Block));
      } else if (t.quant_bits == 2 && !t.int2_blocks.empty()) {
        file.write(reinterpret_cast<const char *>(t.int2_blocks.data()),
                   t.int2_blocks.size() * sizeof(Int2Block));
      } else if (t.quant_bits == 1 && !t.binary_blocks.empty()) {
        file.write(reinterpret_cast<const char *>(t.binary_blocks.data()),
                   t.binary_blocks.size() * sizeof(BinaryBlock));
      } else if (!t.blocks.empty()) {
        file.write(reinterpret_cast<const char *>(t.blocks.data()),
                   t.blocks.size() * sizeof(BitPlaneBlock));
      }
    } else {
      file.write(reinterpret_cast<const char *>(t.raw_data.data()),
                 t.raw_data.size() * sizeof(float));
    }
  }

  return file.good();
}

bool load_lutmac(const std::string &path, ModelConfig &config,
                 std::vector<PackedTensor> &tensors, bool header_only) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    return false;

  // Read and verify magic
  char magic[8];
  file.read(magic, 8);
  if (std::memcmp(magic, LUTMAC_MAGIC, 8) != 0)
    return false;

  // Read version
  uint32_t version;
  file.read(reinterpret_cast<char *>(&version), 4);
  if (version != LUTMAC_FORMAT_VERSION)
    return false;

  // Read header size
  uint32_t header_size;
  file.read(reinterpret_cast<char *>(&header_size), 4);

  // Read metadata
  std::string metadata(header_size, ' ');
  file.read(&metadata[0], header_size);
  // std::cerr << "DEBUG: Metadata read (" << header_size << " bytes):\n" <<
  // metadata.substr(0, 500) << "...\n" << std::flush;

  // Parse Header (Basic implementation)
  config.model_type = json::get_string(metadata, "model_type");
  config.vocab_size = json::get_int(metadata, "vocab_size");
  config.hidden_size = json::get_int(metadata, "hidden_size");
  config.intermediate_size = json::get_int(metadata, "intermediate_size");
  config.num_hidden_layers = json::get_int(metadata, "num_hidden_layers");
  config.num_attention_heads = json::get_int(metadata, "num_attention_heads");
  config.num_key_value_heads = json::get_int(metadata, "num_key_value_heads");
  config.bits_per_weight = json::get_float(metadata, "bits_per_weight");
  config.rms_norm_eps = json::get_float(metadata, "rms_norm_eps");
  config.rope_theta = json::get_float(metadata, "rope_theta");
  config.rope_scaling_factor = json::get_float(metadata, "rope_scaling_factor");
  if (config.rope_scaling_factor <= 0.0f)
    config.rope_scaling_factor = 1.0f;
  config.head_dim_val = json::get_int(metadata, "head_dim");
  if (long val = json::get_int(metadata, "bos_token_id"))
    config.bos_token_id = (int)val;
  if (long val = json::get_int(metadata, "eos_token_id"))
    config.eos_token_id = (int)val;
  config.activation =
      static_cast<ActivationType>(json::get_int(metadata, "activation"));
  config.norm_type =
      static_cast<NormType>(json::get_int(metadata, "norm_type"));

  if (config.vocab_size == 0)
    return false; // Parse failed

  // Parse tensors
  // Search for "tensors": [ ... ]
  size_t tensors_start = metadata.find("\"tensors\": [");
  if (tensors_start == std::string::npos)
    return false;

  size_t pos = tensors_start;
  size_t data_start_offset =
      16 + header_size +
      ((FORMAT_ALIGNMENT - ((16 + header_size) % FORMAT_ALIGNMENT)) %
       FORMAT_ALIGNMENT);

  while (true) {
    size_t open_brace = metadata.find("{", pos);
    if (open_brace == std::string::npos)
      break;

    // Extract object string
    size_t close_brace = metadata.find("}", open_brace);
    std::string tensor_json =
        metadata.substr(open_brace, close_brace - open_brace + 1);

    PackedTensor t;
    t.name = json::get_string(tensor_json, "name");

    // Check for int4 vs ternary vs binary vs raw
    bool is_int8 =
        (tensor_json.find("\"type\": \"int8\"") != std::string::npos);
    bool is_int6 =
        (tensor_json.find("\"type\": \"int6\"") != std::string::npos);
    bool is_int5 =
        (tensor_json.find("\"type\": \"int5\"") != std::string::npos);
    bool is_int4 =
        (tensor_json.find("\"type\": \"int4\"") != std::string::npos);
    bool is_int3 =
        (tensor_json.find("\"type\": \"int3\"") != std::string::npos);
    bool is_int2 =
        (tensor_json.find("\"type\": \"int2\"") != std::string::npos);
    bool is_binary =
        (tensor_json.find("\"type\": \"binary\"") != std::string::npos);
    bool is_ternary =
        (tensor_json.find("\"type\": \"ternary\"") != std::string::npos) ||
        (tensor_json.find("\"type\": \"quantized\"") != std::string::npos);

    size_t num_blocks = json::get_int(tensor_json, "num_blocks");

    // Attempt to parse quant_bits carefully as it can be float (1.58)
    std::string qb_key = "\"quant_bits\":";
    size_t qb_pos = tensor_json.find(qb_key);
    if (qb_pos != std::string::npos) {
      size_t val_start =
          tensor_json.find_first_not_of(" \t", qb_pos + qb_key.length());
      size_t val_end = tensor_json.find_first_of(",}\n", val_start);
      std::string val_str = tensor_json.substr(val_start, val_end - val_start);
      try {
        t.quant_bits = std::stof(val_str);
      } catch (...) {
        t.quant_bits = 0;
      }
    }

    if (is_int8) {
      t.is_quantized = true;
      t.int8_blocks.resize(num_blocks);
    } else if (is_int6) {
      t.is_quantized = true;
      t.int6_blocks.resize(num_blocks);
    } else if (is_int5) {
      t.is_quantized = true;
      t.int5_blocks.resize(num_blocks);
    } else if (is_int4) {
      t.is_quantized = true;
      t.int4_blocks.resize(num_blocks);
    } else if (is_int3) {
      t.is_quantized = true;
      t.int3_blocks.resize(num_blocks);
    } else if (is_int2) {
      t.is_quantized = true;
      t.int2_blocks.resize(num_blocks);
    } else if (is_binary) {
      t.is_quantized = true;
      t.binary_blocks.resize(num_blocks);
    } else if (is_ternary || num_blocks > 0) {
      t.is_quantized = true;
      t.blocks.resize(num_blocks);
    } else {
      t.is_quantized = false;
    }

    // Parse shape
    size_t shape_start = tensor_json.find("\"shape\": [");
    if (shape_start != std::string::npos) {
      size_t shape_end = tensor_json.find("]", shape_start);
      std::string shape_str =
          tensor_json.substr(shape_start + 10, shape_end - shape_start - 10);
      std::stringstream ss(shape_str);
      std::string segment;
      while (std::getline(ss, segment, ',')) {
        t.shape.push_back(std::stoul(segment));
      }
    }

    size_t file_offset =
        data_start_offset + json::get_int(tensor_json, "offset");

    // Skip data reading if header only
    if (header_only) {
      tensors.push_back(std::move(t));
      pos = close_brace + 1;
      continue;
    }

    // Read data
    auto old_pos = file.tellg();
    file.seekg(file_offset);

    if (t.is_quantized) {
      if (t.int8_blocks.size() > 0) {
        file.read(reinterpret_cast<char *>(t.int8_blocks.data()),
                  t.int8_blocks.size() * sizeof(Int8Block));
      } else if (t.int6_blocks.size() > 0) {
        file.read(reinterpret_cast<char *>(t.int6_blocks.data()),
                  t.int6_blocks.size() * sizeof(Int6Block));
      } else if (t.int5_blocks.size() > 0) {
        file.read(reinterpret_cast<char *>(t.int5_blocks.data()),
                  t.int5_blocks.size() * sizeof(Int5Block));
      } else if (t.int4_blocks.size() > 0) {
        file.read(reinterpret_cast<char *>(t.int4_blocks.data()),
                  t.int4_blocks.size() * sizeof(Int4Block));
      } else if (t.int3_blocks.size() > 0) {
        file.read(reinterpret_cast<char *>(t.int3_blocks.data()),
                  t.int3_blocks.size() * sizeof(Int3Block));
      } else if (t.int2_blocks.size() > 0) {
        file.read(reinterpret_cast<char *>(t.int2_blocks.data()),
                  t.int2_blocks.size() * sizeof(Int2Block));
      } else if (t.binary_blocks.size() > 0) {
        file.read(reinterpret_cast<char *>(t.binary_blocks.data()),
                  t.binary_blocks.size() * sizeof(BinaryBlock));
      } else if (t.blocks.size() > 0) {
        file.read(reinterpret_cast<char *>(t.blocks.data()),
                  t.blocks.size() * sizeof(BitPlaneBlock));
      }
    } else {
      size_t num_elements = 1;
      for (auto s : t.shape)
        num_elements *= s;
      t.raw_data.resize(num_elements);
      file.read(reinterpret_cast<char *>(t.raw_data.data()),
                num_elements * sizeof(float));
    }

    file.seekg(old_pos);

    tensors.push_back(std::move(t));
    pos = close_brace + 1;
  }

  return true;
}

bool validate_lutmac(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    return false;
  char magic[8];
  file.read(magic, 8);
  return std::memcmp(magic, LUTMAC_MAGIC, 8) == 0;
}

ModelInfo get_model_info(const std::string &path) {
  ModelInfo info;
  info.total_parameters = 0; // Ensure initialized
  info.num_tensors = 0;

  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    return info;
  info.file_size = file.tellg();
  file.close();

  std::vector<PackedTensor> tensors;
  if (!load_lutmac(path, info.config, tensors, true)) {
    return info;
  }

  info.num_tensors = tensors.size();
  info.bits_per_weight = info.config.bits_per_weight;

  // Calculate total params based on unique tensor names to avoid over-counting
  info.total_parameters = 0;
  for (const auto &t : tensors) {
    size_t n = 1;
    for (auto s : t.shape) {
      if (s > 0)
        n *= s;
    }
    if (n > 1) { // Skip empty/scalar tensors
      info.total_parameters += n;
    }
  }

  return info;
}

} // namespace lutmac
