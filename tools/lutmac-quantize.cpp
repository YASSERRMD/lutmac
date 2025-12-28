/**
 * LutMac Quantization Tool
 *
 * Convert safetensors/GGUF models to .lutmac format.
 *
 * Usage:
 *   lutmac-quantize --input model.safetensors --output model.lutmac
 */

#include "lutmac/format.hpp"
#include "lutmac/quantize.hpp"
#include "lutmac/safetensors_parser.hpp"
#include "lutmac/transform.hpp"
#include "lutmac/types.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace lutmac;

void print_usage(const char *prog) {
  std::cout << "LutMac Quantization Tool v" << LUTMAC_FORMAT_VERSION << "\n\n";
  std::cout << "Usage: " << prog << " [options]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --input, -i <path>    Input model path (safetensors file)\n";
  std::cout << "  --output, -o <path>   Output .lutmac file path\n";
  std::cout << "  --bits, -b <value>    Bits per weight (default: 1.58 for "
               "ternary)\n";
  std::cout << "  --verbose, -v         Verbose output\n";
  std::cout << "  --help, -h            Show this help\n";
}

struct QuantizeArgs {
  std::string input_path;
  std::string output_path;
  float bits = 1.58f;
  bool verbose = false;
};

bool parse_args(int argc, char **argv, QuantizeArgs &args) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return false;
    } else if (arg == "--input" || arg == "-i") {
      if (i + 1 < argc)
        args.input_path = argv[++i];
    } else if (arg == "--output" || arg == "-o") {
      if (i + 1 < argc)
        args.output_path = argv[++i];
    } else if (arg == "--bits" || arg == "-b") {
      if (i + 1 < argc)
        args.bits = std::stof(argv[++i]);
    } else if (arg == "--verbose" || arg == "-v") {
      args.verbose = true;
    }
  }

  return !args.input_path.empty() && !args.output_path.empty();
}

int main(int argc, char **argv) {
  QuantizeArgs args;

  if (!parse_args(argc, argv, args)) {
    if (args.input_path.empty() && args.output_path.empty()) {
      print_usage(argv[0]);
    }
    return args.input_path.empty() ? 1 : 0;
  }

  // ASCII Art Logo
  std::cout << "\n";
  std::cout << "\033[1;36m"; // Cyan bold
  std::cout << R"(
    ██╗     ██╗   ██╗████████╗███╗   ███╗ █████╗  ██████╗
    ██║     ██║   ██║╚══██╔══╝████╗ ████║██╔══██╗██╔════╝
    ██║     ██║   ██║   ██║   ██╔████╔██║███████║██║     
    ██║     ██║   ██║   ██║   ██║╚██╔╝██║██╔══██║██║     
    ███████╗╚██████╔╝   ██║   ██║ ╚═╝ ██║██║  ██║╚██████╗
    ╚══════╝ ╚═════╝    ╚═╝   ╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝
)" << "\033[0m";
  std::cout << "\033[1;33m    ⚡ Bit-Serial LUT Engine for Ultra-Low-Bit "
               "Quantization ⚡\033[0m\n";
  std::cout
      << "\033[90m    "
         "─────────────────────────────────────────────────────────\033[0m\n\n";

  std::cout << "Input:  " << args.input_path << "\n";
  std::cout << "Output: " << args.output_path << "\n";
  std::cout << "Bits:   " << args.bits << " bpw\n\n";

  auto start_time = std::chrono::high_resolution_clock::now();

  // Check if input exists
  std::ifstream check(args.input_path);
  if (!check.good()) {
    std::cerr << "Error: Input file not found: " << args.input_path << "\n";
    return 1;
  }
  check.close();

  // Load safetensors
  std::cout << "Loading safetensors file...\n";
  std::vector<LoadedTensor> loaded_tensors;
  std::string error;

  if (!load_safetensors(args.input_path, loaded_tensors, error)) {
    std::cerr << "Error loading safetensors: " << error << "\n";
    return 1;
  }

  std::cout << "Loaded " << loaded_tensors.size() << " tensors\n\n";

  // Print tensor info if verbose
  if (args.verbose) {
    std::cout << "Tensors:\n";
    for (const auto &t : loaded_tensors) {
      std::cout << "  " << t.name << " [";
      for (size_t i = 0; i < t.shape.size(); ++i) {
        if (i > 0)
          std::cout << ", ";
        std::cout << t.shape[i];
      }
      std::cout << "] " << dtype_to_string(t.original_dtype) << "\n";
    }
    std::cout << "\n";
  }

  // Infer model config from tensor shapes
  ModelConfig config;
  config.model_type = "qwen2";

  // Detect Gemma model from path
  std::string lower_path = args.input_path;
  std::transform(lower_path.begin(), lower_path.end(), lower_path.begin(),
                 ::tolower);
  if (lower_path.find("gemma") != std::string::npos) {
    config.model_type = "gemma";
    config.activation = ActivationType::GELU;
    std::cout
        << "Auto-detected Gemma architecture (using GELU + Offset RMSNorm)\n";
  }

  config.bits_per_weight = args.bits;

  // Try to infer hidden size from embed_tokens tensor
  for (const auto &t : loaded_tensors) {
    if (t.name.find("embed_tokens") != std::string::npos &&
        t.shape.size() == 2) {
      config.vocab_size = t.shape[0];
      config.hidden_size = t.shape[1];
    }
    if (t.name.find("mlp.gate_proj") != std::string::npos &&
        t.shape.size() == 2) {
      config.intermediate_size = t.shape[0];
    }
    if (t.name.find("self_attn.q_proj.weight") != std::string::npos &&
        t.shape.size() == 2) {
      // hidden_size from q_proj output
      size_t q_out = t.shape[0];
      // Try to determine num_heads
      if (q_out % 64 == 0) {
        config.num_attention_heads = q_out / 64;
      } else if (q_out % 128 == 0) {
        config.num_attention_heads = q_out / 128;
      }
    }
    if (t.name.find("self_attn.k_proj.weight") != std::string::npos &&
        t.shape.size() == 2) {
      // k_proj output size = num_kv_heads * head_dim
      size_t k_out = t.shape[0];
      // Try to determine num_kv_heads (same head_dim as Q)
      if (k_out % 64 == 0) {
        config.num_key_value_heads = k_out / 64;
      } else if (k_out % 128 == 0) {
        config.num_key_value_heads = k_out / 128;
      }
    }
  }

  // Count layers
  size_t max_layer = 0;
  for (const auto &t : loaded_tensors) {
    size_t pos = t.name.find("layers.");
    if (pos != std::string::npos) {
      size_t end_pos = t.name.find(".", pos + 7);
      if (end_pos != std::string::npos) {
        int layer_num = std::stoi(t.name.substr(pos + 7, end_pos - pos - 7));
        max_layer = std::max(max_layer, static_cast<size_t>(layer_num + 1));
      }
    }
  }
  config.num_hidden_layers = max_layer;

  // Load config.json if available
  std::string model_dir;
  size_t last_slash = args.input_path.find_last_of("/\\");
  if (last_slash != std::string::npos) {
    model_dir = args.input_path.substr(0, last_slash + 1);
  } else {
    model_dir = "./";
  }

  std::string config_path = model_dir + "config.json";
  std::ifstream f(config_path);
  if (f.good()) {
    std::cout << "Loading config from " << config_path << "...\n";
    // Simple manual parsing to avoid strict JSON dependency if not present
    // Or just look for specific keys
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());

    // Helper lambda to find value string
    auto find_val_str = [&](const std::string &key) -> std::string {
      size_t pos = content.find("\"" + key + "\"");
      if (pos == std::string::npos)
        return "";
      pos = content.find(":", pos);
      if (pos == std::string::npos)
        return "";
      size_t end = content.find_first_of(",}", pos);
      if (end == std::string::npos)
        return "";
      std::string val_str = content.substr(pos + 1, end - pos - 1);
      // Clean whitespace
      size_t first = val_str.find_first_not_of(" \t\n\r");
      if (first == std::string::npos)
        return "";
      size_t last = val_str.find_last_not_of(" \t\n\r");
      val_str = val_str.substr(first, last - first + 1);
      return val_str;
    };

    auto find_int = [&](const std::string &key) -> int {
      std::string s = find_val_str(key);
      if (s.empty())
        return -1;
      try {
        return std::stoi(s);
      } catch (...) {
        return -1;
      }
    };

    auto find_float = [&](const std::string &key) -> float {
      std::string s = find_val_str(key);
      if (s.empty())
        return -1.0f;
      try {
        return std::stof(s);
      } catch (...) {
        return -1.0f;
      }
    };

    // Auto-detect model type from config.json content
    std::string content_lower = content;
    std::transform(content_lower.begin(), content_lower.end(),
                   content_lower.begin(), ::tolower);

    if (content_lower.find("gemma") != std::string::npos) {
      config.model_type = "gemma";
      config.activation = ActivationType::GELU;
      std::cout << "  Detected model type: Gemma (from config.json)\n";
    } else if (content_lower.find("qwen") != std::string::npos) {
      config.model_type = "qwen2";
      std::cout << "  Detected model type: Qwen2 (from config.json)\n";
    }

    int hidden = find_int("hidden_size");
    int heads = find_int("num_attention_heads");
    int kv_heads = find_int("num_key_value_heads");
    int layers = find_int("num_hidden_layers");
    int vocab = find_int("vocab_size");
    int inter = find_int("intermediate_size");
    int head_dim = find_int("head_dim");
    int bos_id = find_int("bos_token_id");
    int eos_id = find_int("eos_token_id");
    float theta = find_float("rope_theta");
    float eps = find_float("rms_norm_eps");

    if (hidden > 0)
      config.hidden_size = hidden;
    if (heads > 0)
      config.num_attention_heads = heads;
    if (kv_heads > 0)
      config.num_key_value_heads = kv_heads;
    if (layers > 0)
      config.num_hidden_layers = layers;
    if (vocab > 0)
      config.vocab_size = vocab;
    if (inter > 0)
      config.intermediate_size = inter;
    if (head_dim > 0)
      config.head_dim_val = head_dim;
    if (bos_id != -1)
      config.bos_token_id = bos_id;
    if (eos_id != -1)
      config.eos_token_id = eos_id;
    if (theta > 0.0f)
      config.rope_theta = theta;
    if (eps > 0.0f)
      config.rms_norm_eps = eps;

    // Parse RoPE scaling (crude parsing of nested object)
    size_t scaling_pos = content.find("\"rope_scaling\"");
    if (scaling_pos != std::string::npos) {
      size_t factor_pos = content.find("\"factor\"", scaling_pos);
      if (factor_pos != std::string::npos) {
        size_t colon = content.find(":", factor_pos);
        size_t end = content.find_first_of(",}", colon);
        if (colon != std::string::npos && end != std::string::npos) {
          try {
            config.rope_scaling_factor =
                std::stof(content.substr(colon + 1, end - colon - 1));
          } catch (...) {
          }
        }
      }
    }
  } else {
    std::cout << "Warning: config.json not found, using inferred values\n";
    // Keep heuristic fallback...
  }

  // Ensure KV heads is set
  if (config.num_key_value_heads == 0) {
    config.num_key_value_heads = config.num_attention_heads;
  }

  std::cout << "Inferred Model Configuration:\n";
  std::cout << "  Type:            " << config.model_type << "\n";
  std::cout << "  Vocab size:      " << config.vocab_size << "\n";
  std::cout << "  Hidden size:     " << config.hidden_size << "\n";
  std::cout << "  Intermediate:    " << config.intermediate_size << "\n";
  std::cout << "  Layers:          " << config.num_hidden_layers << "\n";
  std::cout << "  Attention heads: " << config.num_attention_heads << "\n";
  std::cout << "  KV heads:        " << config.num_key_value_heads << "\n";
  std::cout << "  Head Dim:        " << config.head_dim() << "\n\n";

  // Set bits per weight in config
  config.bits_per_weight = args.bits;

  // Quantize tensors
  std::cout << "Quantizing tensors...\n";
  std::vector<PackedTensor> packed_tensors;
  size_t total_params = 0;

  for (size_t i = 0; i < loaded_tensors.size(); ++i) {
    const auto &t = loaded_tensors[i];

    // Decide whether to quantize
    // Quantize ALL large 2D matrices to meet strict size target
    // Heuristic: keep small 1D tensors (biases, norms) in FP32
    bool is_2d = t.shape.size() >= 2;
    bool is_embedding = (t.name.find("embed_tokens") != std::string::npos) ||
                        (t.name.find("lm_head") != std::string::npos);

    // For 4-bit mode, quantize embeddings too for better compression
    // For ternary mode, also quantize embeddings (use ternary for embeds too)
    bool keep_fp32 = (t.shape.size() <= 1);
    // Note: embeddings will be quantized for all bit-widths now

    bool is_large = t.data.size() > 4096; // > 16KB

    bool is_conv_filter = t.name.find("conv.conv.weight") != std::string::npos;
    bool should_quantize = !keep_fp32 && is_2d && is_large && !is_conv_filter;

    // For bits >= 16, keep everything in FP32 (baseline test)
    if (args.bits >= 32.0f) {
      should_quantize = false;
    }

    if (should_quantize) {
      PackedTensor packed;

      // Use same bit-width for all layers including embeddings
      // (Int8 protection was causing 2x slowdown for low-bit models)
      bool use_int8_forcing = is_embedding && args.bits < 7.5f;

      if (args.bits >= 7.5f || use_int8_forcing) {
        // Use 8-bit quantization
        packed = quantize_tensor_int8(t.data.data(), t.shape, t.name);
        if (args.verbose) {
          std::cout << "  Quantized (8-bit): " << t.name << " ("
                    << t.data.size() << " params)\n";
        }
      } else if (args.bits >= 5.5f) {
        // Use 6-bit quantization
        packed = quantize_tensor_int6(t.data.data(), t.shape, t.name);
        if (args.verbose) {
          std::cout << "  Quantized (6-bit): " << t.name << " ("
                    << t.data.size() << " params)\n";
        }
      } else if (args.bits >= 4.5f) {
        // Use 5-bit quantization
        packed = quantize_tensor_int5(t.data.data(), t.shape, t.name);
        if (args.verbose) {
          std::cout << "  Quantized (5-bit): " << t.name << " ("
                    << t.data.size() << " params)\n";
        }
      } else if (args.bits >= 3.5f) {
        // Use 4-bit quantization (covers 4.0 and values close to it)
        packed = quantize_tensor_int4(t.data.data(), t.shape,
                                      t.name); // NO FWHT for 4-bit
        if (args.verbose) {
          std::cout << "  Quantized (4-bit): " << t.name << " ("
                    << t.data.size() << " params)\n";
        }
      } else if (args.bits >= 2.5f) {
        // Use 3-bit quantization
        packed = quantize_tensor_int3(t.data.data(), t.shape, t.name);
        if (args.verbose) {
          std::cout << "  Quantized (3-bit): " << t.name << " ("
                    << t.data.size() << " params)\n";
        }
      } else if (args.bits >= 1.5f) {
        // Use 2-bit quantization with Hadamard Rotation (RRQ/Incoherence
        // Processing)
        std::vector<float> data_copy = t.data;
        lutmac::apply_hadamard_rotation(data_copy.data(), data_copy.size());

        packed = quantize_tensor_int2(data_copy.data(), t.shape, t.name);
        if (args.verbose) {
          std::cout << "  Quantized (2-bit+FWHT): " << t.name << " ("
                    << t.data.size() << " params)\n";
        }
      } else if (args.bits <= 1.1f) {
        // Use binary (1-bit) quantization with Hadamard Rotation
        std::vector<float> data_copy = t.data;
        lutmac::apply_hadamard_rotation(data_copy.data(), data_copy.size());

        packed = quantize_tensor_binary(data_copy.data(), t.shape, t.name);
        if (args.verbose) {
          std::cout << "  Quantized (binary+FWHT): " << t.name << " ("
                    << t.data.size() << " params)\n";
        }
      } else {
        // Use ternary (1.58-bit) quantization with Hadamard Rotation
        std::vector<float> data_copy = t.data;
        lutmac::apply_hadamard_rotation(data_copy.data(), data_copy.size());

        packed = quantize_tensor(data_copy.data(), t.shape, t.name);
        if (args.verbose) {
          std::cout << "  Quantized (ternary+FWHT): " << t.name << " ("
                    << t.data.size() << " params)\n";
        }
      }
      packed_tensors.push_back(std::move(packed));
    } else {
      // Store as raw FP32 (norms, biases, etc)
      PackedTensor raw;
      raw.name = t.name;
      raw.shape = t.shape;
      raw.raw_data = t.data;
      raw.is_quantized = false;

      packed_tensors.push_back(std::move(raw));

      if (args.verbose) {
        std::cout << "  Stored FP32: " << t.name << " (" << t.data.size()
                  << " params)\n";
      }
    }

    total_params += t.data.size();

    if (!args.verbose && i % 10 == 0) {
      std::cout << "\r  Progress: " << (i + 1) << "/" << loaded_tensors.size()
                << std::flush;
    }
  }
  std::cout << "\n\n";

  // Calculate statistics
  size_t total_blocks = 0;
  for (const auto &t : packed_tensors) {
    total_blocks += t.blocks.size();
    total_blocks += t.int2_blocks.size();
    total_blocks += t.int3_blocks.size();
    total_blocks += t.int4_blocks.size();
    total_blocks += t.int5_blocks.size();
    total_blocks += t.int6_blocks.size();
    total_blocks += t.int8_blocks.size();
    total_blocks += t.binary_blocks.size();
  }

  std::cout << "Quantization Statistics:\n";
  std::cout << "  Quantized tensors: " << packed_tensors.size() << "\n";
  std::cout << "  Total parameters:  " << (total_params / 1000000.0) << "M\n";
  std::cout << "  Total blocks:      " << total_blocks << "\n";

  size_t file_size = 0;
  for (const auto &t : packed_tensors) {
    if (!t.blocks.empty())
      file_size += t.blocks.size() * sizeof(BitPlaneBlock);
    if (!t.int2_blocks.empty())
      file_size += t.int2_blocks.size() * sizeof(Int2Block);
    if (!t.int3_blocks.empty())
      file_size += t.int3_blocks.size() * sizeof(Int3Block);
    if (!t.int4_blocks.empty())
      file_size += t.int4_blocks.size() * sizeof(Int4Block);
    if (!t.int5_blocks.empty())
      file_size += t.int5_blocks.size() * sizeof(Int5Block);
    if (!t.int6_blocks.empty())
      file_size += t.int6_blocks.size() * sizeof(Int6Block);
    if (!t.int8_blocks.empty())
      file_size += t.int8_blocks.size() * sizeof(Int8Block);
    if (!t.binary_blocks.empty())
      file_size += t.binary_blocks.size() * sizeof(BinaryBlock);
    file_size += t.raw_data.size() * 4;
  }
  file_size += 4096; // Header overhead estimate

  std::cout << "  Estimated size:    " << (file_size / 1024.0 / 1024.0)
            << " MB\n";
  std::cout << "  Bits per weight:   " << args.bits << "\n";
  std::cout << "  Compression ratio: " << (32.0 / args.bits) << "x vs FP32\n\n";

  // Save
  std::cout << "Saving to: " << args.output_path << "\n";

  if (!save_lutmac(args.output_path, config, packed_tensors)) {
    std::cerr << "Error: Failed to save model\n";
    return 1;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  std::cout << "\n✓ Quantization complete in " << (duration.count() / 1000.0)
            << "s\n";

  return 0;
}
