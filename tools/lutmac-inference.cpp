/**
 * LutMac Inference Tool
 *
 * Run inference on quantized .lutmac models.
 *
 * Usage:
 *   lutmac-inference --model model.lutmac --prompt "Hello"
 */

#include "lutmac/format.hpp"
#include "lutmac/inference.hpp"
#include "lutmac/model.hpp"
#include "lutmac/types.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

using namespace lutmac;

void print_usage(const char *prog) {
  std::cout << "LutMac Inference Engine v1.1\n";
  std::cout << "Usage: " << prog << " [options]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --model, -m <path>     Path to .lutmac model\n";
  std::cout << "  --tokenizer <path>     Path to tokenizer.json (optional)\n";
  std::cout << "  --prompt, -p <text>    Input prompt\n";
  std::cout
      << "  --max-tokens, -n <n>   Maximum tokens to generate (default: 128)\n";
  std::cout << "  --temperature, -t <f>  Sampling temperature (default: 0.7)\n";
  std::cout << "  --top-p <f>            Top-p sampling (default: 0.9)\n";
  std::cout << "  --min-p <f>            Min-p sampling (default: 0.0)\n";
  std::cout << "  --top-k <n>            Top-k sampling (default: 40)\n";
  std::cout << "  --rep-penalty <f>      Repetition penalty (default: 1.1)\n";
  std::cout << "  --seed <n>             Random seed (default: 42)\n";
  std::cout << "  --streaming            Stream output tokens\n";
  std::cout << "  --benchmark            Run benchmark mode\n";
  std::cout << "  --help, -h             Show this help\n";
}

struct InferArgs {
  std::string model_path;
  std::string tokenizer_path;
  std::string prompt;
  size_t max_tokens = 128;
  float temperature = 0.7f;
  float top_p = 0.9f;
  float min_p = 0.0f;
  int top_k = 40;
  float repetition_penalty = 1.1f;
  uint64_t seed = 42;
  bool streaming = false;
  bool benchmark = false;
};

bool parse_args(int argc, char **argv, InferArgs &args) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return false;
    } else if (arg == "--model" || arg == "-m") {
      if (i + 1 < argc)
        args.model_path = argv[++i];
    } else if (arg == "--tokenizer") {
      if (i + 1 < argc)
        args.tokenizer_path = argv[++i];
    } else if (arg == "--prompt" || arg == "-p") {
      if (i + 1 < argc)
        args.prompt = argv[++i];
    } else if (arg == "--max-tokens" || arg == "-n") {
      if (i + 1 < argc)
        args.max_tokens = std::stoul(argv[++i]);
    } else if (arg == "--temperature" || arg == "-t") {
      if (i + 1 < argc)
        args.temperature = std::stof(argv[++i]);
    } else if (arg == "--top-p") {
      if (i + 1 < argc)
        args.top_p = std::stof(argv[++i]);
    } else if (arg == "--min-p") {
      if (i + 1 < argc)
        args.min_p = std::stof(argv[++i]);
    } else if (arg == "--top-k") {
      if (i + 1 < argc)
        args.top_k = std::stoi(argv[++i]);
    } else if (arg == "--rep-penalty") {
      if (i + 1 < argc)
        args.repetition_penalty = std::stof(argv[++i]);
    } else if (arg == "--seed") {
      if (i + 1 < argc)
        args.seed = std::stoull(argv[++i]);
    } else if (arg == "--streaming") {
      args.streaming = true;
    } else if (arg == "--benchmark") {
      args.benchmark = true;
    }
  }

  return !args.model_path.empty();
}

int main(int argc, char **argv) {
  std::cout.setf(std::ios::unitbuf); // Disable buffering
  std::cerr.setf(std::ios::unitbuf);

  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  InferArgs args;

  if (!parse_args(argc, argv, args)) {
    if (args.model_path.empty()) {
      print_usage(argv[0]);
    }
    return args.model_path.empty() ? 1 : 0;
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
  std::cout << "\033[1;33m    ⚡ Bit-Serial LUT Engine for Ultra-Low-Bit LLM "
               "Inference ⚡\033[0m\n";
  std::cout
      << "\033[90m    "
         "─────────────────────────────────────────────────────────\033[0m\n\n";

  // Check model exists
  if (!validate_lutmac(args.model_path)) {
    std::cerr << "Error: Invalid or missing model file: " << args.model_path
              << "\n";
    return 1;
  }

  // Get model info
  ModelInfo info = get_model_info(args.model_path);

  std::cout << "Model: " << args.model_path << "\n";
  std::cout << "  Size:       " << (info.file_size / 1024.0 / 1024.0)
            << " MB\n";
  std::cout << "  Parameters: ~" << (info.total_parameters / 1000000.0)
            << "M\n";
  std::cout << "  Bits/weight: " << info.bits_per_weight << "\n\n";

  // Select tokenizer
  if (args.tokenizer_path.empty()) {
    // Try to find tokenizer.json in model directory
    std::string model_dir;
    size_t last_slash = args.model_path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
      model_dir = args.model_path.substr(0, last_slash + 1);
    } else {
      model_dir = "./";
    }

    std::string default_tok = model_dir + "tokenizer.json";
    std::ifstream f(default_tok.c_str());
    if (f.good()) {
      args.tokenizer_path = default_tok;
      std::cout << "Auto-detected tokenizer: " << args.tokenizer_path << "\n";
    }
  }

  std::shared_ptr<Tokenizer> tokenizer;
  if (!args.tokenizer_path.empty()) {
    std::cout << "Loading tokenizer from: " << args.tokenizer_path << "\n";
    tokenizer = std::make_shared<JSONTokenizer>(args.tokenizer_path);
  } else {
    std::cout << "Using default ByteTokenizer (raw bytes)\n";
    tokenizer = std::make_shared<ByteTokenizer>();
  }

  // Load model
  std::cout << "Loading model from: " << args.model_path << " ... ";
  std::cout.flush();
  auto start_load = std::chrono::high_resolution_clock::now();

  auto model = load_model(args.model_path);
  if (!model) {
    std::cerr << "\nError: Failed to load model from " << args.model_path
              << "\n";
    return 1;
  }

  auto end_load = std::chrono::high_resolution_clock::now();
  auto load_dur = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_load - start_load);
  std::cout << "Done (" << (load_dur.count() / 1000.0) << "s)\n\n";

  std::cout << "Model Config:\n";
  std::cout << "  Hidden size: " << model->config.hidden_size << "\n";
  std::cout << "  Layers:      " << model->config.num_hidden_layers << "\n";
  std::cout << "  Heads:       " << model->config.num_attention_heads << "\n";
  std::cout << "  KV Heads:    " << model->config.num_key_value_heads << "\n";
  std::cout << "  Vocab:       " << model->config.vocab_size << "\n\n";

  // Transfer ownership to shared_ptr
  std::shared_ptr<Model> shared_model = std::move(model);

  // Create inference engine
  InferenceEngine engine(shared_model, tokenizer);

  // Set up generation config
  GenerationConfig gen_config;
  gen_config.max_new_tokens = args.max_tokens;
  gen_config.temperature = args.temperature;
  gen_config.top_p = args.top_p;
  gen_config.min_p = args.min_p;
  gen_config.top_k = args.top_k;
  gen_config.repetition_penalty = args.repetition_penalty;
  gen_config.seed = args.seed;

  if (args.benchmark) {
    std::cout << "Running benchmark...\n\n";

    // Warm-up
    std::string warmup_prompt = "The quick brown fox";
    auto warmup_result = engine.generate(warmup_prompt, gen_config);

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    const int num_runs = 3;
    size_t total_tokens = 0;

    for (int i = 0; i < num_runs; ++i) {
      auto result = engine.generate(warmup_prompt, gen_config);
      total_tokens += result.num_tokens_generated;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double tokens_per_sec = total_tokens * 1000.0 / duration.count();

    std::cout << "Benchmark Results:\n";
    std::cout << "  Runs:            " << num_runs << "\n";
    std::cout << "  Total tokens:    " << total_tokens << "\n";
    std::cout << "  Total time:      " << (duration.count() / 1000.0) << "s\n";
    std::cout << "  Tokens/second:   " << tokens_per_sec << "\n";

    return 0;
  }

  if (args.prompt.empty()) {
    // Interactive mode
    std::cout << "Interactive mode (type 'quit' to exit)\n\n";

    while (true) {
      std::cout << ">>> ";
      std::getline(std::cin, args.prompt);

      if (args.prompt == "quit" || args.prompt == "exit") {
        break;
      }

      if (args.prompt.empty()) {
        continue;
      }

      if (args.streaming) {
        engine.generate_streaming(args.prompt, gen_config,
                                  [](const std::string &token) {
                                    std::cout << token << std::flush;
                                    return true;
                                  });
        std::cout << "\n\n";
      } else {
        auto result = engine.generate(args.prompt, gen_config);
        std::cout << result.text << "\n\n";
        std::cout << "[" << result.num_tokens_generated << " tokens, "
                  << result.tokens_per_second << " tok/s]\n\n";
      }
    }
  } else {
    // Single prompt
    std::cout << "Prompt: " << args.prompt << "\n\n";
    std::cout << "Response: ";

    if (args.streaming) {
      engine.generate_streaming(args.prompt, gen_config,
                                [](const std::string &token) {
                                  std::cout << token << std::flush;
                                  return true;
                                });
      std::cout << "\n";
    } else {
      auto result = engine.generate(args.prompt, gen_config);
      std::cout << result.text << "\n\n";
      std::cout << "───────────────────────────────────────────\n";
      std::cout << "Tokens generated: " << result.num_tokens_generated << "\n";
      std::cout << "Speed:            " << result.tokens_per_second
                << " tokens/sec\n";
      std::cout << "Stop reason:      " << result.stop_reason << "\n";
    }
  }

  return 0;
}
