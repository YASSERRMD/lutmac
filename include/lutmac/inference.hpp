#pragma once

/**
 * LutMac: Inference Engine
 *
 * Text generation with sampling strategies.
 */

#include "model.hpp"
#include "types.hpp"
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lutmac {

/**
 * Simple tokenizer interface
 */
class Tokenizer {
public:
  virtual ~Tokenizer() = default;

  virtual std::vector<int> encode(const std::string &text) const = 0;
  virtual std::string decode(const std::vector<int> &tokens) const = 0;
  virtual std::string decode(int token) const = 0;

  virtual int bos_token() const = 0;
  virtual int eos_token() const = 0;
  virtual int pad_token() const = 0;

  virtual size_t vocab_size() const = 0;
};

/**
 * Byte-level tokenizer for testing
 */
class ByteTokenizer : public Tokenizer {
public:
  std::vector<int> encode(const std::string &text) const override {
    std::vector<int> tokens;
    tokens.reserve(text.size());
    for (char c : text) {
      tokens.push_back(static_cast<unsigned char>(c));
    }
    return tokens;
  }

  std::string decode(const std::vector<int> &tokens) const override {
    std::string text;
    text.reserve(tokens.size());
    for (int t : tokens) {
      if (t >= 0 && t < 256) {
        text.push_back(static_cast<char>(t));
      }
    }
    return text;
  }

  std::string decode(int token) const override {
    if (token >= 0 && token < 256) {
      return std::string(1, static_cast<char>(token));
    }
    return "";
  }

  int bos_token() const override { return 1; }
  int eos_token() const override { return 2; }
  int pad_token() const override { return 0; }
  size_t vocab_size() const override { return 256; }
};

/**
 * JSON-based Tokenizer (reads tokenizer.json)
 */
class JSONTokenizer : public Tokenizer {
public:
  explicit JSONTokenizer(const std::string &json_path);
  std::vector<int> encode(const std::string &text) const override;
  std::string decode(const std::vector<int> &tokens) const override;
  std::string decode(int token) const override;

  int bos_token() const override { return bos_token_id; }
  int eos_token() const override { return eos_token_id; }
  int pad_token() const override { return unk_token_id; }
  size_t vocab_size() const override { return vocab.size(); }

private:
  std::unordered_map<std::string, int> vocab;
  std::unordered_map<int, std::string> inv_vocab;
  int unk_token_id = 0;
  int bos_token_id = 1;
  int eos_token_id = 2;
};

/**
 * Sampling strategies
 */
enum class SamplingMethod { GREEDY, TOP_K, TOP_P, TEMPERATURE };

/**
 * Generation result
 */
struct GenerationResult {
  std::vector<int> tokens;
  std::string text;
  size_t input_token_count = 0;
  size_t num_tokens_generated = 0;
  double tokens_per_second = 0.0;
  bool finished = false;
  std::string stop_reason; // "eos", "max_tokens", "stop_sequence"
};

/**
 * Inference engine
 */
class InferenceEngine {
public:
  InferenceEngine(std::shared_ptr<Model> model,
                  std::shared_ptr<Tokenizer> tokenizer);

  /**
   * Generate text continuation
   */
  GenerationResult generate(const std::string &prompt,
                            const GenerationConfig &config);

  /**
   * Generate with streaming callback
   */
  void
  generate_streaming(const std::string &prompt, const GenerationConfig &config,
                     std::function<bool(const std::string &token)> callback);

  /**
   * Get model config
   */
  const ModelConfig &get_config() const { return model_->config; }

private:
  std::shared_ptr<Model> model_;
  std::shared_ptr<Tokenizer> tokenizer_;

  int sample_token(const float *logits, const GenerationConfig &config,
                   const std::vector<int> &generated_tokens);

  void apply_repetition_penalty(float *logits,
                                const std::vector<int> &generated_tokens,
                                float penalty);
};

} // namespace lutmac
