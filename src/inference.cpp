/**
 * LutMac: Inference Engine Implementation
 */

#include "lutmac/inference.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

namespace lutmac {

// JSONTokenizer implementation
JSONTokenizer::JSONTokenizer(const std::string &json_path) {
  std::ifstream f(json_path);
  if (!f.good()) {
    std::cerr << "Warning: Could not open tokenizer file: " << json_path
              << "\n";
    return;
  }

  // Read full file
  std::stringstream buffer;
  buffer << f.rdbuf();
  std::string content = buffer.str();

  // Very basic parsing to find "vocab": { ... }
  size_t vocab_pos = content.find("\"vocab\"");
  if (vocab_pos == std::string::npos)
    return;

  size_t open_brace = content.find("{", vocab_pos);
  if (open_brace == std::string::npos)
    return;

  size_t current_pos = open_brace + 1;
  while (true) {
    // Find key
    size_t key_start = content.find("\"", current_pos);
    if (key_start == std::string::npos)
      break;

    // Check if we hit the closing brace of vocab object
    size_t next_close = content.find("}", current_pos);
    if (next_close != std::string::npos && next_close < key_start)
      break;

    size_t key_end = content.find("\"", key_start + 1);
    if (key_end == std::string::npos)
      break;

    std::string token_str =
        content.substr(key_start + 1, key_end - key_start - 1);

    // Find value
    size_t val_sep = content.find(":", key_end);
    if (val_sep == std::string::npos)
      break;

    size_t val_start = content.find_first_not_of(" \t\n\r", val_sep + 1);
    size_t val_end = content.find_first_of(",}", val_start);

    if (val_end == std::string::npos)
      break;

    std::string param_str = content.substr(val_start, val_end - val_start);

    int token_id = 0;
    try {
      token_id = std::stoi(param_str);
    } catch (const std::exception &e) {
      // Skip invalid entries (might be some metadata inside vocab if unexpected
      // format) std::cerr << "Skipping invalid vocab entry: " << token_str << "
      // -> " << param_str << "\n";
      current_pos = val_end + 1;
      continue;
    }

    // Handle escaped characters in token (basic)
    std::string processed_token = token_str;

    // Helper to replace all occurrences
    auto replace_all = [&](std::string &str, const std::string &from,
                           const std::string &to) {
      size_t pos = 0;
      while ((pos = str.find(from, pos)) != std::string::npos) {
        str.replace(pos, from.length(), to);
        pos += to.length();
      }
    };

    // Replace unicode escape sequences common in tokenizer.json
    replace_all(processed_token, "\\u0120", "\xC4\xA0");     // Ġ
    replace_all(processed_token, "\\u2581", "\xE2\x96\x81"); // ▁ (SPI)

    vocab[processed_token] = token_id;
    inv_vocab[token_id] = processed_token;

    current_pos = val_end + 1;
  }
}

std::vector<int> JSONTokenizer::encode(const std::string &text) const {
  std::vector<int> tokens;

  // Greedy Longest-Match Encoding
  // This is a simplified BPE-like approach without merges file.
  // We try to match the longest prefix of the current text against the
  // vocabulary.

  // First, preprocess text to replace spaces with 'Ġ' (U+0120) equivalent?
  // Qwen uses byte-level BPE.
  // For simplicity in this engine, we will map space ' ' to 'Ġ' (if that's how
  // it's stored). In the file viewer, I saw: "Ġ": 220.

  std::string input = text;

  // Detect space token: Ġ (U+0120) or ▁ (U+2581)
  std::string space_token = "\xC4\xA0"; // Default to Ġ
  if (vocab.count("\xE2\x96\x81")) {    // Check for ▁
    space_token = "\xE2\x96\x81";
  } else if (vocab.count(" ")) {
    space_token = " ";
  }

  std::string processed_input;
  for (char c : input) {
    if (c == ' ')
      processed_input += space_token;
    else
      processed_input += c;
  }

  size_t idx = 0;
  while (idx < processed_input.length()) {
    bool matched = false;
    // Try to find longest matching prefix
    // Limit max token length to reasonable number (e.g. 20 chars) for
    // performance
    for (size_t len = std::min((size_t)32, processed_input.length() - idx);
         len > 0; --len) {
      std::string sub = processed_input.substr(idx, len);
      if (vocab.count(sub)) {
        tokens.push_back(vocab.at(sub));
        idx += len;
        matched = true;
        break;
      }
    }

    if (!matched) {
      // If no match found, use UNK or byte fallback
      // For now, skip character or use '!'
      // Qwen vocab likely has single chars mapped.
      // If we fail, it's a character not in vocab.
      // Check if single char exists
      std::string chr = processed_input.substr(idx, 1);
      if (vocab.count(chr)) {
        tokens.push_back(vocab.at(chr));
      } else {
        // Fallback: UNK or ignore
        // Use '?' or similar if UNK not defined?
        // Let's just use token 0 or ignore.
      }
      idx++;
    }
  }

  return tokens;
}

std::string JSONTokenizer::decode(const std::vector<int> &tokens) const {
  std::string text;
  for (int t : tokens) {
    if (inv_vocab.count(t)) {
      std::string s = inv_vocab.at(t);

      // Replace "Ġ" (U+0120) with space
      size_t pos = 0;
      while ((pos = s.find("\xC4\xA0")) != std::string::npos) {
        s.replace(pos, 2, " ");
      }

      // Replace "▁" (U+2581) with space
      pos = 0;
      while ((pos = s.find("\xE2\x96\x81")) != std::string::npos) {
        s.replace(pos, 3, " ");
      }

      // Also naive check for "Ġ" in case it wasn't replaced
      // If s contains the UTF8 sequence for Ġ, replace it.
      // Or if it contains valid chars.

      text += s;
    }
  }

  // Post-processing: Replace UTF-8 Ġ (0xC4 0xA0) with space
  // And maybe remove other artifacts
  // Simpler: iterate and replace.
  std::string clean_text;
  for (size_t i = 0; i < text.length();) {
    if (text.size() > i + 1 && (unsigned char)text[i] == 0xC4 &&
        (unsigned char)text[i + 1] == 0xA0) {
      clean_text += " ";
      i += 2;
    } else {
      clean_text += text[i];
      i++;
    }
  }

  return clean_text;
}

std::string JSONTokenizer::decode(int token) const {
  if (inv_vocab.count(token)) {
    std::string text = inv_vocab.at(token);
    // Replace Ġ (U+0120) with space - used by many tokenizers
    size_t pos = 0;
    while ((pos = text.find("\xC4\xA0", pos)) != std::string::npos) {
      text.replace(pos, 2, " ");
    }
    pos = 0;
    while ((pos = text.find("\xE2\x96\x81", pos)) != std::string::npos) {
      text.replace(pos, 3, " ");
    }
    return text;
  }
  return "";
}

// InferenceEngine implementation

InferenceEngine::InferenceEngine(std::shared_ptr<Model> model,
                                 std::shared_ptr<Tokenizer> tokenizer)
    : model_(model), tokenizer_(tokenizer) {}

GenerationResult InferenceEngine::generate(const std::string &prompt,
                                           const GenerationConfig &config) {
  // Tokenize
  std::vector<int> tokens = tokenizer_->encode(prompt);

  // Prepend BOS if configured (critical for some models like Gemma)
  if (model_->config.bos_token_id != -1) {
    if (tokens.empty() || tokens[0] != model_->config.bos_token_id) {
      tokens.insert(tokens.begin(), model_->config.bos_token_id);
    }
  }

  // std::cerr << "Prompt Tokens: ";
  // for (int t : tokens)
  //   std::cerr << t << " ";
  // std::cerr << "\n" << std::flush;

  GenerationResult result;
  result.input_token_count = tokens.size();

  model_->reset_cache();
  model_->generated_tokens.clear();
  // Initialize with prompt tokens for repetition penalty
  for (int t : tokens) {
    model_->generated_tokens.push_back(t);
  }

  // Prefill
  auto start_time = std::chrono::high_resolution_clock::now();

  // Process input prompt
  // Process prompt
  // // fprintf(stderr, "DEBUG: Start Prefill (%zu tokens)\n", tokens.size());
  for (size_t i = 0; i < tokens.size() - 1; ++i) {
    // // fprintf(stderr, "DEBUG: Prefill Token %zu\n", i);
    model_->forward(tokens[i], i);
  }
  // // fprintf(stderr, "DEBUG: End Prefill Loop\n");

  // Generate
  int next_token = tokens.empty()
                       ? tokenizer_->bos_token()
                       : tokens.back(); // Default start token if empty
  size_t pos = tokens.size() > 0 ? tokens.size() - 1 : 0;

  for (size_t i = 0; i < config.max_new_tokens; ++i) {
    // Forward pass
    model_->forward(next_token, pos);

    // Sample
    next_token = model_->sample(config);

    tokens.push_back(next_token);
    model_->generated_tokens.push_back(
        next_token); // Track for repetition penalty
    result.num_tokens_generated++;
    pos++;

    // Stop conditions
    if (next_token == tokenizer_->eos_token() ||
        (model_->config.eos_token_id != -1 &&
         next_token == model_->config.eos_token_id)) {
      result.stop_reason = "eos";
      break;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  result.tokens_per_second = result.num_tokens_generated * 1000.0 /
                             std::max(1.0, (double)duration.count());
  result.text = tokenizer_->decode(tokens);
  if (result.stop_reason.empty())
    result.stop_reason = "max_tokens";

  return result;
}

void InferenceEngine::generate_streaming(
    const std::string &prompt, const GenerationConfig &config,
    std::function<bool(const std::string &)> callback) {
  std::vector<int> tokens = tokenizer_->encode(prompt);
  model_->reset_cache();

  // Prefill
  if (!tokens.empty()) {
    for (size_t i = 0; i < tokens.size() - 1; ++i) {
      model_->forward(tokens[i], i);
    }
  }

  int next_token = tokens.empty() ? tokenizer_->bos_token() : tokens.back();
  size_t pos = tokens.size() > 0 ? tokens.size() - 1 : 0;

  for (size_t i = 0; i < config.max_new_tokens; ++i) {
    // // fprintf(stderr, "DEBUG: Call Forward\n");
    model_->forward(next_token, pos);
    // // fprintf(stderr, "DEBUG: Return Forward\n");

    // // fprintf(stderr, "DEBUG: Call Sample\n");
    next_token = model_->sample(config);

    // Token generated - call callback (returns false to stop)
    if (!callback(tokenizer_->decode(next_token)))
      break;

    if (next_token == tokenizer_->eos_token())
      break;
    pos++;
  }
}

} // namespace lutmac
