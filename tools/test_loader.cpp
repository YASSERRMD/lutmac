#include "lutmac/format.hpp"
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: test_loader <model_path>" << std::endl;
    return 1;
  }

  std::string path = argv[1];
  lutmac::ModelConfig config;
  std::vector<lutmac::PackedTensor> tensors;

  std::cout << "Loading " << path << "..." << std::endl;
  if (!lutmac::load_lutmac(path, config, tensors)) {
    std::cerr << "Failed to load model." << std::endl;
    return 1;
  }

  std::cout << "Loaded " << tensors.size() << " tensors." << std::endl;
  bool found_lm_head = false;
  for (const auto &t : tensors) {
    if (t.name == "lm_head.weight" || t.name == "model.lm_head.weight") {
      found_lm_head = true;
      std::cout << "FOUND: " << t.name << " (offset=" << t.raw_data.size()
                << " elements)" << std::endl;
    }
  }

  if (!found_lm_head) {
    std::cout << "MISSING: lm_head.weight" << std::endl;
    // Print last 5 tensors to see where it stopped
    size_t start = tensors.size() > 5 ? tensors.size() - 5 : 0;
    for (size_t i = start; i < tensors.size(); ++i) {
      std::cout << "Tensor[" << i << "]: " << tensors[i].name << std::endl;
    }
  }

  return 0;
}
