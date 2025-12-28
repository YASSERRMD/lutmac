#!/bin/bash
set -e

MODEL_ID="LiquidAI/LFM2-2.6B-Exp"
BASE_URL="https://huggingface.co/${MODEL_ID}/resolve/main"
OUTPUT_DIR="models/lfm2"

mkdir -p $OUTPUT_DIR

echo "Downloading LiquidAI LFM2-2.6B-Exp to $OUTPUT_DIR..."

files=(
    "config.json"
    "generation_config.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "model.safetensors.index.json"
    "model-00001-of-00002.safetensors"
    "model-00002-of-00002.safetensors"
)

for file in "${files[@]}"; do
    if [ -f "$OUTPUT_DIR/$file" ]; then
        echo "File $file already exists, skipping."
    else
        echo "Downloading $file..."
        curl -L -f --progress-bar "$BASE_URL/$file" -o "$OUTPUT_DIR/$file"
    fi
done

echo "Download complete."
