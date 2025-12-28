import os
from huggingface_hub import snapshot_download

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
local_dir = "../models/qwen2.5-1.5b"

print(f"Downloading {model_id} to {local_dir}...")
snapshot_download(repo_id=model_id, 
                  local_dir=local_dir, 
                  allow_patterns=["*.safetensors", "*.json", "tokenizer.model"],
                  local_dir_use_symlinks=False)
print("Download complete.")
