import sys
import os
from huggingface_hub import snapshot_download

if len(sys.argv) < 3:
    print("Usage: python3 download_generic.py <repo_id> <local_dir>")
    sys.exit(1)

model_id = sys.argv[1]
local_dir = sys.argv[2]

if not os.path.exists(local_dir):
    os.makedirs(local_dir)

print(f"Downloading {model_id} to {local_dir}...")
try:
    snapshot_download(repo_id=model_id, 
                      local_dir=local_dir, 
                      allow_patterns=["*.safetensors", "*.json", "tokenizer.model", "tokenizer.json"],
                      local_dir_use_symlinks=False)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

print("Download complete.")
