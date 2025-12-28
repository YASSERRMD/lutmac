import os
from huggingface_hub import snapshot_download

# Define models to download (Total size kept under 4GB)
models = [
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "dir": "models/qwen2.5-0.5b-instruct"
    },
    {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "dir": "models/tinyllama-1.1b-chat"
    },
    {
        "id": "unsloth/functiongemma-270m-it",
        "dir": "models/functiongemma-270m"
    }
]

print("Starting download of sample models (< 4GB total)...")

# Determine base directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

for model in models:
    print(f"\nDownloading {model['id']}...")
    try:
        output_dir = os.path.join(project_root, model['dir'])
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
            
        snapshot_download(
            repo_id=model['id'],
            local_dir=output_dir,
            allow_patterns=["*.safetensors", "*.json", "tokenizer.model", "tokenizer_config.json", "config.json", "special_tokens_map.json"], 
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded to {output_dir}")
    except Exception as e:
        print(f"Failed to download {model['id']}: {e}")

print("\nAll downloads processed.")
