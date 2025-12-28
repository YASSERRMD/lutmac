import os
import sys
import json
from safetensors.torch import load_file, save_file
from tqdm import tqdm

def merge_model(model_dir, output_file):
    print(f"Merging model from {model_dir} to {output_file}")
    
    # Check for index file
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    shards = []
    
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shards = sorted(list(set(weight_map.values())))
        print(f"Found {len(shards)} shards in index.")
    else:
        # Check files directly
        files = os.listdir(model_dir)
        shards = [f for f in files if f.endswith(".safetensors") and "model" in f]
        shards.sort()
        print(f"Found {len(shards)} .safetensors files (no index).")
    
    if not shards:
        print("No shards found!")
        return

    merged_tensors = {}
    
    for shard_file in tqdm(shards, desc="Loading shards"):
        shard_path = os.path.join(model_dir, shard_file)
        print(f"  Loading {shard_file}...")
        try:
            tensors = load_file(shard_path)
            merged_tensors.update(tensors)
        except Exception as e:
            print(f"Error loading {shard_file}: {e}")
            return

    print(f"Saving merged model ({len(merged_tensors)} tensors) to {output_file}...")
    save_file(merged_tensors, output_file)
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_model.py <model_dir> <output_file>")
        sys.exit(1)
        
    merge_model(sys.argv[1], sys.argv[2])
