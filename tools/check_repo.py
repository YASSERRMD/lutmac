import sys
from huggingface_hub import HfApi

def check_repo(repo_id):
    api = HfApi()
    try:
        info = api.model_info(repo_id)
        print(f"Repo {repo_id} exists. Private: {info.private}, Gated: {info.gated}")
        return True
    except Exception as e:
        print(f"Repo {repo_id} check failed: {e}")
        return False

repos_to_check = [
    "google/functiongemma-270m-it",
    "alpindale/functiongemma-270m-it",
    "unsloth/functiongemma-270m-it",
    "google/functiongemma-2b-it", # guess
]

for r in repos_to_check:
    check_repo(r)
