from datasets import load_from_disk
from huggingface_hub import create_repo


repo_name = "anonymous/coco_val"
dataset = load_from_disk(f"/path/to/data/{repo_name}")

try:
    create_repo(repo_name, repo_type="dataset", private=False)
    print(f"Create public repo: {repo_name}")
except Exception as e:
    print(f"Repo may already exist: {e}")

dataset.push_to_hub(repo_name, private=False)
print(f"Dataset uploaded to: {repo_name}")

print("\nDataset info:")
print(dataset)
