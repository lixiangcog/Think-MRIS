import argparse

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Upload a merged Think-MRIS checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=args.checkpoint,
        commit_message="Upload Think-MRIS checkpoint",
    )


if __name__ == "__main__":
    main()
