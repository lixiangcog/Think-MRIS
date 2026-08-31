import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Download MRIS-Bench into /workspace")
    parser.add_argument("--repo-id", default="lixiangcog/MRIS-Bench")
    parser.add_argument("--output-dir", default="/workspace/datasets/MRIS-Bench-original")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    return parser.parse_args()


def main():
    args = parse_args()
    cache_root = Path(os.environ.get("THINK_MRIS_CACHE_ROOT", "/workspace/.cache"))
    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ["HF_HOME"] = os.environ.get("THINK_MRIS_HF_HOME", str(cache_root / "huggingface"))
    os.environ["HF_HUB_CACHE"] = os.environ.get(
        "THINK_MRIS_HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub")
    )
    os.environ["TRANSFORMERS_CACHE"] = os.environ.get(
        "THINK_MRIS_TRANSFORMERS_CACHE", os.environ["HF_HUB_CACHE"]
    )

    # Import only after redirecting the container's read-only /cache defaults.
    from huggingface_hub import snapshot_download

    output_dir = Path(args.output_dir).resolve()
    if not str(output_dir).startswith("/workspace/"):
        raise ValueError(f"output directory must be under /workspace, got {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        args.repo_id,
        repo_type="dataset",
        local_dir=output_dir,
    )
    print(f"Original review-stage snapshot saved to: {output_dir}")


if __name__ == "__main__":
    main()
