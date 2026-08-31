#!/usr/bin/env python3
"""Publish the audited paper-aligned dataset as a non-destructive Hub config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import CommitOperationAdd, HfApi, get_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="lixiangcog/MRIS-Bench")
    parser.add_argument("--config_name", default="paper_aligned_v4")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("/workspace/datasets/MRIS-Bench-calibrated-25k-paper-v4-final"),
    )
    parser.add_argument(
        "--documentation",
        type=Path,
        default=Path("/workspace/Think-MRIS/dataset/PAPER_ALIGNED_V4.md"),
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max_shard_size", default="500MB")
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--create_pr", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError(
            "Hugging Face authentication is missing. Set HF_TOKEN to a write token; "
            "the token is never written into the dataset repository."
        )

    audit_path = args.dataset_dir / "split_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("all_checks_passed") is not True:
        raise RuntimeError("refusing to upload a dataset whose split audit did not pass")
    if audit.get("rows") != 25809 or audit["coverage"]["test"]["rows"] != 5162:
        raise RuntimeError("unexpected audited row counts")
    if audit.get("removed_columns") != [
        "id",
        "group_id",
        "global_group_id",
        "original_problem",
        "query_style",
        "geometry_provenance",
        "repair_reasons",
        "sam_score",
        "sam_bbox_iou",
        "difficulty_provenance",
    ]:
        raise RuntimeError("unexpected compact-schema column removal")
    if audit.get("removed_solution_keys") != [
        "geometry_provenance",
        "difficulty_provenance",
        "split_provenance",
    ]:
        raise RuntimeError("unexpected compact-solution key removal")
    if audit["input_image_manifest_sha256"] != audit["output_image_manifest_sha256"]:
        raise RuntimeError("image-byte parity check failed")
    if audit["input_non_split_content_sha256"] != audit["output_non_split_content_sha256"]:
        raise RuntimeError("non-split-content parity check failed")

    api = HfApi(token=token)
    identity = api.whoami(token=token)
    repo = api.repo_info(args.repo_id, repo_type="dataset", token=token)
    print(
        json.dumps(
            {
                "authenticated_as": identity.get("name"),
                "repo_id": repo.id,
                "repo_sha_before": repo.sha,
                "config_name": args.config_name,
                "dataset_dir": str(args.dataset_dir),
                "rows": audit["rows"],
                "splits": {
                    split: values["rows"] for split, values in audit["coverage"].items()
                },
                "split_manifest_sha256": audit["split_manifest_sha256"],
                "image_manifest_sha256": audit["output_image_manifest_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    dataset = load_from_disk(str(args.dataset_dir))
    commit = dataset.push_to_hub(
        args.repo_id,
        config_name=args.config_name,
        set_default=False,
        data_dir=args.config_name,
        commit_message="Publish final audited paper-aligned v4 split",
        commit_description=(
            "25,809 calibrated rows with leakage-free 70/10/20 splits. "
            "The 5,162-row test split covers every recovered target category, modality, "
            "and source dataset."
        ),
        token=token,
        revision=args.revision,
        create_pr=args.create_pr,
        max_shard_size=args.max_shard_size,
        embed_external_files=True,
        num_proc=args.num_proc,
    )

    metadata_commit = api.create_commit(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        create_pr=args.create_pr,
        token=token,
        commit_message="Add paper-aligned v4 audit and documentation",
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{args.config_name}/split_audit.json",
                path_or_fileobj=str(audit_path),
            ),
            CommitOperationAdd(
                path_in_repo=f"{args.config_name}/README.md",
                path_or_fileobj=str(args.documentation),
            ),
        ],
    )

    report = {
        "status": "uploaded",
        "repo_id": args.repo_id,
        "config_name": args.config_name,
        "dataset_commit_url": commit.commit_url,
        "metadata_commit_url": metadata_commit.commit_url,
        "split_audit_sha256": sha256(audit_path),
        "split_manifest_sha256": audit["split_manifest_sha256"],
        "image_manifest_sha256": audit["output_image_manifest_sha256"],
    }
    report_path = args.dataset_dir / "hub_upload_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
