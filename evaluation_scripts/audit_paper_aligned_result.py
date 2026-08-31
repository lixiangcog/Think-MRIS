#!/usr/bin/env python3
"""Validate completeness and hash the full paper-aligned evaluation result."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_dir", required=True, type=Path)
    parser.add_argument(
        "--dataset_dir",
        default=Path("/workspace/datasets/MRIS-Bench-calibrated-25k-paper-v4-final-integer"),
        type=Path,
    )
    parser.add_argument("--expected_rows", default=5162, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    predictions_path = args.evaluation_dir / "predictions.jsonl"
    summary_path = args.evaluation_dir / "summary.json"
    config_path = args.evaluation_dir / "run_config.json"
    dataset_audit_path = args.dataset_dir / "split_audit.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dataset_audit = json.loads(dataset_audit_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indices = [int(record["dataset_index"]) for record in records]
    original_ids = [str(record["original_id"]) for record in records]

    assert len(records) == args.expected_rows, (len(records), args.expected_rows)
    assert sorted(indices) == list(range(args.expected_rows)), (
        len(set(indices)),
        min(indices, default=None),
        max(indices, default=None),
    )
    assert summary["dataset_rows"] == args.expected_rows, summary["dataset_rows"]
    assert summary["evaluated_rows"] == args.expected_rows, summary["evaluated_rows"]
    assert summary["parse_success_count"] > 0, summary["parse_success_count"]
    assert summary["sam2_success_count"] > 0, summary["sam2_success_count"]
    assert "paper_reference_delta" in summary
    assert "by_target_category" in summary
    assert dataset_audit["all_checks_passed"] is True
    assert dataset_audit["coverage"]["test"]["rows"] == args.expected_rows
    assert dataset_audit["input_image_manifest_sha256"] == dataset_audit["output_image_manifest_sha256"]
    assert dataset_audit["input_non_split_content_sha256"] == dataset_audit["output_non_split_content_sha256"]

    code_commit = subprocess.check_output(
        ["git", "-C", "/workspace/Think-MRIS", "rev-parse", "HEAD"], text=True
    ).strip()
    code_tree = subprocess.check_output(
        ["git", "-C", "/workspace/Think-MRIS", "rev-parse", "HEAD^{tree}"], text=True
    ).strip()

    report = {
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": code_commit,
        "code_tree": code_tree,
        "dataset_dir": str(args.dataset_dir),
        "dataset_split_manifest_sha256": dataset_audit["split_manifest_sha256"],
        "dataset_image_manifest_sha256": dataset_audit["output_image_manifest_sha256"],
        "dataset_non_split_content_sha256": dataset_audit["output_non_split_content_sha256"],
        "evaluation_dir": str(args.evaluation_dir),
        "expected_rows": args.expected_rows,
        "prediction_rows": len(records),
        "unique_dataset_indices": len(set(indices)),
        "unique_original_ids": len(set(original_ids)),
        "parse_success_count": summary["parse_success_count"],
        "sam2_success_count": summary["sam2_success_count"],
        "paper_reference_delta": summary["paper_reference_delta"],
        "artifacts": {
            "predictions": {"path": str(predictions_path), "sha256": sha256(predictions_path)},
            "summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
            "run_config": {"path": str(config_path), "sha256": sha256(config_path)},
            "dataset_split_audit": {
                "path": str(dataset_audit_path),
                "sha256": sha256(dataset_audit_path),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
