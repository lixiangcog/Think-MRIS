#!/usr/bin/env python3
"""Normalize paper-aligned difficulty factors to integer ratings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from datasets import DatasetDict, Features, Value, load_from_disk


SPLITS = ("train", "validation", "test")
FACTOR_KEYS = ("scene_complexity", "segmentation_challenge", "linguistic_ambiguity")
ALL_SCORE_KEYS = FACTOR_KEYS + ("difficulty",)


def nearest_rating(value: Any) -> int:
    """Round a finite score to the nearest paper rating in [1, 10]."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite difficulty score: {value!r}")
    return max(1, min(10, int(math.floor(number + 0.5))))


def normalize_solution(solution: str) -> str:
    payload = json.loads(solution)
    for key in FACTOR_KEYS:
        if key in payload:
            payload[key] = nearest_rating(payload[key])
    if all(key in payload for key in FACTOR_KEYS):
        payload["difficulty"] = round(sum(payload[key] for key in FACTOR_KEYS) / 3.0, 2)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="/workspace/datasets/MRIS-Bench-calibrated-25k-paper-v4-final",
    )
    parser.add_argument(
        "--output_dir",
        default="/workspace/datasets/MRIS-Bench-calibrated-25k-paper-v4-final-integer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)

    source = load_from_disk(args.input_dir)
    for split in SPLITS:
        missing = set(ALL_SCORE_KEYS) - set(source[split].column_names)
        if missing:
            raise ValueError(f"{split} is missing score columns: {sorted(missing)}")

    def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
        factors = {key: nearest_rating(row[key]) for key in FACTOR_KEYS}
        return {
            **factors,
            "difficulty": round(sum(factors.values()) / 3.0, 2),
            "solution": normalize_solution(row["solution"]),
        }

    normalized = DatasetDict(
        {
            split: source[split].map(normalize_row, desc=f"integer scores:{split}")
            for split in SPLITS
        }
    )
    score_features = normalized["train"].features.copy()
    for key in FACTOR_KEYS:
        score_features[key] = Value("int64")
    score_features["difficulty"] = Value("float64")
    normalized = DatasetDict(
        {split: normalized[split].cast(Features(score_features)) for split in SPLITS}
    )
    normalized.save_to_disk(str(output), max_shard_size="500MB")

    persisted = load_from_disk(str(output))
    checks: dict[str, Any] = {
        "rows": {split: len(persisted[split]) for split in SPLITS},
        "schema_matches_input": all(
            persisted[split].column_names == source[split].column_names for split in SPLITS
        ),
        "factor_types_are_integer": all(
            str(persisted["train"].features[key]) == "Value('int64')" for key in FACTOR_KEYS
        ),
        "factor_ranges_are_1_to_10": all(
            1 <= value <= 10
            for split in SPLITS
            for key in FACTOR_KEYS
            for value in persisted[split][key]
        ),
        "difficulty_is_mean_of_factors": all(
            abs(row["difficulty"] - round(sum(row[key] for key in FACTOR_KEYS) / 3.0, 2)) < 1e-9
            for split in SPLITS
            for row in persisted[split]
        ),
        "solution_scores_match_columns": all(
            all(
                json.loads(row["solution"])[key] == row[key]
                for key in FACTOR_KEYS + ("difficulty",)
            )
            for split in SPLITS
            for row in persisted[split]
        ),
    }
    checks["all_checks_passed"] = all(
        value if isinstance(value, bool) else value == {"train": 18066, "validation": 2581, "test": 5162}
        for value in checks.values()
    )
    audit = {
        "input_dir": args.input_dir,
        "output_dir": str(output),
        "schema_variant": "paper_aligned_v4_integer_scores",
        "score_columns": list(FACTOR_KEYS),
        "derived_column": "difficulty",
        "rounding": "nearest integer, clipped to [1, 10]",
        "difficulty_rule": "round((scene_complexity + segmentation_challenge + linguistic_ambiguity) / 3, 2)",
        "rows": checks["rows"],
        "checks": checks,
        "source_split_audit_sha256": hashlib.sha256(
            (Path(args.input_dir) / "split_audit.json").read_bytes()
        ).hexdigest(),
    }
    parent_audit = json.loads(
        (Path(args.input_dir) / "split_audit.json").read_text(encoding="utf-8")
    )
    split_audit = {
        "input_dir": args.input_dir,
        "output_dir": str(output),
        "schema_variant": "paper_aligned_v4_integer_scores",
        "rows": sum(checks["rows"].values()),
        "coverage": parent_audit["coverage"],
        "removed_columns": parent_audit["removed_columns"],
        "removed_solution_keys": parent_audit["removed_solution_keys"],
        "retained_columns": persisted["train"].column_names,
        "input_image_manifest_sha256": parent_audit["output_image_manifest_sha256"],
        "output_image_manifest_sha256": parent_audit["output_image_manifest_sha256"],
        "input_non_split_content_sha256": parent_audit["output_non_split_content_sha256"],
        "output_non_split_content_sha256": "changed_by_integer_score_normalization",
        "score_audit": audit,
        "all_checks_passed": checks["all_checks_passed"],
    }
    (output / "split_audit.json").write_text(
        json.dumps(split_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "score_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not checks["all_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
