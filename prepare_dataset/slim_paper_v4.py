#!/usr/bin/env python3
"""Remove redundant review-only columns from the audited v4 DatasetDict."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from datasets import Dataset, DatasetDict, Image as HFImage, concatenate_datasets, load_from_disk


SPLITS = ("train", "validation", "test")
REMOVED_COLUMNS = (
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
)
REMOVED_SOLUTION_KEYS = (
    "geometry_provenance",
    "difficulty_provenance",
    "split_provenance",
)
REQUIRED_COLUMNS = (
    "problem",
    "solution",
    "image",
    "img_height",
    "img_width",
    "original_id",
    "source_dataset",
    "modality",
    "target_category",
    "split",
    "scene_complexity",
    "segmentation_challenge",
    "linguistic_ambiguity",
    "difficulty",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="/workspace/datasets/MRIS-Bench-calibrated-25k-paper-v4",
    )
    parser.add_argument(
        "--output_dir",
        default="/workspace/datasets/MRIS-Bench-calibrated-25k-paper-v4-final",
    )
    return parser.parse_args()


def image_bytes(image: Mapping[str, Any]) -> bytes:
    if image.get("bytes") is not None:
        return image["bytes"]
    if image.get("path"):
        return Path(image["path"]).read_bytes()
    raise ValueError("image has neither bytes nor path")


def compact_solution(solution: str) -> str:
    payload = json.loads(solution)
    for key in REMOVED_SOLUTION_KEYS:
        payload.pop(key, None)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def manifests(dataset: Dataset) -> tuple[str, str]:
    images: list[tuple[str, str]] = []
    retained: list[tuple[str, str]] = []
    raw = dataset.cast_column("image", HFImage(decode=False))
    for index, row in enumerate(raw):
        row_key = f"{index:08d}:{row['original_id']}"
        images.append((row_key, hashlib.sha256(image_bytes(row["image"])).hexdigest()))
        logical = {
            key: value
            for key, value in row.items()
            if key != "image" and key not in REMOVED_COLUMNS
        }
        logical["solution"] = compact_solution(logical["solution"])
        encoded = json.dumps(logical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        retained.append((row_key, hashlib.sha256(encoded.encode("utf-8")).hexdigest()))
    image_manifest = hashlib.sha256(
        "\n".join(f"{key}\t{value}" for key, value in sorted(images)).encode("utf-8")
    ).hexdigest()
    retained_manifest = hashlib.sha256(
        "\n".join(f"{key}\t{value}" for key, value in sorted(retained)).encode("utf-8")
    ).hexdigest()
    return image_manifest, retained_manifest


def coverage(dataset: Dataset) -> dict[str, Any]:
    return {
        "rows": len(dataset),
        "target_categories": dict(sorted(Counter(dataset["target_category"]).items())),
        "modalities": dict(sorted(Counter(dataset["modality"]).items())),
        "source_datasets": dict(sorted(Counter(dataset["source_dataset"]).items())),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)

    source = load_from_disk(args.input_dir)
    for split in SPLITS:
        missing = set(REMOVED_COLUMNS) - set(source[split].column_names)
        if missing:
            raise ValueError(f"{split} is already missing requested columns: {sorted(missing)}")

    source_combined = concatenate_datasets(
        [source[split].cast_column("image", HFImage(decode=False)) for split in SPLITS]
    )
    input_image_hash, input_retained_hash = manifests(source_combined)
    source_group_sets = {split: set(source[split]["global_group_id"]) for split in SPLITS}

    def compact_row(row: dict[str, Any]) -> dict[str, str]:
        return {"solution": compact_solution(row["solution"])}

    slimmed = DatasetDict(
        {
            split: source[split]
            .cast_column("image", HFImage(decode=False))
            .remove_columns(list(REMOVED_COLUMNS))
            .map(compact_row, desc=f"compact solution:{split}")
            .cast_column("image", HFImage())
            for split in SPLITS
        }
    )
    slimmed.save_to_disk(str(output_dir), max_shard_size="500MB")

    persisted = load_from_disk(str(output_dir))
    output_combined = concatenate_datasets(
        [persisted[split].cast_column("image", HFImage(decode=False)) for split in SPLITS]
    )
    output_image_hash, output_retained_hash = manifests(output_combined)

    overlaps = {
        f"{left}_{right}": sorted(source_group_sets[left].intersection(source_group_sets[right]))
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    schema = persisted["train"].column_names
    solution_keys = {
        key
        for split in SPLITS
        for solution in persisted[split]["solution"]
        for key in json.loads(solution)
    }
    checks = {
        "row_total_is_25809": sum(len(persisted[split]) for split in SPLITS) == 25809,
        "split_counts_match": {split: len(persisted[split]) for split in SPLITS}
        == {"train": 18066, "validation": 2581, "test": 5162},
        "requested_columns_removed": not set(REMOVED_COLUMNS).intersection(schema),
        "only_original_id_identifier_retained": (
            "original_id" in schema
            and not {"id", "group_id", "global_group_id"}.intersection(schema)
        ),
        "solution_provenance_removed": not set(REMOVED_SOLUTION_KEYS).intersection(solution_keys),
        "required_columns_present": set(REQUIRED_COLUMNS).issubset(schema),
        "schemas_identical": all(persisted[split].column_names == schema for split in SPLITS),
        "image_bytes_unchanged": input_image_hash == output_image_hash,
        "retained_content_unchanged": input_retained_hash == output_retained_hash,
        "global_group_overlap_is_zero": not any(overlaps.values()),
        "test_has_all_14_target_categories": len(persisted["test"].unique("target_category")) == 14,
        "test_has_all_5_modalities": len(persisted["test"].unique("modality")) == 5,
        "test_has_all_8_sources": len(persisted["test"].unique("source_dataset")) == 8,
    }
    parent_audit_path = Path(args.input_dir) / "split_audit.json"
    parent_audit = json.loads(parent_audit_path.read_text(encoding="utf-8"))
    audit = {
        "input_dir": args.input_dir,
        "output_dir": str(output_dir),
        "schema_variant": "paper_aligned_v4_final",
        "rows": sum(len(persisted[split]) for split in SPLITS),
        "removed_columns": list(REMOVED_COLUMNS),
        "removed_solution_keys": list(REMOVED_SOLUTION_KEYS),
        "retained_columns": schema,
        "coverage": {split: coverage(persisted[split]) for split in SPLITS},
        "global_group_overlaps": overlaps,
        "split_manifest_sha256": parent_audit["split_manifest_sha256"],
        "parent_split_audit_sha256": hashlib.sha256(parent_audit_path.read_bytes()).hexdigest(),
        "input_image_manifest_sha256": input_image_hash,
        "output_image_manifest_sha256": output_image_hash,
        "input_non_split_content_sha256": input_retained_hash,
        "output_non_split_content_sha256": output_retained_hash,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    (output_dir / "split_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
