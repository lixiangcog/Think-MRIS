#!/usr/bin/env python3
"""Create a larger, coverage-constrained and group-safe MRIS-Bench test split.

The calibrated public snapshot does not contain official author splits.  This
script therefore assigns complete ``(source_dataset, group_id)`` groups to a
deterministic 70/10/20 train/validation/test partition.  The test split is
seeded to cover every recovered target category, modality and source dataset.

Images are handled with ``datasets.Image(decode=False)`` throughout the
rewrite.  The generated audit verifies image-byte and non-split-content parity
against the input snapshot, so repartitioning cannot silently alter samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

from datasets import Dataset, DatasetDict, Image as HFImage, concatenate_datasets, load_from_disk


SPLITS = ("train", "validation", "test")
FRACTIONS = {"train": 0.70, "validation": 0.10, "test": 0.20}
PROVENANCE = (
    "deterministic_source_namespaced_group_stratified_70_10_20_"
    "all_test_targets_modalities_sources_official_splits_absent"
)
LABEL_FIELDS = {
    "category": "target_category",
    "modality": "modality",
    "source": "source_dataset",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="/workspace/datasets/MRIS-Bench-calibrated-25k-paper-v3",
    )
    parser.add_argument(
        "--output_dir",
        default="/workspace/datasets/MRIS-Bench-calibrated-25k-paper-v4",
    )
    parser.add_argument("--seed", default="think-mris-paper-v4")
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def stable_rank(seed: str, value: str) -> int:
    return int(hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest(), 16)


def global_group_id(source: str, group_id: Any) -> str:
    return f"{source}::{group_id}"


def row_labels(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(f"{kind}:{row[field]}" for kind, field in LABEL_FIELDS.items())


def raw_image_bytes(image: Mapping[str, Any]) -> bytes:
    payload = image.get("bytes")
    if payload is not None:
        return payload
    path = image.get("path")
    if path:
        return Path(path).read_bytes()
    raise ValueError("image has neither bytes nor path")


def digest_records(dataset: Dataset) -> tuple[str, str]:
    """Return image-byte and split-invariant logical-content digests."""

    image_records: list[tuple[str, str]] = []
    content_records: list[tuple[str, str]] = []
    raw = dataset.cast_column("image", HFImage(decode=False))
    for row in raw:
        row_id = str(row["id"])
        image_records.append((row_id, hashlib.sha256(raw_image_bytes(row["image"])).hexdigest()))

        logical = {key: value for key, value in row.items() if key not in {"image", "split", "global_group_id"}}
        solution = json.loads(logical["solution"])
        solution.pop("split_provenance", None)
        logical["solution"] = solution
        encoded = json.dumps(logical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        content_records.append((row_id, hashlib.sha256(encoded.encode("utf-8")).hexdigest()))

    image_digest = hashlib.sha256(
        "\n".join(f"{key}\t{value}" for key, value in sorted(image_records)).encode("utf-8")
    ).hexdigest()
    content_digest = hashlib.sha256(
        "\n".join(f"{key}\t{value}" for key, value in sorted(content_records)).encode("utf-8")
    ).hexdigest()
    return image_digest, content_digest


def build_groups(metadata: Dataset) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(metadata):
        key = global_group_id(str(row["source_dataset"]), row["group_id"])
        group = groups.setdefault(
            key,
            {"indices": [], "labels": Counter(), "rows": 0, "bare_group_id": str(row["group_id"])},
        )
        group["indices"].append(index)
        group["rows"] += 1
        group["labels"].update(row_labels(row))
    return groups


def seed_coverage(
    *,
    split: str,
    required: set[str],
    groups: Mapping[str, Mapping[str, Any]],
    assignments: MutableMapping[str, str],
    seed: str,
    label_group_frequency: Mapping[str, int],
) -> set[str]:
    """Greedily choose small, information-dense groups until coverage is complete."""

    covered = {
        label
        for key, assigned in assignments.items()
        if assigned == split
        for label in groups[key]["labels"]
    }
    uncovered = set(required) - covered
    while uncovered:
        candidates = []
        for key, group in groups.items():
            if key in assignments:
                continue
            hit = uncovered.intersection(group["labels"])
            if not hit:
                continue
            rarity = sum(1.0 / label_group_frequency[label] for label in hit)
            efficiency = (len(hit) + 3.0 * rarity) / math.sqrt(group["rows"])
            candidates.append((efficiency, len(hit), -group["rows"], -stable_rank(seed, key), key))
        if not candidates:
            break
        key = max(candidates)[-1]
        assignments[key] = split
        uncovered.difference_update(groups[key]["labels"])
    return uncovered


def assign_groups(
    groups: Mapping[str, Mapping[str, Any]], total_rows: int, seed: str
) -> tuple[Dict[str, str], Dict[str, int], Dict[str, set[str]]]:
    target_rows = {
        "test": round(total_rows * FRACTIONS["test"]),
        "validation": round(total_rows * FRACTIONS["validation"]),
    }
    target_rows["train"] = total_rows - target_rows["test"] - target_rows["validation"]

    label_totals: Counter[str] = Counter()
    label_group_frequency: Counter[str] = Counter()
    for group in groups.values():
        label_totals.update(group["labels"])
        label_group_frequency.update(group["labels"].keys())

    assignments: Dict[str, str] = {}
    all_labels = set(label_totals)

    # Test coverage is a hard contract.  Pin the only right-atrium group first,
    # then let the weighted set-cover step satisfy all remaining labels.
    right_atrium = "category:right_atrium"
    for key, group in groups.items():
        if right_atrium in group["labels"]:
            assignments[key] = "test"
            break
    missing_test = seed_coverage(
        split="test",
        required=all_labels,
        groups=groups,
        assignments=assignments,
        seed=f"{seed}:test",
        label_group_frequency=label_group_frequency,
    )
    if missing_test:
        raise RuntimeError(f"unable to cover test labels: {sorted(missing_test)}")

    # Validation also receives every label with enough independent groups to
    # support train/test/validation coverage.  Single-group labels are excluded.
    validation_required = {label for label, count in label_group_frequency.items() if count >= 3}
    missing_validation = seed_coverage(
        split="validation",
        required=validation_required,
        groups=groups,
        assignments=assignments,
        seed=f"{seed}:validation",
        label_group_frequency=label_group_frequency,
    )
    if missing_validation:
        raise RuntimeError(f"unable to cover validation labels: {sorted(missing_validation)}")

    split_rows = Counter()
    split_labels: Dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for key, split in assignments.items():
        split_rows[split] += groups[key]["rows"]
        split_labels[split].update(groups[key]["labels"])

    def placement_score(split: str, group: Mapping[str, Any]) -> tuple[float, int]:
        row_ratio = (split_rows[split] + group["rows"]) / target_rows[split]
        label_scores = {kind: [] for kind in LABEL_FIELDS}
        for label, count in group["labels"].items():
            kind = label.split(":", 1)[0]
            target = max(label_totals[label] * FRACTIONS[split], 1.0)
            label_scores[kind].append((split_labels[split][label] + count) / target)
        stratification = sum(
            sum(values) / len(values) for values in label_scores.values() if values
        )
        overshoot = max(0.0, row_ratio - 1.0)
        score = 3.0 * row_ratio + stratification + 25.0 * overshoot * overshoot
        return score, stable_rank(f"{seed}:{split}", str(group["indices"][0]))

    remaining = [key for key in groups if key not in assignments]
    remaining.sort(
        key=lambda key: (
            -groups[key]["rows"],
            -sum(1.0 / label_group_frequency[label] for label in groups[key]["labels"]),
            stable_rank(seed, key),
        )
    )
    for key in remaining:
        group = groups[key]
        split = min(SPLITS, key=lambda candidate: placement_score(candidate, group))
        assignments[key] = split
        split_rows[split] += group["rows"]
        split_labels[split].update(group["labels"])

    required_by_split = {
        "test": all_labels,
        "validation": validation_required,
        "train": {label for label, count in label_group_frequency.items() if count >= 2},
    }

    def protected(key: str, split: str) -> bool:
        return any(
            label in required_by_split[split]
            and split_labels[split][label] == groups[key]["labels"][label]
            for label in groups[key]["labels"]
        )

    def move_delta(key: str, source: str, destination: str) -> float:
        group = groups[key]
        delta = 0.0
        for label, count in group["labels"].items():
            target_source = max(label_totals[label] * FRACTIONS[source], 1.0)
            target_destination = max(label_totals[label] * FRACTIONS[destination], 1.0)
            before = abs(split_labels[source][label] - target_source) / target_source
            before += abs(split_labels[destination][label] - target_destination) / target_destination
            after = abs(split_labels[source][label] - count - target_source) / target_source
            after += abs(split_labels[destination][label] + count - target_destination) / target_destination
            delta += after - before
        return delta

    # Singleton-heavy public sources make exact row targets attainable.  Move
    # only whole, non-essential groups from overfull to underfull partitions.
    for _ in range(len(groups) * 2):
        over = [split for split in SPLITS if split_rows[split] > target_rows[split]]
        under = [split for split in SPLITS if split_rows[split] < target_rows[split]]
        if not over or not under:
            break
        source = max(over, key=lambda split: split_rows[split] - target_rows[split])
        destination = max(under, key=lambda split: target_rows[split] - split_rows[split])
        gap = min(
            split_rows[source] - target_rows[source],
            target_rows[destination] - split_rows[destination],
        )
        candidates = [
            key
            for key, assigned in assignments.items()
            if assigned == source and groups[key]["rows"] <= gap and not protected(key, source)
        ]
        if not candidates:
            break
        key = min(
            candidates,
            key=lambda item: (
                move_delta(item, source, destination),
                -groups[item]["rows"],
                stable_rank(f"{seed}:rebalance", item),
            ),
        )
        group = groups[key]
        assignments[key] = destination
        split_rows[source] -= group["rows"]
        split_rows[destination] += group["rows"]
        split_labels[source].subtract(group["labels"])
        split_labels[destination].update(group["labels"])

    missing = {
        split: {label for label in required if split_labels[split][label] <= 0}
        for split, required in required_by_split.items()
    }
    if any(missing.values()):
        raise RuntimeError(f"coverage invariant failed: {missing}")
    if sum(split_rows.values()) != total_rows:
        raise RuntimeError("row conservation invariant failed")
    return assignments, target_rows, required_by_split


def label_report(dataset: Dataset, field: str) -> Dict[str, int]:
    return dict(sorted(Counter(dataset[field]).items()))


def build_audit(
    *,
    input_dir: str,
    output_dir: str,
    dataset: DatasetDict,
    groups: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, str],
    target_rows: Mapping[str, int],
    required_by_split: Mapping[str, set[str]],
    input_digests: tuple[str, str] | None = None,
    output_digests: tuple[str, str] | None = None,
) -> Dict[str, Any]:
    coverage: Dict[str, Any] = {}
    for split in SPLITS:
        coverage[split] = {
            "rows": len(dataset[split]),
            "target_rows": target_rows[split],
            "row_delta": len(dataset[split]) - target_rows[split],
            "target_categories": label_report(dataset[split], "target_category"),
            "modalities": label_report(dataset[split], "modality"),
            "source_datasets": label_report(dataset[split], "source_dataset"),
        }

    group_sets = {
        split: {key for key, assigned in assignments.items() if assigned == split} for split in SPLITS
    }
    overlaps = {
        f"{left}_{right}": sorted(group_sets[left].intersection(group_sets[right]))
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }

    bare_sources: Dict[str, set[str]] = defaultdict(set)
    for key, group in groups.items():
        source = key.split("::", 1)[0]
        bare_sources[group["bare_group_id"]].add(source)
    bare_collisions = {key: sorted(values) for key, values in bare_sources.items() if len(values) > 1}

    missing_required = {}
    for split, required in required_by_split.items():
        observed = {
            f"{kind}:{value}"
            for kind, field in LABEL_FIELDS.items()
            for value in dataset[split].unique(field)
        }
        missing_required[split] = sorted(required - observed)

    split_manifest = hashlib.sha256(
        "\n".join(f"{key}\t{assignments[key]}" for key in sorted(assignments)).encode("utf-8")
    ).hexdigest()
    image_match = input_digests is None or input_digests[0] == output_digests[0]
    content_match = input_digests is None or input_digests[1] == output_digests[1]
    checks = {
        "row_total_is_25809": sum(len(dataset[split]) for split in SPLITS) == 25809,
        "test_has_all_14_target_categories": len(dataset["test"].unique("target_category")) == 14,
        "test_has_all_5_modalities": len(dataset["test"].unique("modality")) == 5,
        "test_has_all_8_sources": len(dataset["test"].unique("source_dataset")) == 8,
        "global_group_overlap_is_zero": not any(overlaps.values()),
        "required_coverage_complete": not any(missing_required.values()),
        "image_bytes_unchanged": image_match,
        "non_split_content_unchanged": content_match,
    }
    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "split_provenance": PROVENANCE,
        "fractions": FRACTIONS,
        "rows": sum(len(dataset[split]) for split in SPLITS),
        "groups": len(groups),
        "coverage": coverage,
        "missing_required_labels": missing_required,
        "global_group_overlaps": overlaps,
        "bare_group_id_cross_source_collision_count": len(bare_collisions),
        "bare_group_id_cross_source_collision_examples": dict(list(sorted(bare_collisions.items()))[:20]),
        "split_manifest_sha256": split_manifest,
        "input_image_manifest_sha256": input_digests[0] if input_digests else None,
        "output_image_manifest_sha256": output_digests[0] if output_digests else None,
        "input_non_split_content_sha256": input_digests[1] if input_digests else None,
        "output_non_split_content_sha256": output_digests[1] if output_digests else None,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "rare_category_note": (
            "right_atrium has one row in one MM-WHS-MRI group; test coverage therefore makes "
            "train/validation coverage mathematically impossible without leakage or duplication"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.num_proc < 1:
        raise ValueError("--num_proc must be positive")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.dry_run:
        raise FileExistsError(output_dir)

    source = load_from_disk(args.input_dir)
    missing_splits = set(SPLITS) - set(source)
    if missing_splits:
        raise ValueError(f"input is missing splits: {sorted(missing_splits)}")

    combined = concatenate_datasets(
        [source[split].cast_column("image", HFImage(decode=False)) for split in SPLITS]
    )
    metadata = combined.select_columns(
        ["id", "source_dataset", "group_id", "target_category", "modality"]
    )
    if len(set(metadata["id"])) != len(metadata):
        raise ValueError("id values must be globally unique")

    groups = build_groups(metadata)
    assignments, target_rows, required_by_split = assign_groups(groups, len(combined), args.seed)
    split_by_index = [""] * len(combined)
    global_group_by_index = [""] * len(combined)
    for key, split in assignments.items():
        for index in groups[key]["indices"]:
            split_by_index[index] = split
            global_group_by_index[index] = key
    if not all(split_by_index):
        raise RuntimeError("some rows were not assigned")

    indices = {
        split: [index for index, assigned in enumerate(split_by_index) if assigned == split]
        for split in SPLITS
    }

    if args.dry_run:
        preview = DatasetDict({split: metadata.select(indices[split]) for split in SPLITS})
        audit = build_audit(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            dataset=preview,
            groups=groups,
            assignments=assignments,
            target_rows=target_rows,
            required_by_split=required_by_split,
        )
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return

    input_digests = digest_records(combined)

    def update_row(row: Dict[str, Any], index: int) -> Dict[str, Any]:
        payload = json.loads(row["solution"])
        payload["split_provenance"] = PROVENANCE
        return {
            "solution": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            "split": split_by_index[index],
            "global_group_id": global_group_by_index[index],
        }

    rewritten = combined.map(
        update_row,
        with_indices=True,
        num_proc=args.num_proc,
        desc="updating split provenance",
        writer_batch_size=64,
    ).cast_column("image", HFImage())
    result = DatasetDict({split: rewritten.select(indices[split]) for split in SPLITS})
    result.save_to_disk(str(output_dir), max_shard_size="500MB")

    persisted = load_from_disk(str(output_dir))
    persisted_combined = concatenate_datasets(
        [persisted[split].cast_column("image", HFImage(decode=False)) for split in SPLITS]
    )
    output_digests = digest_records(persisted_combined)
    audit = build_audit(
        input_dir=args.input_dir,
        output_dir=str(output_dir),
        dataset=persisted,
        groups=groups,
        assignments=assignments,
        target_rows=target_rows,
        required_by_split=required_by_split,
        input_digests=input_digests,
        output_digests=output_digests,
    )
    (output_dir / "split_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
