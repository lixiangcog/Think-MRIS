#!/usr/bin/env python3
"""Audit and calibrate the review-stage public MRIS-Bench release.

The public dataset omits source masks, split/source metadata, difficulty scores,
and the paper's second interior point. This tool never overwrites that release.
It removes objectively invalid text/geometry, rewrites referring expressions,
and can derive two explicitly *pseudo-labeled* points from a SAM2 mask prompted
by the legacy box and point. The audit report keeps this provenance visible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the public MRIS-Bench snapshot")
    parser.add_argument("--input-dir", default="/workspace/datasets/MRIS-Bench-original")
    parser.add_argument("--output-dir", default="/workspace/datasets/MRIS-Bench-calibrated-25k")
    parser.add_argument("--audit-dir", default="/workspace/datasets/MRIS-Bench-calibration-audit-25k")
    parser.add_argument("--geometry-policy", choices=("sam2-pseudo", "strict"), default="sam2-pseudo")
    parser.add_argument("--sam-model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument(
        "--devices",
        default="",
        help="comma-separated CUDA devices; defaults to every device visible to PyTorch",
    )
    parser.add_argument("--min-sam-bbox-iou", type=float, default=0.40)
    parser.add_argument("--max-problem-words", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _workspace_path(value: str) -> Path:
    path = Path(value).resolve()
    if not str(path).startswith("/workspace/"):
        raise ValueError(f"all dataset artifacts must remain under /workspace: {path}")
    return path


def _load_dataset(path: str, decode_images: bool):
    from datasets import Dataset, DatasetDict, Image, concatenate_datasets, load_from_disk

    loaded = load_from_disk(path)
    if isinstance(loaded, DatasetDict):
        parts = []
        for split, values in loaded.items():
            parts.append(values.add_column("original_split", [split] * len(values)))
        dataset = concatenate_datasets(parts)
    elif isinstance(loaded, Dataset):
        dataset = loaded.add_column("original_split", ["unavailable"] * len(loaded))
    else:
        raise TypeError(f"unsupported dataset object: {type(loaded)!r}")
    return dataset.cast_column("image", Image(decode=decode_images))


def _parse_solution(solution: str) -> Tuple[List[dict], Dict[str, Any]]:
    value = json.loads(solution)
    if isinstance(value, list):
        return value, {}
    if not isinstance(value, dict):
        raise ValueError("solution must be a JSON object or list")
    if "solution" in value:
        objects = value["solution"]
        if not isinstance(objects, list):
            raise ValueError("nested solution must be a list")
        return objects, value
    if "bbox_2d" in value:
        return [value], {}
    raise ValueError("solution contains no recognized geometry")


def _preaudit(args: argparse.Namespace, dataset) -> Tuple[List[dict], List[dict], Counter]:
    from think_mris.data_calibration import (
        hard_quality_reasons,
        infer_sample_metadata,
        quality_reasons,
        rewrite_problem,
        stable_sample_id,
        validate_legacy_geometry,
    )

    candidates: List[dict] = []
    rejected: List[dict] = []
    reason_counts: Counter = Counter()
    seen = set()
    limit = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    for index in range(limit):
        row = dataset[index]
        original_id = str(row.get("id", ""))
        original_problem = str(row.get("problem", "") or "").strip()
        metadata = infer_sample_metadata(original_id, original_problem)
        reasons: List[str] = []
        try:
            objects, solution_metadata = _parse_solution(row.get("solution", ""))
        except Exception as exc:
            objects, solution_metadata = [], {}
            reasons.append("invalid_solution_json_or_schema")
        if len(objects) != 1:
            reasons.append("solution_must_contain_one_target")
        geometry = objects[0] if len(objects) == 1 and isinstance(objects[0], dict) else {}
        bbox = geometry.get("bbox_2d")
        points = geometry.get("points_2d")
        legacy_point = geometry.get("point_2d")
        if isinstance(points, list) and len(points) == 2:
            legacy_point = points[0]
        reasons.extend(validate_legacy_geometry(bbox, legacy_point, row["img_width"], row["img_height"]))
        detected_quality_reasons = quality_reasons(
            original_problem, metadata, max_words=args.max_problem_words
        )
        hard_reasons = hard_quality_reasons(
            original_problem, metadata, max_words=args.max_problem_words
        )
        reasons.extend(hard_reasons)
        repair_reasons = sorted(set(detected_quality_reasons) - set(hard_reasons))

        if not reasons:
            repair_reasons = sorted(set(repair_reasons) | {"geometry_grounded_rewrite"})
            rewritten_problem, query_style = rewrite_problem(
                f"{index}:{original_id}:{bbox}",
                original_problem,
                metadata,
                bbox=bbox,
                image_width=int(row["img_width"]),
                image_height=int(row["img_height"]),
                use_original_description=False,
            )
            if len(rewritten_problem.split()) < 12:
                reasons.append("insufficient_supported_description")
                rewritten_problem, query_style = "", ""
        else:
            rewritten_problem, query_style = "", ""

        if not reasons:
            dedupe_key = (
                metadata.source_dataset,
                original_id,
                metadata.target_category,
                tuple(float(value) for value in bbox),
                rewritten_problem.lower(),
            )
            if dedupe_key in seen:
                reasons.append("exact_duplicate_target_record")
            seen.add(dedupe_key)

        if reasons:
            unique_reasons = sorted(set(reasons))
            reason_counts.update(unique_reasons)
            rejected.append(
                {
                    "original_index": index,
                    "original_id": original_id,
                    "reasons": unique_reasons,
                    "problem": original_problem,
                }
            )
            continue

        candidates.append(
            {
                "original_index": index,
                "original_id": original_id,
                "original_problem": original_problem,
                "problem": rewritten_problem,
                "query_style": query_style,
                "repair_reasons": repair_reasons,
                "bbox_2d": [int(round(value)) for value in bbox],
                "legacy_point": [int(round(value)) for value in legacy_point],
                "existing_points": points if isinstance(points, list) and len(points) == 2 else None,
                "img_width": int(row["img_width"]),
                "img_height": int(row["img_height"]),
                "source_dataset": metadata.source_dataset,
                "modality": metadata.modality,
                "target_category": metadata.target_category,
                "target_kind": metadata.target_kind,
                "group_id": metadata.group_id,
                "original_split": row["original_split"],
                "stable_id": stable_sample_id(metadata, original_id, bbox, rewritten_problem),
                "legacy_solution_metadata": solution_metadata,
            }
        )
    return candidates, rejected, reason_counts


def _sam_worker(
    input_dir: str,
    tasks: Sequence[dict],
    device: str,
    sam_model: str,
    minimum_bbox_iou: float,
    output_path: str,
) -> None:
    import numpy as np
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    from think_mris.data_calibration import bbox_iou, derive_two_interior_points, mask_bbox

    dataset = _load_dataset(input_dir, decode_images=True)
    torch.cuda.set_device(device)
    predictor = SAM2ImagePredictor.from_pretrained(sam_model, device=device)
    temporary = f"{output_path}.tmp-{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as stream:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for offset, task in enumerate(tasks):
                result: Dict[str, Any] = {"original_index": task["original_index"]}
                try:
                    row = dataset[task["original_index"]]
                    image = np.asarray(row["image"].convert("RGB")).copy()
                    bbox = np.asarray(task["bbox_2d"], dtype=np.float32)
                    point = np.asarray([task["legacy_point"]], dtype=np.float32)
                    predictor.set_image(image)
                    masks, scores, _ = predictor.predict(
                        point_coords=point,
                        point_labels=np.ones(1, dtype=np.int32),
                        box=bbox,
                        multimask_output=True,
                    )
                    ranked = []
                    for candidate_index, (mask, score) in enumerate(zip(masks, scores)):
                        candidate_bbox = mask_bbox(mask)
                        overlap = bbox_iou(candidate_bbox, bbox) if candidate_bbox is not None else 0.0
                        ranked.append((overlap + 0.25 * max(float(score), 0.0), overlap, float(score), candidate_index))
                    _, overlap, score, candidate_index = max(ranked)
                    if overlap < minimum_bbox_iou:
                        raise ValueError(f"sam_bbox_iou_below_threshold:{overlap:.6f}")
                    points = derive_two_interior_points(masks[candidate_index], bbox)
                    if any(not bool(masks[candidate_index][point_y, point_x]) for point_x, point_y in points):
                        raise ValueError("derived_point_outside_sam_mask")
                    result.update(
                        {
                            "points_2d": points,
                            "sam_score": float(score),
                            "sam_bbox_iou": float(overlap),
                        }
                    )
                except Exception as exc:
                    message = str(exc).splitlines()[0][:240]
                    result["geometry_error"] = message or type(exc).__name__
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                if (offset + 1) % 500 == 0:
                    print(f"[{device}] processed {offset + 1}/{len(tasks)}", flush=True)
    os.replace(temporary, output_path)


def _run_geometry(args: argparse.Namespace, candidates: List[dict], audit_dir: Path) -> Dict[int, dict]:
    if args.geometry_policy == "strict":
        return {
            item["original_index"]: {
                "original_index": item["original_index"],
                "points_2d": item["existing_points"],
                "sam_score": None,
                "sam_bbox_iou": None,
            }
            for item in candidates
            if item["existing_points"] is not None
        }

    import torch

    visible_device_count = torch.cuda.device_count()
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        devices = [f"cuda:{index}" for index in range(visible_device_count)]
    if not devices:
        raise ValueError("SAM2 calibration requires at least one CUDA device visible to PyTorch")
    for device in devices:
        match = re.fullmatch(r"cuda:(\d+)", device)
        if match is None or int(match.group(1)) >= visible_device_count:
            raise ValueError(
                f"requested {device}, but PyTorch sees {visible_device_count} CUDA device(s); "
                "check CUDA_VISIBLE_DEVICES or omit --devices for automatic selection"
            )
    signature_payload = json.dumps(
        {
            "indices": [item["original_index"] for item in candidates],
            "model": args.sam_model,
            "threshold": args.min_sam_bbox_iou,
        },
        sort_keys=True,
    )
    signature = hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()[:12]
    part_dir = audit_dir / "geometry_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    shards = [[] for _ in devices]
    for offset, item in enumerate(candidates):
        shards[offset % len(devices)].append(item)
    processes = []
    part_paths = []
    context = mp.get_context("spawn")
    for rank, (device, tasks) in enumerate(zip(devices, shards)):
        part_path = part_dir / f"{signature}-part-{rank:02d}.jsonl"
        part_paths.append(part_path)
        if part_path.exists():
            print(f"Reusing completed geometry shard: {part_path}")
            continue
        process = context.Process(
            target=_sam_worker,
            args=(
                args.input_dir,
                tasks,
                device,
                args.sam_model,
                args.min_sam_bbox_iou,
                str(part_path),
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"SAM2 geometry worker failed with exit code {process.exitcode}")
    results: Dict[int, dict] = {}
    for part_path in part_paths:
        with part_path.open(encoding="utf-8") as stream:
            for line in stream:
                result = json.loads(line)
                results[int(result["original_index"])] = result
    return results


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_report(
    original_count: int,
    candidates: Sequence[dict],
    accepted: Sequence[dict],
    rejected: Sequence[dict],
    reason_counts: Counter,
) -> Dict[str, Any]:
    from think_mris.data_calibration import count_direct_target_mentions

    return {
        "original_rows": original_count,
        "text_and_schema_candidates": len(candidates),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "retention_rate": len(accepted) / original_count if original_count else 0.0,
        "rejection_reasons": dict(reason_counts.most_common()),
        "repaired_rows": sum(bool(item["repair_reasons"]) for item in accepted),
        "repair_reasons": dict(
            Counter(reason for item in accepted for reason in item["repair_reasons"]).most_common()
        ),
        "sources_inferred": dict(Counter(item["source_dataset"] for item in accepted)),
        "modalities": dict(Counter(item["modality"] for item in accepted)),
        "target_categories": dict(Counter(item["target_category"] for item in accepted)),
        "query_styles": dict(Counter(item["query_style"] for item in accepted)),
        "question_queries": sum(item["problem"].endswith("?") for item in accepted),
        "imperative_queries": sum(item["problem"].endswith(".") for item in accepted),
        "direct_target_mentions_after_rewrite": count_direct_target_mentions(
            (item["target_category"], item["problem"]) for item in accepted
        ),
        "paper_alignment": {
            "paper_unique_sources": 22,
            "public_release_explicit_source_column": False,
            "public_release_mask_column": False,
            "public_release_difficulty_columns": False,
            "public_release_split_metadata": False,
            "public_release_geometry": "one bbox plus one point",
            "calibrated_geometry": "one bbox plus two SAM2-pseudo-mask-derived points",
            "geometry_warning": "Pseudo masks are not source-provided ground truth and are never represented as such.",
            "split_warning": "Official source splits were absent; deterministic group-hash 80/10/10 splits are provided for leakage-safe development only.",
        },
    }


def _write_markdown_report(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# MRIS-Bench calibration audit",
        "",
        "## Outcome",
        "",
        f"- Original rows: {report['original_rows']:,}",
        f"- Accepted rows: {report['accepted_rows']:,}",
        f"- Rejected rows: {report['rejected_rows']:,}",
        f"- Retention rate: {report['retention_rate']:.2%}",
        f"- Question-form queries: {report['question_queries']:,}",
        f"- Imperative queries: {report['imperative_queries']:,}",
        f"- Direct target-name leaks after rewriting: {report['direct_target_mentions_after_rewrite']:,}",
        f"- Repaired and retained rows: {report['repaired_rows']:,}",
        "",
        "## Rejection reasons",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {reason} | {count:,} |" for reason, count in report["rejection_reasons"].items())
    lines.extend(
        [
            "",
            "## Repaired issues retained in the dataset",
            "",
            "| Repair | Count |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| {reason} | {count:,} |" for reason, count in report["repair_reasons"].items())
    lines.extend(
        [
            "",
            "## Paper-alignment limitations in the public release",
            "",
            "The paper describes 22 sources, source masks, two mask-derived points, difficulty scores, and source-aware splits. "
            "The review-stage Hugging Face snapshot exposes none of the source, modality, target, volume, mask, difficulty, or split columns and stores only one point. "
            "This calibration recovers source groups only where identifiers support the inference, creates group-safe development splits, and labels SAM2-derived geometry as pseudo-labels.",
            "",
            "The absent source masks mean this artifact is suitable for code bring-up and controlled data iteration, not for claiming exact paper-dataset or benchmark reproduction.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_dataset(args: argparse.Namespace, dataset, accepted: List[dict], output_dir: Path) -> Dict[str, int]:
    if not accepted:
        raise RuntimeError("no samples passed both text/schema and geometry calibration")

    from datasets import DatasetDict, Image

    from think_mris.data_calibration import assign_group_split

    indices = [item["original_index"] for item in accepted]
    selected = dataset.select(indices)
    selected = selected.remove_columns([name for name in selected.column_names if name not in {"image"}])
    columns: Dict[str, Any] = {
        "id": [],
        "problem": [],
        "solution": [],
        "image": None,
        "img_height": [],
        "img_width": [],
        "original_id": [],
        "original_problem": [],
        "source_dataset": [],
        "modality": [],
        "target_category": [],
        "group_id": [],
        "split": [],
        "query_style": [],
        "geometry_provenance": [],
        "repair_reasons": [],
        "sam_score": [],
        "sam_bbox_iou": [],
    }
    for item in accepted:
        split = assign_group_split(item["source_dataset"], item["group_id"])
        payload = {
            "solution": [
                {
                    "bbox_2d": item["bbox_2d"],
                    "points_2d": item["points_2d"],
                }
            ],
            "img_width": item["img_width"],
            "img_height": item["img_height"],
            "source_dataset": item["source_dataset"],
            "modality": item["modality"],
            "target_category": item["target_category"],
            "geometry_provenance": item["geometry_provenance"],
            "difficulty_provenance": "runtime_heuristic_because_public_scores_are_absent",
            "split_provenance": "deterministic_group_hash_80_10_10_because_official_splits_are_absent",
        }
        columns["id"].append(item["stable_id"])
        columns["problem"].append(item["problem"])
        columns["solution"].append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        columns["img_height"].append(item["img_height"])
        columns["img_width"].append(item["img_width"])
        columns["original_id"].append(item["original_id"])
        columns["original_problem"].append(item["original_problem"])
        columns["source_dataset"].append(item["source_dataset"])
        columns["modality"].append(item["modality"])
        columns["target_category"].append(item["target_category"])
        columns["group_id"].append(item["group_id"])
        columns["split"].append(split)
        columns["query_style"].append(item["query_style"])
        columns["geometry_provenance"].append(item["geometry_provenance"])
        columns["repair_reasons"].append(item["repair_reasons"])
        columns["sam_score"].append(item["sam_score"])
        columns["sam_bbox_iou"].append(item["sam_bbox_iou"])

    curated = selected
    for name, values in columns.items():
        if name != "image":
            curated = curated.add_column(name, values)
    order = list(columns)
    curated = curated.select_columns(order)
    split_counts = Counter(columns["split"])
    dataset_dict = DatasetDict(
        {
            split: curated.select([index for index, value in enumerate(columns["split"]) if value == split]).cast_column(
                "image", Image()
            )
            for split in ("train", "validation", "test")
        }
    )
    dataset_dict.save_to_disk(output_dir)
    return dict(split_counts)


def main() -> None:
    args = parse_args()
    input_dir = _workspace_path(args.input_dir)
    output_dir = _workspace_path(args.output_dir)
    audit_dir = _workspace_path(args.audit_dir)
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    if output_dir.exists() and not args.dry_run and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output_dir}; pass --overwrite to replace it")
    audit_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(str(input_dir), decode_images=False)
    candidates, rejected, reason_counts = _preaudit(args, dataset)
    audited_count = min(len(dataset), args.max_samples or len(dataset))
    print(f"Pre-audit retained {len(candidates):,}/{audited_count:,} rows")
    if args.dry_run:
        report = _build_report(audited_count, candidates, candidates, rejected, reason_counts)
        report["dry_run"] = True
        _write_jsonl(audit_dir / "rejected_samples.jsonl", rejected)
        (audit_dir / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _write_markdown_report(audit_dir / "AUDIT.md", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    geometry = _run_geometry(args, candidates, audit_dir)
    accepted = []
    for item in candidates:
        result = geometry.get(item["original_index"])
        if result is None or result.get("points_2d") is None:
            reason = "missing_paper_geometry" if args.geometry_policy == "strict" else "sam2_geometry_failure"
            geometry_error = result.get("geometry_error", "") if result else "missing_geometry_result"
            if geometry_error.startswith("sam_bbox_iou_below_threshold"):
                reason = "sam2_bbox_mismatch"
            reason_counts[reason] += 1
            rejected.append(
                {
                    "original_index": item["original_index"],
                    "original_id": item["original_id"],
                    "reasons": [reason],
                    "geometry_error": geometry_error,
                    "problem": item["original_problem"],
                }
            )
            continue
        enriched = dict(item)
        enriched.update(result)
        enriched["geometry_provenance"] = (
            "source_provided_two_points" if item["existing_points"] is not None else "sam2_pseudo_mask_from_legacy_box_point"
        )
        accepted.append(enriched)

    report = _build_report(audited_count, candidates, accepted, rejected, reason_counts)
    _write_jsonl(audit_dir / "rejected_samples.jsonl", rejected)
    (audit_dir / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown_report(audit_dir / "AUDIT.md", report)
    if output_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(output_dir)
    split_counts = _save_dataset(args, dataset, accepted, output_dir)
    report["splits"] = split_counts
    (audit_dir / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown_report(audit_dir / "AUDIT.md", report)
    print(f"Saved calibrated dataset to {output_dir}")
    print(f"Saved audit artifacts to {audit_dir}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
