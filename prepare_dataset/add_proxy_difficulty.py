#!/usr/bin/env python3
"""Add auditable three-axis difficulty proxies to a calibrated public snapshot.

The review-stage release omits the Qwen2.5-VL-72B scores described by the
paper. This creates explicit *proxy* fields from image texture, target geometry,
and referring-expression ambiguity without presenting them as author labels.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from datasets import DatasetDict, Image as HFImage, load_from_disk
from PIL import Image

from think_mris.reasoning import TaskAdaptiveReasoningController


PROVENANCE = "deterministic_visual_geometry_language_proxy_public_scores_absent"
_CONTROLLER = TaskAdaptiveReasoningController()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/workspace/datasets/MRIS-Bench-calibrated-25k")
    parser.add_argument(
        "--output_dir",
        default="/workspace/datasets/MRIS-Bench-calibrated-25k-paper-v2",
    )
    parser.add_argument("--num_proc", type=int, default=8)
    return parser.parse_args()


def decode_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, dict) and image.get("bytes") is not None:
        return Image.open(io.BytesIO(image["bytes"]))
    if isinstance(image, dict) and image.get("path"):
        return Image.open(image["path"])
    raise TypeError(f"unsupported image payload {type(image)!r}")


def image_scene_complexity(image: Any) -> float:
    image = decode_image(image)
    gray = np.asarray(image.convert("L").resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)
    histogram = np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / histogram.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum() / 8.0)
    contrast = float(np.std(gray) / 64.0)
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    gradient = float((dx.mean() + dy.mean()) / 64.0)
    score = 1.0 + 4.5 * min(entropy, 1.0) + 2.5 * min(contrast, 1.0) + 2.0 * min(gradient, 1.0)
    return max(1.0, min(10.0, score))


def enrich(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = json.loads(row["solution"])
    objects = payload.get("solution", [])
    base_metadata = dict(payload)
    base_metadata.pop("difficulty", None)
    base_metadata.pop("task_difficulty", None)
    base_metadata.pop("scene_complexity", None)
    base_metadata.pop("segmentation_challenge", None)
    base_metadata.pop("linguistic_ambiguity", None)
    base_metadata["difficulty_provenance"] = PROVENANCE

    ground_truth_without_difficulty = json.dumps(
        {**base_metadata, "solution": objects},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    inferred = _CONTROLLER.estimate_task_difficulty(
        ground_truth_without_difficulty,
        problem_text=row["problem"],
    )
    scene = image_scene_complexity(row["image"])
    segmentation = inferred.segmentation_challenge
    linguistic = inferred.linguistic_ambiguity
    difficulty = (scene + segmentation + linguistic) / 3.0

    payload.update(
        {
            "scene_complexity": round(scene, 6),
            "segmentation_challenge": round(segmentation, 6),
            "linguistic_ambiguity": round(linguistic, 6),
            "difficulty": round(difficulty, 6),
            "difficulty_provenance": PROVENANCE,
        }
    )
    return {
        "solution": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        "scene_complexity": payload["scene_complexity"],
        "segmentation_challenge": payload["segmentation_challenge"],
        "linguistic_ambiguity": payload["linguistic_ambiguity"],
        "difficulty": payload["difficulty"],
        "difficulty_provenance": PROVENANCE,
    }


def main() -> None:
    args = parse_args()
    if args.num_proc < 1:
        raise ValueError("--num_proc must be positive")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)

    source = load_from_disk(args.input_dir)
    enriched = DatasetDict(
        {
            split: dataset.cast_column("image", HFImage(decode=False))
            .map(
                enrich,
                num_proc=args.num_proc,
                desc=f"difficulty:{split}",
                writer_batch_size=32,
            )
            .cast_column("image", HFImage())
            for split, dataset in source.items()
        }
    )
    enriched.save_to_disk(str(output_dir), max_shard_size="500MB")

    values = [value for split in enriched.values() for value in split["difficulty"]]
    report = {
        "input_dir": args.input_dir,
        "output_dir": str(output_dir),
        "rows": sum(len(split) for split in enriched.values()),
        "splits": {name: len(split) for name, split in enriched.items()},
        "difficulty_provenance": PROVENANCE,
        "difficulty_min": min(values),
        "difficulty_mean": sum(values) / len(values),
        "difficulty_max": max(values),
        "easy_below_3_5": sum(value < 3.5 for value in values),
        "medium_3_5_to_5": sum(3.5 <= value < 5.0 for value in values),
        "hard_at_least_5": sum(value >= 5.0 for value in values),
        "warning": "Proxy scores are not the paper's withheld Qwen2.5-VL-72B annotations.",
    }
    (output_dir / "difficulty_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
