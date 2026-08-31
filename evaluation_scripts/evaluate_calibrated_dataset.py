"""Run resumable Think-MRIS inference on a calibrated local DatasetDict.

The released evaluation scripts target the legacy ReasonSeg schema.  This
entry point uses the same prompt and image resizing path as training, evaluates
the sparse geometry emitted by the reasoning model, and can optionally execute
SAM2 for every valid prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List, Sequence


_CACHE_ROOT = Path(os.environ.get("THINK_MRIS_CACHE_ROOT", "/workspace/.cache"))
os.environ["HF_HOME"] = os.environ.get("THINK_MRIS_HF_HOME", str(_CACHE_ROOT / "huggingface"))
os.environ["HF_HUB_CACHE"] = os.environ.get(
    "THINK_MRIS_HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub")
)
os.environ["TRANSFORMERS_CACHE"] = os.environ.get(
    "THINK_MRIS_TRANSFORMERS_CACHE", os.environ["HF_HUB_CACHE"]
)

import numpy as np
import torch
from datasets import load_from_disk
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from think_mris.modeling import build_vlki_priors
from think_mris.prompting import ThinkMRISPromptTemplate
from think_mris.utils import extract_think_answer, scale_answer_objects, validate_answer_objects
from verl.utils.rl_dataset import image_processor_size_kwargs, process_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Think-MRIS on the calibrated local dataset")
    parser.add_argument("--reasoning_model_path", required=True)
    parser.add_argument("--processor_path", default=None)
    parser.add_argument(
        "--test_data_path",
        default="/workspace/datasets/MRIS-Bench-calibrated-25k-paper-v4-final",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--max_side_length", type=int, default=640)
    parser.add_argument("--max_pixels", type=int, default=409600)
    parser.add_argument("--min_pixels", type=int, default=3136)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn_implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_sam2", action="store_true")
    parser.add_argument(
        "--disable_vlki",
        action="store_true",
        help="Run sparse box/point SAM2 without the paper's VLKI dense prompt.",
    )
    parser.add_argument("--segmentation_model_path", default="facebook/sam2-hiera-large")
    return parser.parse_args()


def load_solution(solution: str) -> Dict[str, Any]:
    payload = json.loads(solution)
    objects = payload.get("solution") if isinstance(payload, dict) else payload
    if not isinstance(objects, list) or len(objects) != 1:
        raise ValueError("calibrated solutions must contain exactly one target")
    validate_answer_objects(objects)
    return {"objects": objects, "metadata": payload if isinstance(payload, dict) else {}}


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def point_inside_bbox(point: Sequence[float], bbox: Sequence[float]) -> bool:
    return float(bbox[0]) <= float(point[0]) <= float(bbox[2]) and float(bbox[1]) <= float(point[1]) <= float(bbox[3])


def point_inside_mask(point: Sequence[float], mask: np.ndarray) -> bool:
    x = min(max(int(round(float(point[0]))), 0), mask.shape[1] - 1)
    y = min(max(int(round(float(point[1]))), 0), mask.shape[0] - 1)
    return bool(mask[y, x])


def binary_mask_metrics(predicted: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    predicted = predicted.astype(bool, copy=False)
    reference = reference.astype(bool, copy=False)
    intersection = int(np.logical_and(predicted, reference).sum())
    predicted_area = int(predicted.sum())
    reference_area = int(reference.sum())
    union = predicted_area + reference_area - intersection
    return {
        "proxy_mask_iou": intersection / union if union else 1.0,
        "proxy_mask_dice": (2 * intersection) / (predicted_area + reference_area)
        if predicted_area + reference_area
        else 1.0,
    }


def normalized_center_distance(
    first: Sequence[float], second: Sequence[float], width: int, height: int
) -> float:
    first_center = ((float(first[0]) + float(first[2])) / 2, (float(first[1]) + float(first[3])) / 2)
    second_center = ((float(second[0]) + float(second[2])) / 2, (float(second[1]) + float(second[3])) / 2)
    diagonal = math.hypot(width, height)
    return math.dist(first_center, second_center) / diagonal if diagonal else 0.0


def read_completed_indices(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    completed = set()
    with output_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                completed.add(int(json.loads(line)["dataset_index"]))
    return completed


def batches(values: Sequence[int], batch_size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def make_messages(row: Dict[str, Any], model_image, prompt_template: ThinkMRISPromptTemplate):
    return [
        {"role": "system", "content": prompt_template.system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": model_image},
                {
                    "type": "text",
                    "text": prompt_template.format_user_prompt(row["problem"], include_image_token=False),
                },
            ],
        },
    ]


def _predict_combined_sam2_mask(
    predictor,
    objects: Sequence[Dict[str, Any]],
    image_shape: tuple[int, int],
    dense_prior: torch.Tensor | None = None,
) -> tuple[np.ndarray, List[float]]:
    combined = np.zeros(image_shape, dtype=bool)
    best_scores = []
    mask_input = None if dense_prior is None else dense_prior[0].float().cpu().numpy()
    for item in objects:
        prediction_kwargs = {
            "point_coords": np.asarray(item["points_2d"], dtype=np.float32),
            "point_labels": np.ones(len(item["points_2d"]), dtype=np.int32),
            "box": np.asarray(item["bbox_2d"], dtype=np.float32),
            "multimask_output": True,
        }
        if mask_input is not None:
            prediction_kwargs["mask_input"] = mask_input
        masks, scores, _ = predictor.predict(**prediction_kwargs)
        best = int(np.argmax(scores))
        combined |= masks[best].astype(bool)
        best_scores.append(float(scores[best]))
    return combined, best_scores


def run_sam2_prediction(
    predictor,
    image,
    predicted_objects: Sequence[Dict[str, Any]],
    reference_objects: Sequence[Dict[str, Any]],
    dense_prior: torch.Tensor | None = None,
) -> Dict[str, Any]:
    predictor.set_image(np.asarray(image.convert("RGB")))
    combined, best_scores = _predict_combined_sam2_mask(
        predictor,
        predicted_objects,
        (image.height, image.width),
        dense_prior=dense_prior,
    )
    reference_mask, _ = _predict_combined_sam2_mask(
        predictor,
        reference_objects,
        (image.height, image.width),
    )
    result = {
        "sam2_success": True,
        "vlki_success": dense_prior is not None,
        "sam2_mask_area": int(combined.sum()),
        "sam2_mask_fraction": float(combined.mean()),
        "sam2_score": fmean(best_scores),
        "points_inside_proxy_mask": fmean(
            float(point_inside_mask(point, reference_mask))
            for item in predicted_objects
            for point in item["points_2d"]
        ),
    }
    result.update(binary_mask_metrics(combined, reference_mask))
    return result


def summarize(
    records: Sequence[Dict[str, Any]],
    dataset_rows: int,
    include_category_breakdown: bool = True,
) -> Dict[str, Any]:
    count = len(records)
    parsed = [record for record in records if record["parse_success"]]
    sam_records = [record for record in records if record.get("sam2_success")]
    reasoning_records = [record for record in records if record.get("reasoning_token_count") is not None]
    response_records = [record for record in records if record.get("response_token_count") is not None]
    result = {
        "dataset_rows": dataset_rows,
        "evaluated_rows": count,
        "parse_success_count": len(parsed),
        "parse_success_rate": len(parsed) / count if count else 0.0,
        "mean_bbox_iou": fmean(record["bbox_iou"] for record in records) if count else 0.0,
        "bbox_accuracy_iou_gt_0_5": (
            sum(record["bbox_iou"] > 0.5 for record in records) / count if count else 0.0
        ),
        # Kept as a compatibility alias for earlier local reports.
        "bbox_iou_at_0_5": sum(record["bbox_iou"] > 0.5 for record in records) / count
        if count
        else 0.0,
        "mean_points_inside_gt_bbox": (
            fmean(record["points_inside_gt_bbox"] for record in records) if count else 0.0
        ),
        "mean_normalized_center_distance": (
            fmean(record["normalized_center_distance"] for record in records) if count else 0.0
        ),
        "sam2_success_count": len(sam_records),
        "sam2_success_rate": len(sam_records) / count if count else 0.0,
        "mean_sam2_score": fmean(record["sam2_score"] for record in sam_records) if sam_records else None,
        "vlki_success_rate": (
            sum(bool(record.get("vlki_success")) for record in records) / count if count else 0.0
        ),
        "mean_proxy_mask_iou_all_rows": (
            fmean(float(record.get("proxy_mask_iou", 0.0)) for record in records) if count else 0.0
        ),
        "mean_proxy_mask_dice_all_rows": (
            fmean(float(record.get("proxy_mask_dice", 0.0)) for record in records) if count else 0.0
        ),
        "mean_points_inside_proxy_mask_all_rows": (
            fmean(float(record.get("points_inside_proxy_mask", 0.0)) for record in records)
            if count
            else 0.0
        ),
        "mean_reasoning_tokens": (
            fmean(record["reasoning_token_count"] for record in reasoning_records)
            if reasoning_records
            else None
        ),
        "reasoning_token_coverage": len(reasoning_records) / count if count else 0.0,
        "mean_response_tokens": (
            fmean(record["response_token_count"] for record in response_records)
            if response_records
            else None
        ),
        "paper_test_references": {
            "dice": 0.8145,
            "bbox_accuracy_iou_gt_0_5": 0.9012,
            "point_accuracy_in_gt_mask": 0.9245,
            "mean_reasoning_tokens": 48.36,
        },
        "metric_provenance": {
            "bbox_accuracy_iou_gt_0_5": "paper definition; current public snapshot uses reconstructed development splits",
            "mean_points_inside_gt_bbox": "box-only diagnostic; not the paper's point accuracy",
            "mean_points_inside_proxy_mask_all_rows": "SAM2 reference-mask proxy; source ground-truth masks are absent",
            "mean_proxy_mask_iou_all_rows": "SAM2-vs-SAM2 proxy; not paper-comparable segmentation IoU",
            "mean_proxy_mask_dice_all_rows": "SAM2-vs-SAM2 proxy; not paper-comparable Dice",
            "mean_reasoning_tokens": "paper-aligned tokenizer count inside parsed <think> tags",
        },
    }
    if include_category_breakdown:
        result["paper_reference_delta"] = {
            "bbox_accuracy_iou_gt_0_5": result["bbox_accuracy_iou_gt_0_5"] - 0.9012,
            "mean_reasoning_tokens": (
                result["mean_reasoning_tokens"] - 48.36
                if result["mean_reasoning_tokens"] is not None
                else None
            ),
            "dice": None,
            "point_accuracy_in_gt_mask": None,
            "note": "Dice and GT-mask point deltas are unavailable because source masks are absent.",
        }
        categories: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            categories.setdefault(str(record.get("target_category", "unknown")), []).append(record)
        result["by_target_category"] = {
            category: summarize(group, dataset_rows=len(group), include_category_breakdown=False)
            for category, group in sorted(categories.items())
        }
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "run_config.json"
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"{output_path} already exists; pass --resume or choose another output directory")

    dataset_dict = load_from_disk(args.test_data_path)
    if args.split not in dataset_dict:
        raise KeyError(f"split {args.split!r} is absent from {args.test_data_path}")
    dataset = dataset_dict[args.split]
    stop_index = len(dataset) if args.limit is None else min(len(dataset), args.start_index + args.limit)
    completed_indices = read_completed_indices(output_path) if args.resume else set()
    indices = [
        index
        for index in range(args.start_index, stop_index)
        if index not in completed_indices
    ]

    run_config = vars(args).copy()
    run_config.update(
        {
            "dataset_rows": len(dataset),
            "selected_rows": stop_index - args.start_index,
            "already_completed_rows": len(completed_indices),
            "remaining_rows": len(indices),
        }
    )
    config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        device_map={"": str(device)},
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.processor_path or args.reasoning_model_path,
        padding_side="left",
    )
    prompt_template = ThinkMRISPromptTemplate()

    predictor = None
    if args.run_sam2:
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        predictor = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path, device=str(device))

    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8", buffering=1) as output_stream:
        for batch_number, batch_indices in enumerate(batches(indices, args.batch_size), start=1):
            rows = [dataset[index] for index in batch_indices]
            original_images = [row["image"].convert("RGB") for row in rows]
            model_images = [
                process_image(
                    image,
                    max_pixels=args.max_pixels,
                    min_pixels=args.min_pixels,
                    max_side_length=args.max_side_length,
                )
                for image in original_images
            ]
            messages = [
                make_messages(row, model_image, prompt_template)
                for row, model_image in zip(rows, model_images)
            ]
            prompt_texts = [
                processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                for message in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=prompt_texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **image_processor_size_kwargs(args.min_pixels, args.max_pixels),
            ).to(device)

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    use_cache=True,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            response_ids = generated_ids[:, inputs.input_ids.shape[1] :]
            outputs = processor.batch_decode(
                response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            dense_priors: List[torch.Tensor | None] = [None] * len(outputs)
            if predictor is not None and not args.disable_vlki:
                dense_priors = build_vlki_priors(
                    model,
                    processor,
                    inputs,
                    generated_ids,
                    allow_missing=True,
                )

            for index, row, original_image, model_image, output_text, dense_prior in zip(
                batch_indices, rows, original_images, model_images, outputs, dense_priors
            ):
                truth = load_solution(row["solution"])
                truth_object = truth["objects"][0]
                record: Dict[str, Any] = {
                    "dataset_index": index,
                    "original_id": row["original_id"],
                    "source_dataset": row["source_dataset"],
                    "modality": row["modality"],
                    "target_category": row["target_category"],
                    "problem": row["problem"],
                    "output_text": output_text,
                    "response_token_count": len(
                        processor.tokenizer(output_text, add_special_tokens=False).input_ids
                    ),
                    "reasoning_token_count": None,
                    "parse_success": False,
                    "bbox_iou": 0.0,
                    "points_inside_gt_bbox": 0.0,
                    "normalized_center_distance": 1.0,
                    "sam2_success": False,
                    "vlki_success": False,
                }
                try:
                    think_text, answer_objects = extract_think_answer(output_text)
                    validate_answer_objects(answer_objects)
                    if len(answer_objects) != 1:
                        raise ValueError(f"expected one target, received {len(answer_objects)}")
                    predicted_objects = scale_answer_objects(
                        answer_objects,
                        model_image.size,
                        original_image.size,
                    )
                    predicted_object = predicted_objects[0]
                    record.update(
                        {
                            "parse_success": True,
                            "think": think_text,
                            "reasoning_token_count": len(
                                processor.tokenizer(think_text, add_special_tokens=False).input_ids
                            ),
                            "predicted_objects": predicted_objects,
                            "ground_truth_objects": truth["objects"],
                            "bbox_iou": bbox_iou(predicted_object["bbox_2d"], truth_object["bbox_2d"]),
                            "points_inside_gt_bbox": fmean(
                                float(point_inside_bbox(point, truth_object["bbox_2d"]))
                                for point in predicted_object["points_2d"]
                            ),
                            "normalized_center_distance": normalized_center_distance(
                                predicted_object["bbox_2d"],
                                truth_object["bbox_2d"],
                                original_image.width,
                                original_image.height,
                            ),
                        }
                    )
                    if predictor is not None:
                        record.update(
                            run_sam2_prediction(
                                predictor,
                                original_image,
                                predicted_objects,
                                truth["objects"],
                                dense_prior=dense_prior,
                            )
                        )
                except Exception as error:
                    record["parse_error"] = f"{type(error).__name__}: {error}"
                output_stream.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(
                f"batch={batch_number} completed={min(batch_number * args.batch_size, len(indices))}/{len(indices)}",
                flush=True,
            )
            del inputs, generated_ids, response_ids
            if device.type == "cuda":
                torch.cuda.empty_cache()

    records = []
    with output_path.open("r", encoding="utf-8") as stream:
        records.extend(json.loads(line) for line in stream if line.strip())
    summary = summarize(records, dataset_rows=len(dataset))
    summary.update(
        {
            "model_path": args.reasoning_model_path,
            "processor_path": args.processor_path or args.reasoning_model_path,
            "dataset_path": args.test_data_path,
            "split": args.split,
            "predictions_path": str(output_path),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
