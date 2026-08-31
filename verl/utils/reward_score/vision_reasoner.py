"""Legacy VisionReasoner reward kept for backward-compatible experiments."""

import json
import re
from typing import List, Sequence


def _objects(text: str) -> List[dict]:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if not match:
        return []
    try:
        payload = json.loads(match.group(1))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def vision_reasoner_format_reward(predict_str: str) -> float:
    response_format = re.fullmatch(r"<think>.*?</think>\s*<answer>.*?</answer>", predict_str.strip(), re.DOTALL)
    objects = _objects(predict_str)
    geometry_format = bool(objects) and all(
        isinstance(item.get("bbox_2d"), list)
        and len(item["bbox_2d"]) == 4
        and (
            isinstance(item.get("point_2d"), list)
            or isinstance(item.get("points_2d"), list)
        )
        for item in objects
    )
    return float(bool(response_format)) + float(geometry_format)


def vision_reasoner_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    try:
        targets = json.loads(ground_truth)
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0
    if isinstance(targets, dict):
        targets = targets.get("solution", [])
    predicted = _objects(predict_str)
    if not predicted or not isinstance(targets, list) or not targets:
        return 0.0
    count = min(len(predicted), len(targets))
    return sum(_iou(predicted[index]["bbox_2d"], targets[index]["bbox_2d"]) for index in range(count)) / max(
        len(predicted), len(targets)
    )


def vision_reasoner_non_repeat_reward(predict_str: str) -> float:
    sentences = [sentence.strip() for sentence in predict_str.split(".") if sentence.strip()]
    return 1.0 if len(sentences) == len(set(sentences)) else 0.0


def vision_reasoner_compute_score(predict_str: str, ground_truth: str) -> float:
    return (
        vision_reasoner_format_reward(predict_str)
        + vision_reasoner_accuracy_reward(predict_str, ground_truth)
        + vision_reasoner_non_repeat_reward(predict_str)
    )
