import json
import math
import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from think_mris.reasoning import TaskAdaptiveReasoningController


_controller = TaskAdaptiveReasoningController()
_FORMAT_PATTERN = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)


def _answer_objects(text: str) -> List[Dict[str, Any]]:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if not match:
        return []
    try:
        value = json.loads(match.group(1))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def _ground_truth_payload(ground_truth: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    try:
        value = json.loads(ground_truth)
    except (TypeError, ValueError, json.JSONDecodeError):
        return [], {}
    if isinstance(value, list):
        return value, {}
    if not isinstance(value, dict):
        return [], {}
    solution = value.get("solution", value.get("target", value.get("answer", [])))
    if isinstance(solution, str):
        try:
            solution = json.loads(solution)
        except (TypeError, ValueError, json.JSONDecodeError):
            solution = []
    return (solution if isinstance(solution, list) else []), value


def _bbox(item: Dict[str, Any]) -> List[float]:
    value = item.get("bbox_2d", item.get("bbox", []))
    if not isinstance(value, list) or len(value) != 4:
        return []
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return []


def _points(item: Dict[str, Any]) -> List[List[float]]:
    value = item.get("points_2d", item.get("points", []))
    if not value and "point_2d" in item:
        # Legacy public data contains one point per object. It can still receive
        # a geometric reward, but it does not pass the canonical format reward.
        value = [item["point_2d"]]
    if not isinstance(value, list):
        return []
    points = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            continue
        try:
            points.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            continue
    return points[:2]


def _valid_box(box: Sequence[float]) -> float:
    return float(len(box) == 4 and all(math.isfinite(v) for v in box) and box[0] < box[2] and box[1] < box[3])


def _points_inside_box(points: Sequence[Sequence[float]], box: Sequence[float]) -> float:
    if not points or not _valid_box(box):
        return 0.0
    inside = sum(box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3] for p in points)
    return inside / len(points)


def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    if not _valid_box(box_a) or not _valid_box(box_b):
        return 0.0
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _s_log(value: float, k: float = 3.0) -> float:
    value = max(0.0, min(1.0, value))
    return math.log(k * value + 1.0) / math.log(k + 1.0)


def _s_exp(distance: float, k: float = 3.0, center: float = 1.0) -> float:
    exponent = max(-60.0, min(60.0, k * (distance - center)))
    return 1.0 / (1.0 + math.exp(exponent))


def _modulate(value: float, box_validity: float, point_validity: float, lam: float = 0.7) -> float:
    return lam * value + (1.0 - lam) * value * (box_validity + point_validity) / 2.0


def _frame_size(metadata: Dict[str, Any], gt_box: Sequence[float]) -> Tuple[float, float]:
    width = float(metadata.get("img_width", metadata.get("width", 0)) or 0)
    height = float(metadata.get("img_height", metadata.get("height", 0)) or 0)
    if width <= 0:
        width = max(float(gt_box[2]) if len(gt_box) == 4 else 1.0, 1.0)
    if height <= 0:
        height = max(float(gt_box[3]) if len(gt_box) == 4 else 1.0, 1.0)
    return width, height


def _point_alignment_distance(
    predicted: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
    width: float,
    height: float,
) -> float:
    if not predicted or not target:
        return 2.0

    def distance(order: Sequence[int]) -> float:
        count = min(len(predicted), len(order))
        return sum(
            abs(predicted[i][0] / width - target[order[i]][0] / width)
            + abs(predicted[i][1] / height - target[order[i]][1] / height)
            for i in range(count)
        ) / max(count, 1)

    if len(predicted) >= 2 and len(target) >= 2:
        return min(distance((0, 1)), distance((1, 0)))
    return distance((0,))


def _pair_accuracy(predicted: Dict[str, Any], target: Dict[str, Any], metadata: Dict[str, Any]) -> float:
    pred_box, gt_box = _bbox(predicted), _bbox(target)
    pred_points, gt_points = _points(predicted), _points(target)
    if not gt_box:
        return 0.0

    box_validity = _valid_box(pred_box)
    point_validity = _points_inside_box(pred_points, pred_box)
    iou_score = _s_log(_iou(pred_box, gt_box))

    gt_diagonal = math.hypot(gt_box[2] - gt_box[0], gt_box[3] - gt_box[1])
    box_distance = sum(abs(a - b) for a, b in zip(pred_box, gt_box)) / max(4.0 * gt_diagonal, 1.0)
    width, height = _frame_size(metadata, gt_box)
    point_distance = _point_alignment_distance(pred_points, gt_points, width, height)
    alignment_score = 0.5 * (_s_exp(box_distance) + _s_exp(point_distance))

    if _valid_box(pred_box):
        pred_area = max((pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1]), 1.0e-12)
        gt_area = max((gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]), 1.0e-12)
        pred_ratio = max((pred_box[2] - pred_box[0]) / (pred_box[3] - pred_box[1]), 1.0e-12)
        gt_ratio = max((gt_box[2] - gt_box[0]) / (gt_box[3] - gt_box[1]), 1.0e-12)
        scale_distance = math.hypot(math.log(pred_area) - math.log(gt_area), math.log(pred_ratio) - math.log(gt_ratio))
        scale_score = _s_exp(scale_distance)
    else:
        scale_score = 0.0

    components = (
        _modulate(iou_score, box_validity, point_validity),
        _modulate(alignment_score, box_validity, point_validity),
        _modulate(scale_score, box_validity, point_validity),
    )
    return sum(components) / len(components)


def think_mris_reasoning_format_reward(predict_str: str) -> float:
    return 1.0 if _FORMAT_PATTERN.fullmatch(predict_str.strip()) else 0.0


def think_mris_segmentation_format_reward(predict_str: str) -> float:
    objects = _answer_objects(predict_str)
    if not objects:
        return 0.0
    for item in objects:
        if not isinstance(item, dict) or not _valid_box(_bbox(item)) or len(_points(item)) != 2:
            return 0.0
    return 1.0


def think_mris_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    predicted = _answer_objects(predict_str)
    targets, metadata = _ground_truth_payload(ground_truth)
    if not predicted or not targets:
        return 0.0
    scores = np.asarray([[_pair_accuracy(p, t, metadata) for t in targets] for p in predicted], dtype=np.float64)
    rows, columns = linear_sum_assignment(1.0 - scores)
    return float(scores[rows, columns].sum() / max(len(predicted), len(targets)))


def think_mris_compute_score(
    predict_str: str,
    ground_truth: str,
    token_confidence_margins=None,
    problem_text: str = "",
    tokenizer=None,
):
    reasoning_format = think_mris_reasoning_format_reward(predict_str)
    segmentation_format = think_mris_segmentation_format_reward(predict_str)
    accuracy = think_mris_accuracy_reward(predict_str, ground_truth)
    original_reward = reasoning_format + segmentation_format + accuracy

    final_reward, diagnostics = _controller.final_reward(
        original_reward=original_reward,
        output_text=predict_str,
        ground_truth=ground_truth,
        token_confidence_margins=token_confidence_margins,
        problem_text=problem_text,
        tokenizer=tokenizer,
    )
    diagnostics.update(
        {
            "reasoning_format_reward": reasoning_format,
            "segmentation_format_reward": segmentation_format,
            "accuracy_reward": accuracy,
            "original_reward": original_reward,
        }
    )
    return final_reward, diagnostics
