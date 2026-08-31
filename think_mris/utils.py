import json
import re
from typing import Any, Dict, List, Sequence, Tuple


def extract_think_answer(output_text: str) -> Tuple[str, List[dict]]:
    think_match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL)
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", output_text, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""
    answer_payload = []
    if answer_match:
        try:
            parsed = json.loads(answer_match.group(1))
            if isinstance(parsed, list):
                answer_payload = parsed
        except Exception:
            pass
    return think_text, answer_payload


def validate_answer_objects(objects: Sequence[Dict[str, Any]]) -> None:
    if not objects:
        raise ValueError("the model output contains no targets inside <answer> tags")
    for index, item in enumerate(objects):
        bbox = item.get("bbox_2d")
        points = item.get("points_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"target {index} must contain bbox_2d with four coordinates")
        if not isinstance(points, list) or len(points) != 2 or any(
            not isinstance(point, list) or len(point) != 2 for point in points
        ):
            raise ValueError(f"target {index} must contain exactly two points_2d")


def scale_answer_objects(
    objects: Sequence[Dict[str, Any]],
    source_size: Tuple[int, int],
    destination_size: Tuple[int, int],
) -> List[Dict[str, List]]:
    """Scale predicted geometry from the MLLM image frame to the original image."""
    validate_answer_objects(objects)
    source_width, source_height = source_size
    destination_width, destination_height = destination_size
    x_scale = destination_width / source_width
    y_scale = destination_height / source_height
    scaled = []
    for item in objects:
        bbox = [
            round(item["bbox_2d"][0] * x_scale),
            round(item["bbox_2d"][1] * y_scale),
            round(item["bbox_2d"][2] * x_scale),
            round(item["bbox_2d"][3] * y_scale),
        ]
        bbox[0], bbox[2] = sorted((max(0, min(destination_width - 1, bbox[0])), max(0, min(destination_width - 1, bbox[2]))))
        bbox[1], bbox[3] = sorted((max(0, min(destination_height - 1, bbox[1])), max(0, min(destination_height - 1, bbox[3]))))
        points = [
            [
                max(0, min(destination_width - 1, round(point[0] * x_scale))),
                max(0, min(destination_height - 1, round(point[1] * y_scale))),
            ]
            for point in item["points_2d"]
        ]
        scaled.append({"bbox_2d": bbox, "points_2d": points})
    return scaled

