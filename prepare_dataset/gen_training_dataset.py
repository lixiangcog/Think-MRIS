import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage
from tqdm import tqdm


DIFFICULTY_KEYS = (
    "task_difficulty",
    "scene_complexity",
    "segmentation_challenge",
    "linguistic_ambiguity",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a paper-compatible Think-MRIS training dataset")
    parser.add_argument("--annotations", required=True, help="JSON list with image_path, problem, bboxes, and points")
    parser.add_argument("--output-dir", default="data/MRIS-Bench-prepared")
    parser.add_argument("--max-side-length", type=int, default=640)
    return parser.parse_args()


def resize_max_side(image: PILImage.Image, max_side_length: int):
    if max(image.size) <= max_side_length:
        return image, 1.0, 1.0
    scale = max_side_length / max(image.size)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        PILImage.Resampling.BILINEAR,
    )
    return resized, resized.width / image.width, resized.height / image.height


def _two_points_per_target(item, target_count: int):
    candidates = item.get("points_2d", item.get("interior_points", item.get("key_points")))
    if not isinstance(candidates, list) or len(candidates) != target_count:
        raise ValueError("each target must provide exactly two mask-derived interior points")
    for points in candidates:
        if not isinstance(points, list) or len(points) != 2 or any(
            not isinstance(point, list) or len(point) != 2 for point in points
        ):
            raise ValueError("each target must provide points_2d=[[x1,y1],[x2,y2]]")
    return candidates


def prepare_record(item, max_side_length: int):
    image = PILImage.open(item["image_path"]).convert("RGB")
    image, x_scale, y_scale = resize_max_side(image, max_side_length)
    boxes = item.get("bboxes", [])
    points_per_target = _two_points_per_target(item, len(boxes))
    solution = []
    for box, points in zip(boxes, points_per_target):
        solution.append(
            {
                "bbox_2d": [
                    round(box[0] * x_scale),
                    round(box[1] * y_scale),
                    round(box[2] * x_scale),
                    round(box[3] * y_scale),
                ],
                "points_2d": [
                    [round(point[0] * x_scale), round(point[1] * y_scale)] for point in points
                ],
            }
        )
    metadata = {
        "solution": solution,
        "img_height": image.height,
        "img_width": image.width,
    }
    for key in DIFFICULTY_KEYS:
        if key in item:
            metadata[key] = item[key]
    return {
        "id": str(item["id"]),
        "problem": item["problem"],
        "solution": json.dumps(metadata, ensure_ascii=False),
        "image": image,
        "img_height": image.height,
        "img_width": image.width,
    }


def main():
    args = parse_args()
    records = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    prepared = [prepare_record(item, args.max_side_length) for item in tqdm(records)]
    columns = {key: [record[key] for record in prepared] for key in prepared[0]}
    features = Features(
        {
            "id": Value("string"),
            "problem": Value("string"),
            "solution": Value("string"),
            "image": Image(),
            "img_height": Value("int64"),
            "img_width": Value("int64"),
        }
    )
    dataset = DatasetDict({"train": Dataset.from_dict(columns, features=features)})
    dataset.save_to_disk(args.output_dir)
    print(f"Dataset saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
