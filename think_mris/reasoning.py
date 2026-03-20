import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _safe_json_loads(payload: str) -> Any:
    try:
        return json.loads(payload)
    except Exception:
        return None


def _extract_between_tags(output_text: str, tag_name: str) -> str:
    match = re.search(rf"<{tag_name}>(.*?)</{tag_name}>", output_text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_answer_objects(output_text: str) -> List[Dict[str, Any]]:
    payload = _safe_json_loads(_extract_between_tags(output_text, "answer"))
    return payload if isinstance(payload, list) else []


def _normalize_ground_truth(ground_truth: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    payload = _safe_json_loads(ground_truth)
    if isinstance(payload, list):
        return payload, {}
    if isinstance(payload, dict):
        solution = payload.get("solution", payload.get("target", payload.get("answer", [])))
        if isinstance(solution, str):
            solution = _safe_json_loads(solution)
        solution = solution if isinstance(solution, list) else []
        return solution, payload
    return [], {}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class DifficultyScore:
    scene_complexity: float
    segmentation_challenge: float
    linguistic_ambiguity: float

    @property
    def mean(self) -> float:
        return (self.scene_complexity + self.segmentation_challenge + self.linguistic_ambiguity) / 3.0


@dataclass
class TaskAdaptiveReasoningController:
    tau_hard: float = 7.0
    tau_easy: float = 4.0
    base_budget: float = 80.0
    low_budget: float = 32.0
    alpha: float = 48.0
    beta: float = 0.008

    def extract_reasoning_length(self, output_text: str) -> int:
        think_text = _extract_between_tags(output_text, "think")
        if not think_text:
            return 0
        return len(think_text.split())

    def compute_uncertainty(self, token_confidence_margins: Optional[List[float]]) -> float:
        if not token_confidence_margins:
            return 0.5
        valid_margins = [_clamp(float(m), 0.0, 1.0) for m in token_confidence_margins if m is not None]
        if not valid_margins:
            return 0.5
        return _clamp(1.0 - sum(valid_margins) / len(valid_margins), 0.0, 1.0)

    def estimate_scene_complexity(
        self,
        solution_objects: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> float:
        explicit = metadata.get("scene_complexity")
        if explicit is not None:
            return _clamp(float(explicit), 1.0, 10.0)

        distractor_count = metadata.get("distractor_count")
        distractor_similarity = metadata.get("distractor_similarity")
        if distractor_count is not None or distractor_similarity is not None:
            distractor_count = float(distractor_count or 0.0)
            distractor_similarity = float(distractor_similarity or 0.0)
            score = 1.0 + min(5.0, distractor_count) + 4.0 * _clamp(distractor_similarity, 0.0, 1.0)
            return _clamp(score, 1.0, 10.0)

        object_count = len(solution_objects)
        score = 1.0 + min(4.0, max(object_count - 1, 0) * 1.5)
        return _clamp(score, 1.0, 10.0)

    def estimate_segmentation_challenge(
        self,
        solution_objects: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> float:
        explicit = metadata.get("segmentation_challenge")
        if explicit is not None:
            return _clamp(float(explicit), 1.0, 10.0)

        img_height = float(metadata.get("img_height", metadata.get("height", 840)) or 840)
        img_width = float(metadata.get("img_width", metadata.get("width", 840)) or 840)
        image_area = max(img_height * img_width, 1.0)
        artifact_level = float(metadata.get("artifact_level", 0.0) or 0.0)
        part_level = 1.5 if metadata.get("target_granularity") in {"part", "sub-part", "local"} else 0.0

        if not solution_objects:
            return 5.0

        object_scores = []
        for item in solution_objects:
            bbox = item.get("bbox_2d")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            width = max(float(bbox[2]) - float(bbox[0]), 1.0)
            height = max(float(bbox[3]) - float(bbox[1]), 1.0)
            area_ratio = _clamp((width * height) / image_area, 1e-6, 1.0)
            center_x = (float(bbox[0]) + float(bbox[2])) / 2.0 / img_width
            center_y = (float(bbox[1]) + float(bbox[3])) / 2.0 / img_height
            border_distance = min(center_x, center_y, 1 - center_x, 1 - center_y)
            small_target = 4.0 * (1.0 - min(1.0, math.sqrt(area_ratio) * 4.0))
            peripheral_target = 2.0 * (1.0 - _clamp(border_distance / 0.25, 0.0, 1.0))
            elongated = min(1.5, abs(math.log(width / height)))
            object_scores.append(1.0 + small_target + peripheral_target + elongated)

        if not object_scores:
            return 5.0

        score = sum(object_scores) / len(object_scores) + 2.0 * _clamp(artifact_level, 0.0, 1.0) + part_level
        return _clamp(score, 1.0, 10.0)

    def estimate_linguistic_ambiguity(self, problem_text: str, metadata: Dict[str, Any]) -> float:
        explicit = metadata.get("linguistic_ambiguity")
        if explicit is not None:
            return _clamp(float(explicit), 1.0, 10.0)

        text = (metadata.get("referring_expression") or problem_text or "").lower().strip()
        if not text:
            return 5.0

        ambiguity_terms = [
            "near",
            "adjacent",
            "around",
            "approximately",
            "possible",
            "suspected",
            "region",
            "area",
            "lesion",
            "abnormality",
            "focus",
            "structure",
        ]
        location_terms = [
            "left",
            "right",
            "upper",
            "lower",
            "anterior",
            "posterior",
            "medial",
            "lateral",
            "center",
        ]

        token_count = len(text.split())
        ambiguity_hits = sum(term in text for term in ambiguity_terms)
        location_hits = sum(term in text for term in location_terms)

        score = 2.0 + min(4.0, ambiguity_hits * 1.2)
        if token_count <= 3:
            score += 2.0
        elif token_count <= 6:
            score += 1.0
        if location_hits >= 2:
            score -= 1.5
        elif location_hits == 1:
            score -= 0.75

        return _clamp(score, 1.0, 10.0)

    def estimate_task_difficulty(self, ground_truth: str, problem_text: str = "") -> DifficultyScore:
        solution_objects, metadata = _normalize_ground_truth(ground_truth)

        explicit = metadata.get("task_difficulty", metadata.get("difficulty"))
        if explicit is not None:
            score = _clamp(float(explicit), 1.0, 10.0)
            return DifficultyScore(score, score, score)

        scene_complexity = self.estimate_scene_complexity(solution_objects, metadata)
        segmentation_challenge = self.estimate_segmentation_challenge(solution_objects, metadata)
        linguistic_ambiguity = self.estimate_linguistic_ambiguity(problem_text, metadata)
        return DifficultyScore(scene_complexity, segmentation_challenge, linguistic_ambiguity)

    def reasoning_budget(self, difficulty_score: float, uncertainty_score: float) -> Optional[float]:
        if difficulty_score >= self.tau_hard:
            return self.base_budget + self.alpha * uncertainty_score
        if difficulty_score < self.tau_easy:
            return self.low_budget
        return None

    def soft_length_penalty(self, used_length: int, budget: Optional[float]) -> float:
        if budget is None:
            return 1.0
        if used_length > budget:
            return max(0.0, 1.0 - self.beta * (used_length - budget))
        return 1.0

    def final_reward(
        self,
        original_reward: float,
        output_text: str,
        ground_truth: str,
        token_confidence_margins: Optional[List[float]] = None,
        problem_text: str = "",
    ) -> Tuple[float, Dict[str, float]]:
        difficulty = self.estimate_task_difficulty(ground_truth, problem_text=problem_text)
        difficulty_score = difficulty.mean
        uncertainty_score = self.compute_uncertainty(token_confidence_margins)
        used_length = self.extract_reasoning_length(output_text)
        budget = self.reasoning_budget(difficulty_score, uncertainty_score)
        penalty = self.soft_length_penalty(used_length, budget)
        final_reward = original_reward * penalty

        diagnostics = {
            "scene_complexity": difficulty.scene_complexity,
            "segmentation_challenge": difficulty.segmentation_challenge,
            "linguistic_ambiguity": difficulty.linguistic_ambiguity,
            "difficulty_score": difficulty_score,
            "uncertainty_score": uncertainty_score,
            "used_reasoning_length": float(used_length),
            "reasoning_budget": -1.0 if budget is None else float(budget),
            "soft_penalty": penalty,
        }
        return final_reward, diagnostics

