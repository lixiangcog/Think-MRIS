from think_mris.reasoning import TaskAdaptiveReasoningController

from .vision_reasoner import (
    vision_reasoner_accuracy_reward,
    vision_reasoner_format_reward,
    vision_reasoner_non_repeat_reward,
)


_controller = TaskAdaptiveReasoningController()


def think_mris_compute_score(
    predict_str: str,
    ground_truth: str,
    token_confidence_margins=None,
    problem_text: str = "",
):
    format_reward = vision_reasoner_format_reward(predict_str)
    accuracy_reward = vision_reasoner_accuracy_reward(predict_str, ground_truth)
    non_repeat_reward = vision_reasoner_non_repeat_reward(predict_str)
    original_reward = format_reward + accuracy_reward + non_repeat_reward
    final_reward, diagnostics = _controller.final_reward(
        original_reward=original_reward,
        output_text=predict_str,
        ground_truth=ground_truth,
        token_confidence_margins=token_confidence_margins,
        problem_text=problem_text,
    )
    return final_reward, diagnostics

