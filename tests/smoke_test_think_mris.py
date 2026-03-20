import importlib.util
from pathlib import Path


def load_module(module_name: str, relative_path: str):
    path = Path(relative_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    reasoning = load_module("tm_reasoning", "think_mris/reasoning.py")
    prompting = load_module("tm_prompting", "think_mris/prompting.py")

    controller = reasoning.TaskAdaptiveReasoningController()
    template = prompting.ThinkMRISPromptTemplate()

    ground_truth = (
        '{"solution":[{"bbox_2d":[12,24,60,88],"point_2d":[32,50]}],'
        '"scene_complexity":7.0,"segmentation_challenge":8.0,"linguistic_ambiguity":5.0}'
    )
    output_text = (
        "<think>The target is small and partially ambiguous, so a slightly longer reasoning chain is acceptable."
        "</think><answer>[{\"bbox_2d\":[12,24,60,88],\"point_2d\":[32,50]}]</answer>"
    )
    final_reward, diagnostics = controller.final_reward(
        original_reward=3.0,
        output_text=output_text,
        ground_truth=ground_truth,
        token_confidence_margins=[0.72, 0.65, 0.61, 0.59],
        problem_text="segment the small lesion near the upper-left ventricle",
    )

    prompt = template.format_user_prompt("segment the enhancing tumor core")
    print("reward", round(final_reward, 4))
    print("budget", diagnostics["reasoning_budget"])
    print("prompt_has_tags", "<think>" in prompt and "<answer>" in prompt)


if __name__ == "__main__":
    main()

