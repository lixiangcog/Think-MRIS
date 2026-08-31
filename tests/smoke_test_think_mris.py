import json
import unittest

import torch

from think_mris.modeling import VisionLanguageKnowledgeInjector
from think_mris.prompting import ThinkMRISPromptTemplate
from think_mris.reasoning import TaskAdaptiveReasoningController
from think_mris.utils import extract_think_answer, scale_answer_objects
from verl.utils.reward_score.think_mris import think_mris_compute_score
from verl.workers.sharding_manager.lora_utils import iter_merged_lora_weights


class ThinkMRISSmokeTest(unittest.TestCase):
    def setUp(self):
        self.target = {
            "solution": [
                {
                    "bbox_2d": [12, 24, 60, 88],
                    "points_2d": [[32, 50], [45, 70]],
                }
            ],
            "scene_complexity": 7.0,
            "segmentation_challenge": 8.0,
            "linguistic_ambiguity": 5.0,
            "img_width": 100,
            "img_height": 100,
        }
        self.output = (
            "<think>The small target is distinguished from the adjacent structure.</think>"
            '<answer>[{"bbox_2d":[12,24,60,88],"points_2d":[[32,50],[45,70]]}]</answer>'
        )

    def test_paper_hyperparameters_and_reward(self):
        controller = TaskAdaptiveReasoningController()
        self.assertEqual(controller.base_budget, 256.0)
        self.assertEqual(controller.low_budget, 96.0)
        self.assertEqual(controller.tau_hard, 5.0)
        self.assertEqual(controller.tau_easy, 3.5)
        self.assertEqual(controller.alpha, 25.0)
        self.assertEqual(controller.beta, 2.0e-3)

        reward, diagnostics = think_mris_compute_score(
            self.output,
            json.dumps(self.target),
            token_confidence_margins=[0.72, 0.65, 0.61, 0.59],
            problem_text="segment the small lesion near the upper-left ventricle",
        )
        self.assertGreater(reward, 2.8)
        self.assertEqual(diagnostics["reasoning_format_reward"], 1.0)
        self.assertEqual(diagnostics["segmentation_format_reward"], 1.0)
        self.assertGreater(diagnostics["accuracy_reward"], 0.9)

    def test_three_axis_scores_take_precedence_over_cached_mean(self):
        controller = TaskAdaptiveReasoningController()
        payload = dict(self.target)
        payload["difficulty"] = 4.0
        score = controller.estimate_task_difficulty(json.dumps(payload), problem_text="the indicated area")
        self.assertEqual(score.scene_complexity, 7.0)
        self.assertEqual(score.segmentation_challenge, 8.0)
        self.assertEqual(score.linguistic_ambiguity, 5.0)
        self.assertAlmostEqual(score.mean, 20 / 3)

    def test_prompt_and_geometry_scaling(self):
        prompt = ThinkMRISPromptTemplate().format_user_prompt("segment the enhancing tumor core")
        self.assertIn("points_2d", prompt)
        self.assertIn("two interior key points", prompt)
        _, objects = extract_think_answer(self.output)
        scaled = scale_answer_objects(objects, (100, 100), (200, 50))
        self.assertEqual(scaled[0]["bbox_2d"], [24, 12, 120, 44])
        self.assertEqual(scaled[0]["points_2d"], [[64, 25], [90, 35]])

    def test_parameter_free_vlki(self):
        torch.manual_seed(7)
        visual = torch.randn(2, 12, 16)
        reasoning = torch.randn(2, 5, 16)
        injector = VisionLanguageKnowledgeInjector()
        prior = injector(visual, reasoning, visual_grid_size=(3, 4))
        self.assertEqual(tuple(prior.shape), (2, 1, 256, 256))
        self.assertGreaterEqual(float(prior.min()), 0.0)
        self.assertLessEqual(float(prior.max()), 1.0)
        self.assertEqual(sum(parameter.numel() for parameter in injector.parameters()), 0)

    def test_lora_merge_for_vllm_sync(self):
        state = {
            "base_model.model.model.layers.0.q_proj.base_layer.weight": torch.eye(2),
            "base_model.model.model.layers.0.q_proj.lora_A.default.weight": torch.tensor([[1.0, 2.0]]),
            "base_model.model.model.layers.0.q_proj.lora_B.default.weight": torch.tensor([[3.0], [4.0]]),
        }
        merged = dict(iter_merged_lora_weights(state, lora_rank=1, lora_alpha=2))
        expected = torch.eye(2) + 2 * torch.tensor([[3.0, 6.0], [4.0, 8.0]])
        self.assertTrue(torch.equal(merged["model.layers.0.q_proj.weight"], expected))


if __name__ == "__main__":
    unittest.main()
