import unittest

import torch

from training_scripts.model_merger import merge_checkpoint_shards, merge_lora_state_dict


class ModelMergerTest(unittest.TestCase):
    def test_world_size_one_plain_checkpoint(self):
        state = {"model.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
        merged = merge_checkpoint_shards([state])
        self.assertTrue(torch.equal(merged["model.weight"], state["model.weight"]))

    def test_peft_keys_are_normalized_and_lora_is_folded(self):
        prefix = "base_model.model.model.layers.0.self_attn.q_proj"
        state = {
            f"{prefix}.base_layer.weight": torch.zeros((2, 3), dtype=torch.bfloat16),
            f"{prefix}.lora_A.default.weight": torch.ones((1, 3), dtype=torch.bfloat16),
            f"{prefix}.lora_B.default.weight": torch.full((2, 1), 2.0, dtype=torch.bfloat16),
            "base_model.model.model.norm.weight": torch.ones(2, dtype=torch.bfloat16),
        }
        merged, adapter_count = merge_lora_state_dict(state, lora_alpha=2.0)
        self.assertEqual(adapter_count, 1)
        self.assertEqual(set(merged), {
            "model.layers.0.self_attn.q_proj.weight",
            "model.norm.weight",
        })
        self.assertTrue(torch.equal(
            merged["model.layers.0.self_attn.q_proj.weight"],
            torch.full((2, 3), 4.0, dtype=torch.bfloat16),
        ))


if __name__ == "__main__":
    unittest.main()
