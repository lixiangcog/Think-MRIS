import unittest

import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


class RolloutPaddingTest(unittest.TestCase):
    def test_partial_prompt_batch_removes_all_repeated_padding(self):
        prompts = DataProto.from_dict(tensors={"sample_id": torch.arange(6).unsqueeze(1)})
        padded, pad_size = pad_dataproto_to_divisor(prompts, size_divisor=4)
        self.assertEqual(len(padded), 8)
        self.assertEqual(pad_size, 2)

        completions_per_prompt = 2
        generated = DataProto.from_dict(
            tensors={
                "sample_id": padded.batch["sample_id"].repeat_interleave(completions_per_prompt, dim=0)
            }
        )
        unpadded = unpad_dataproto(generated, pad_size=pad_size * completions_per_prompt)
        self.assertEqual(len(unpadded), 12)
        self.assertEqual(unpadded.batch["sample_id"].flatten().tolist(), [
            0,
            0,
            1,
            1,
            2,
            2,
            3,
            3,
            4,
            4,
            5,
            5,
        ])

    def test_split_keeps_a_smaller_final_minibatch(self):
        data = DataProto.from_dict(tensors={"sample_id": torch.arange(10).unsqueeze(1)})
        parts = data.split(4)
        self.assertEqual([len(part) for part in parts], [4, 4, 2])
        self.assertEqual(parts[-1].batch["sample_id"].flatten().tolist(), [8, 9])

        single_partial_batch = data.split(32)
        self.assertEqual(len(single_partial_batch), 1)
        self.assertEqual(len(single_partial_batch[0]), 10)


if __name__ == "__main__":
    unittest.main()
