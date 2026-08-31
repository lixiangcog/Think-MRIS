import unittest

from verl.utils.rl_dataset import image_processor_size_kwargs


class ImageProcessorSizeKwargsTest(unittest.TestCase):
    def test_forwards_both_visual_size_limits(self):
        self.assertEqual(
            image_processor_size_kwargs(min_pixels=3136, max_pixels=409600),
            {"min_pixels": 3136, "max_pixels": 409600},
        )

    def test_omits_unspecified_visual_size_limits(self):
        self.assertEqual(image_processor_size_kwargs(None, None), {})
        self.assertEqual(image_processor_size_kwargs(3136, None), {"min_pixels": 3136})


if __name__ == "__main__":
    unittest.main()
