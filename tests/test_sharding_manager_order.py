import unittest
from types import SimpleNamespace
from unittest.mock import patch

from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager


class ShardingManagerOrderTest(unittest.TestCase):
    def test_vllm_wakes_before_fsdp_state_dict_materialization(self):
        events = []

        class FakeModule:
            def state_dict(self):
                events.append("state_dict")
                return {}

        model_runner = SimpleNamespace(model=object())
        worker = SimpleNamespace(model_runner=model_runner)
        driver_worker = SimpleNamespace(worker=worker)
        model_executor = SimpleNamespace(driver_worker=driver_worker)
        llm_engine = SimpleNamespace(model_executor=model_executor)

        class FakeInferenceEngine:
            def __init__(self):
                self.llm_engine = llm_engine

            def wake_up(self):
                events.append("wake_up")

        manager = FSDPVLLMShardingManager.__new__(FSDPVLLMShardingManager)
        manager.module = FakeModule()
        manager.inference_engine = FakeInferenceEngine()
        manager.device_mesh = None
        manager.lora_rank = 0
        manager.lora_alpha = 16.0

        with (
            patch("verl.workers.sharding_manager.fsdp_vllm.log_gpu_memory_usage"),
            patch("verl.workers.sharding_manager.fsdp_vllm.load_dtensor_weights"),
            patch("verl.workers.sharding_manager.fsdp_vllm.torch.cuda.empty_cache"),
        ):
            manager.__enter__()

        self.assertEqual(events, ["wake_up", "state_dict"])


if __name__ == "__main__":
    unittest.main()
