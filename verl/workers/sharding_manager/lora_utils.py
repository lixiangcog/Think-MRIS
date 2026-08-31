from typing import Dict, Iterator, Tuple

import torch


def _full_tensor(value):
    return value.full_tensor() if hasattr(value, "full_tensor") else value


def _strip_wrapper_prefix(name: str) -> str:
    for prefix in ("_fsdp_wrapped_module.", "base_model.model."):
        while name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def has_lora_weights(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(".lora_A." in name for name in state_dict)


def iter_merged_lora_weights(
    state_dict: Dict[str, torch.Tensor],
    lora_rank: int,
    lora_alpha: float,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield effective base weights without mutating the trainable PEFT model.

    vLLM owns a regular base model. At rollout time each LoRA-targeted weight is
    materialized one layer at a time as ``W + (alpha/r) * B @ A`` and loaded into
    vLLM. This avoids holding a second merged 7B state dict in GPU memory.
    """
    if lora_rank <= 0:
        return
    scaling = float(lora_alpha) / float(lora_rank)
    for a_name in sorted(name for name in state_dict if ".lora_A." in name and name.endswith(".weight")):
        module_name, adapter_suffix = a_name.split(".lora_A.", maxsplit=1)
        b_name = f"{module_name}.lora_B.{adapter_suffix}"
        base_name = f"{module_name}.base_layer.weight"
        if b_name not in state_dict or base_name not in state_dict:
            raise KeyError(f"incomplete LoRA state for {module_name}")
        base = _full_tensor(state_dict[base_name])
        lora_a = _full_tensor(state_dict[a_name]).to(dtype=base.dtype, device=base.device)
        lora_b = _full_tensor(state_dict[b_name]).to(dtype=base.dtype, device=base.device)
        effective = base + torch.matmul(lora_b, lora_a) * scaling
        canonical_name = _strip_wrapper_prefix(base_name).replace(".base_layer.weight", ".weight")
        yield canonical_name, effective
