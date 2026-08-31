# Copyright 2024 Anonymous contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Merge an FSDP/LoRA trainer checkpoint into a Hugging Face model."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.distributed._tensor import DTensor, Placement
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForVision2Seq


def merge_by_placement(tensors: List[torch.Tensor], placement: Placement) -> torch.Tensor:
    if placement.is_replicate():
        return tensors[0]
    if placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    if placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    raise ValueError(f"Unsupported placement: {placement}")


def checkpoint_world_size(local_dir: Path) -> int:
    for filename in local_dir.iterdir():
        match = re.fullmatch(r"model_world_size_(\d+)_rank_0\.pt", filename.name)
        if match:
            return int(match.group(1))
    raise FileNotFoundError(f"no model_world_size_*_rank_0.pt checkpoint found under {local_dir}")


def load_checkpoint_shards(local_dir: Path, world_size: int) -> List[Dict[str, torch.Tensor]]:
    def load(rank: int):
        path = local_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        return torch.load(path, map_location="cpu", weights_only=False)

    with ThreadPoolExecutor(max_workers=min(16, world_size)) as executor:
        return list(executor.map(load, range(world_size)))


def merge_checkpoint_shards(shards: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not shards:
        raise ValueError("at least one checkpoint shard is required")
    keys = set(shards[0])
    if any(set(shard) != keys for shard in shards[1:]):
        raise ValueError("checkpoint shards contain different parameter keys")

    first_dtensor = next(
        (value for shard in shards for value in shard.values() if isinstance(value, DTensor)),
        None,
    )
    if first_dtensor is not None:
        mesh_dim_names = first_dtensor.device_mesh.mesh_dim_names
        if mesh_dim_names != ("fsdp",):
            raise NotImplementedError(f"unsupported device mesh dimensions: {mesh_dim_names}")
        expected_shards = int(first_dtensor.device_mesh.mesh.numel())
        if expected_shards != len(shards):
            raise ValueError(f"device mesh expects {expected_shards} shards, received {len(shards)}")

    merged: Dict[str, torch.Tensor] = {}
    for key in sorted(keys):
        values = [shard[key] for shard in shards]
        if isinstance(values[0], DTensor):
            if not all(isinstance(value, DTensor) for value in values):
                raise TypeError(f"parameter {key} mixes DTensor and Tensor shards")
            placements = tuple(values[0].placements)
            if len(placements) != 1:
                raise NotImplementedError(f"parameter {key} uses unsupported placements {placements}")
            if any(tuple(value.placements) != placements for value in values[1:]):
                raise ValueError(f"parameter {key} has inconsistent placements")
            tensor = merge_by_placement([value._local_tensor for value in values], placements[0])
        else:
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise TypeError(f"parameter {key} is not tensor-valued")
            if any(value.shape != values[0].shape for value in values[1:]):
                raise ValueError(f"plain parameter {key} has inconsistent shapes across ranks")
            tensor = values[0]
        merged[key] = tensor.cpu()
    return merged


def normalize_peft_key(key: str) -> str:
    if key.startswith("base_model.model."):
        key = key[len("base_model.model.") :]
    return key.replace(".base_layer.", ".")


def merge_lora_state_dict(
    state_dict: Dict[str, torch.Tensor],
    lora_alpha: float,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Normalize PEFT keys and fold every LoRA delta into its base weight."""
    output: Dict[str, torch.Tensor] = {}
    lora_a_pattern = re.compile(r"^(.*)\.lora_A\.[^.]+\.weight$")

    for key, tensor in state_dict.items():
        if ".lora_A." in key or ".lora_B." in key:
            continue
        normalized = normalize_peft_key(key)
        if normalized in output:
            raise ValueError(f"duplicate normalized parameter key {normalized}")
        output[normalized] = tensor.bfloat16() if tensor.is_floating_point() else tensor

    merged_adapters = 0
    for a_key, a_tensor in state_dict.items():
        match = lora_a_pattern.fullmatch(a_key)
        if not match:
            continue
        prefix = match.group(1)
        adapter_name = a_key.rsplit(".lora_A.", 1)[1].rsplit(".weight", 1)[0]
        b_key = f"{prefix}.lora_B.{adapter_name}.weight"
        base_key = f"{prefix}.base_layer.weight"
        if b_key not in state_dict or base_key not in state_dict:
            raise KeyError(f"incomplete LoRA triplet for {prefix}")
        rank = int(a_tensor.shape[0])
        if rank <= 0:
            raise ValueError(f"invalid LoRA rank for {prefix}: {rank}")
        base_target = normalize_peft_key(base_key)
        if base_target not in output:
            raise KeyError(f"normalized base weight is absent for {prefix}")
        b_tensor = state_dict[b_key]
        if a_tensor.ndim != 2 or b_tensor.ndim != 2:
            raise ValueError(f"only linear LoRA matrices are supported for {prefix}")
        delta = torch.matmul(b_tensor.float(), a_tensor.float())
        output[base_target] = (
            output[base_target].float() + delta * (float(lora_alpha) / rank)
        ).bfloat16()
        merged_adapters += 1

    return output, merged_adapters


def model_class_for_config(config):
    architecture = config.architectures[0]
    if "ForTokenClassification" in architecture:
        return AutoModelForTokenClassification
    if "ForCausalLM" in architecture:
        return AutoModelForCausalLM
    if "ForConditionalGeneration" in architecture:
        return AutoModelForVision2Seq
    raise NotImplementedError(f"unknown architecture {config.architectures}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True, type=Path, help="Actor checkpoint directory")
    parser.add_argument("--lora_alpha", default=16.0, type=float)
    parser.add_argument("--hf_upload_path", default=None, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = args.local_dir
    if local_dir.name == "huggingface":
        raise ValueError("--local_dir must be the actor directory, not its huggingface child")

    world_size = checkpoint_world_size(local_dir)
    print(f"Loading {world_size} checkpoint shard(s)")
    raw_state_dict = merge_checkpoint_shards(load_checkpoint_shards(local_dir, world_size))
    state_dict, merged_adapters = merge_lora_state_dict(raw_state_dict, lora_alpha=args.lora_alpha)
    del raw_state_dict
    print(f"Normalized {len(state_dict)} parameters and merged {merged_adapters} LoRA adapters")

    hf_path = local_dir / "huggingface"
    config = AutoConfig.from_pretrained(hf_path)
    auto_model = model_class_for_config(config)
    with torch.device("meta"):
        model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device="cpu")
    model.save_pretrained(hf_path, state_dict=state_dict)
    del state_dict
    del model

    report = {
        "checkpoint_world_size": world_size,
        "merged_lora_adapters": merged_adapters,
        "lora_alpha": args.lora_alpha,
        "output_path": str(hf_path),
    }
    (hf_path / "merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if args.hf_upload_path:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=args.hf_upload_path, private=False, exist_ok=True)
        api.upload_folder(folder_path=hf_path, repo_id=args.hf_upload_path, repo_type="model")


if __name__ == "__main__":
    main()
