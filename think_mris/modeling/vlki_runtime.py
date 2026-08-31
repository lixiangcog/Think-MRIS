"""Runtime extraction of paper-aligned VLKI priors from Qwen2.5-VL states."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .knowledge_injection import VisionLanguageKnowledgeInjector


def reasoning_span(response_ids: Sequence[int], processor) -> Tuple[int, int]:
    """Return token offsets overlapping the text inside ``<think>`` tags."""

    tokenizer = processor.tokenizer
    decode_kwargs = {
        "skip_special_tokens": False,
        "clean_up_tokenization_spaces": False,
    }
    response_text = tokenizer.decode(response_ids, **decode_kwargs)
    opening = response_text.find("<think>")
    if opening < 0:
        raise ValueError("generated response is missing <think> tags required by VLKI")
    content_start = opening + len("<think>")
    content_end = response_text.find("</think>", content_start)
    if content_end <= content_start:
        raise ValueError("generated response has an empty or unterminated <think> block")

    selected_tokens = []
    previous_end = 0
    for index in range(len(response_ids)):
        current_end = len(tokenizer.decode(response_ids[: index + 1], **decode_kwargs))
        if current_end > content_start and previous_end < content_end:
            selected_tokens.append(index)
        previous_end = current_end
    if not selected_tokens:
        raise ValueError("the <think> block contains no tokenized reasoning")
    return selected_tokens[0], selected_tokens[-1] + 1


@torch.inference_mode()
def build_vlki_priors(
    model,
    processor,
    prompt_inputs: Dict[str, torch.Tensor],
    generated_ids: torch.Tensor,
    allow_missing: bool = False,
) -> List[Optional[torch.Tensor]]:
    """Build one 1x1x256x256 VLKI dense prior for each generated sample.

    A single batched forward pass keeps evaluation practical while preserving
    each sample's visual-token grid and ``<think>`` token span.
    """

    prompt_length = prompt_inputs["input_ids"].shape[1]
    response_length = generated_ids.shape[1] - prompt_length
    attention_mask = torch.cat(
        (
            prompt_inputs["attention_mask"],
            torch.ones(
                (generated_ids.shape[0], response_length),
                dtype=prompt_inputs["attention_mask"].dtype,
                device=generated_ids.device,
            ),
        ),
        dim=1,
    )
    forward_inputs = {
        key: value
        for key, value in prompt_inputs.items()
        if key in {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}
    }
    forward_inputs.update(
        {
            "input_ids": generated_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
            "use_cache": False,
        }
    )
    outputs = model(**forward_inputs)
    hidden_states = outputs.hidden_states[-1]
    image_token_id = model.config.image_token_id
    merge_size = int(processor.image_processor.merge_size)
    injector = VisionLanguageKnowledgeInjector().to(hidden_states.device)
    priors: List[Optional[torch.Tensor]] = []

    if prompt_inputs["image_grid_thw"].shape[0] != generated_ids.shape[0]:
        raise ValueError("VLKI evaluation requires exactly one image per generated sample")

    for batch_index in range(generated_ids.shape[0]):
        try:
            visual_mask = generated_ids[batch_index] == image_token_id
            visual_hidden_states = hidden_states[batch_index : batch_index + 1, visual_mask, :]
            if visual_hidden_states.shape[1] == 0:
                raise ValueError(f"sample {batch_index} has no visual tokens in the MLLM forward pass")

            response_ids = generated_ids[batch_index, prompt_length:].tolist()
            reasoning_start, reasoning_end = reasoning_span(response_ids, processor)
            reasoning_hidden_states = hidden_states[
                batch_index : batch_index + 1,
                prompt_length + reasoning_start : prompt_length + reasoning_end,
                :,
            ]
            grid_thw = prompt_inputs["image_grid_thw"][batch_index]
            visual_grid_size = (
                int(grid_thw[1].item()) // merge_size,
                int(grid_thw[2].item()) // merge_size,
            )
            priors.append(
                injector(
                    visual_hidden_states,
                    reasoning_hidden_states,
                    visual_grid_size=visual_grid_size,
                )
            )
        except ValueError:
            if not allow_missing:
                raise
            priors.append(None)
    return priors


@torch.inference_mode()
def build_vlki_prior(
    model,
    processor,
    prompt_inputs: Dict[str, torch.Tensor],
    generated_ids: torch.Tensor,
) -> torch.Tensor:
    """Single-sample compatibility wrapper."""

    if generated_ids.shape[0] != 1:
        raise ValueError("build_vlki_prior expects a single sample; use build_vlki_priors for a batch")
    prior = build_vlki_priors(model, processor, prompt_inputs, generated_ids)[0]
    if prior is None:
        raise RuntimeError("single-sample VLKI extraction unexpectedly returned no prior")
    return prior
