import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class VisionLanguageKnowledgeInjector(nn.Module):
    """Construct the parameter-free VLKI dense prior from Eqs. (14)-(15).

    The public review code previously contained a trainable gated feature-fusion
    block.  The paper instead defines VLKI as a similarity map computed from the
    final-layer visual and reasoning hidden states of the same MLLM pass.  This
    implementation follows that definition and returns the 256x256 dense mask
    prompt consumed by SAM's frozen prompt encoder.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        eps: float = 1.0e-6,
        output_size: Tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.output_size = output_size

    def _normalize(self, value: torch.Tensor) -> torch.Tensor:
        return value / value.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    @staticmethod
    def _resolve_grid(token_count: int, visual_grid_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        if visual_grid_size is not None:
            height, width = int(visual_grid_size[0]), int(visual_grid_size[1])
            if height * width != token_count:
                raise ValueError(
                    f"visual grid {height}x{width} does not match {token_count} visual tokens"
                )
            return height, width
        side = int(math.isqrt(token_count))
        if side * side != token_count:
            raise ValueError("visual_grid_size is required for a non-square visual-token grid")
        return side, side

    def forward(
        self,
        visual_hidden_states: torch.Tensor,
        reasoning_hidden_states: torch.Tensor,
        visual_grid_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if visual_hidden_states.ndim != 3 or reasoning_hidden_states.ndim != 3:
            raise ValueError("visual and reasoning hidden states must have shape [batch, tokens, hidden]")
        if visual_hidden_states.shape[0] != reasoning_hidden_states.shape[0]:
            raise ValueError("visual and reasoning hidden states must have the same batch size")
        if visual_hidden_states.shape[2] != reasoning_hidden_states.shape[2]:
            raise ValueError("visual and reasoning hidden states must have the same hidden size")
        if reasoning_hidden_states.shape[1] == 0:
            raise ValueError("at least one reasoning token is required to construct VLKI")

        visual = self._normalize(visual_hidden_states)
        reasoning = self._normalize(reasoning_hidden_states)
        cross_similarity = torch.einsum("bvc,brc->bvr", visual, reasoning)

        reasoning_relevance = cross_similarity.amax(dim=1)
        reasoning_weights = torch.softmax(reasoning_relevance / self.temperature, dim=1)
        target_representation = self._normalize(
            torch.sum(reasoning_weights.unsqueeze(-1) * reasoning, dim=1)
        )

        visual_scores = torch.einsum("bc,bvc->bv", target_representation, visual)
        height, width = self._resolve_grid(visual_scores.shape[1], visual_grid_size)
        dense_prior = visual_scores.reshape(visual_scores.shape[0], 1, height, width)

        minimum = dense_prior.amin(dim=(-2, -1), keepdim=True)
        maximum = dense_prior.amax(dim=(-2, -1), keepdim=True)
        dense_prior = (dense_prior - minimum) / (maximum - minimum).clamp_min(self.eps)
        return F.interpolate(dense_prior, size=self.output_size, mode="bilinear", align_corners=False)

    def inject(
        self,
        visual_hidden_states: torch.Tensor,
        reasoning_hidden_states: torch.Tensor,
        visual_grid_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        return self.forward(visual_hidden_states, reasoning_hidden_states, visual_grid_size)
