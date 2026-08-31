from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .knowledge_injection import VisionLanguageKnowledgeInjector


@dataclass
class SegmentationBridgeOutput:
    low_res_masks: torch.Tensor
    iou_predictions: torch.Tensor
    vlki_dense_prior: torch.Tensor

    @property
    def injected_visual_features(self) -> torch.Tensor:
        """Backward-compatible alias for the old public-review API."""
        return self.vlki_dense_prior


class KnowledgeInjectedSegmentationBridge(nn.Module):
    """Feed VLKI as a dense mask prompt while leaving SAM image features frozen."""

    def __init__(
        self,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        temperature: float = 0.1,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.knowledge_injector = VisionLanguageKnowledgeInjector(
            temperature=temperature,
            eps=eps,
            output_size=(256, 256),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        visual_hidden_states: torch.Tensor,
        reasoning_hidden_states: torch.Tensor,
        visual_grid_size: Optional[Tuple[int, int]] = None,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
    ) -> SegmentationBridgeOutput:
        dense_prior = self.knowledge_injector(
            visual_hidden_states=visual_hidden_states,
            reasoning_hidden_states=reasoning_hidden_states,
            visual_grid_size=visual_grid_size,
        )
        sparse_embeddings, dense_prompt_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels)
            if point_coords is not None and point_labels is not None
            else None,
            boxes=boxes,
            masks=dense_prior,
        )

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
        )
        return SegmentationBridgeOutput(
            low_res_masks=low_res_masks,
            iou_predictions=iou_predictions,
            vlki_dense_prior=dense_prior,
        )

    @torch.no_grad()
    def predict_best_mask(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.forward(multimask_output=True, **kwargs)
        best_idx = outputs.iou_predictions.argmax(dim=-1)
        batch_idx = torch.arange(outputs.low_res_masks.size(0), device=outputs.low_res_masks.device)
        best_masks = outputs.low_res_masks[batch_idx, best_idx][:, None, ...]
        best_scores = outputs.iou_predictions[batch_idx, best_idx][:, None]
        return best_masks, best_scores
