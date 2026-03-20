from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .knowledge_injection import VisionLanguageKnowledgeInjector


@dataclass
class SegmentationBridgeOutput:
    low_res_masks: torch.Tensor
    iou_predictions: torch.Tensor
    injected_visual_features: torch.Tensor


class KnowledgeInjectedSegmentationBridge(nn.Module):
    """Inject language semantics before SAM-style mask decoding."""

    def __init__(
        self,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        visual_dim: int = 256,
        language_dim: int = 2048,
        downsample_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.knowledge_injector = VisionLanguageKnowledgeInjector(
            visual_dim=visual_dim,
            language_dim=language_dim,
            downsample_ratio=downsample_ratio,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        language_hidden_states: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        mask_inputs: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
    ) -> SegmentationBridgeOutput:
        sparse_embeddings, dense_prompt_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels) if point_coords is not None and point_labels is not None else None,
            boxes=boxes,
            masks=mask_inputs,
        )

        injected_visual_features = self.knowledge_injector.inject(
            dense_visual_features=image_embeddings,
            language_hidden_states=language_hidden_states,
        )

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=injected_visual_features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
        )

        return SegmentationBridgeOutput(
            low_res_masks=low_res_masks,
            iou_predictions=iou_predictions,
            injected_visual_features=injected_visual_features,
        )

    @torch.no_grad()
    def predict_best_mask(
        self,
        image_embeddings: torch.Tensor,
        language_hidden_states: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        mask_inputs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.forward(
            image_embeddings=image_embeddings,
            language_hidden_states=language_hidden_states,
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=boxes,
            mask_inputs=mask_inputs,
            multimask_output=True,
        )
        best_idx = outputs.iou_predictions.argmax(dim=-1)
        batch_idx = torch.arange(outputs.low_res_masks.size(0), device=outputs.low_res_masks.device)
        best_masks = outputs.low_res_masks[batch_idx, best_idx][:, None, ...]
        best_scores = outputs.iou_predictions[batch_idx, best_idx][:, None]
        return best_masks, best_scores

