import math

import torch
from torch import nn


class VisionLanguageKnowledgeInjector(nn.Module):
    """Inject language semantics into dense visual features for MRIS decoding."""

    def __init__(
        self,
        visual_dim: int = 256,
        language_dim: int = 2048,
        downsample_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        patch_expand = int(1 / downsample_ratio) ** 2
        fused_visual_dim = visual_dim * patch_expand

        self.downsample_ratio = downsample_ratio
        self.visual_to_language = nn.Sequential(
            nn.LayerNorm(fused_visual_dim),
            nn.Linear(fused_visual_dim, language_dim),
            nn.GELU(),
            nn.Linear(language_dim, language_dim),
        )
        self.language_to_visual = nn.Sequential(
            nn.LayerNorm(language_dim),
            nn.Linear(language_dim, fused_visual_dim),
            nn.GELU(),
            nn.Linear(fused_visual_dim, fused_visual_dim),
        )
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(language_dim * 2),
            nn.Linear(language_dim * 2, language_dim),
            nn.GELU(),
            nn.Linear(language_dim, language_dim),
            nn.Sigmoid(),
        )

    def _pixel_unshuffle(self, x: torch.Tensor) -> torch.Tensor:
        n, h, w, c = x.size()
        scale = self.downsample_ratio
        x = x.reshape(n, h, int(w * scale), int(c / scale))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(n, int(w * scale), int(h * scale), int(c / (scale * scale)))
        return x.permute(0, 2, 1, 3).contiguous()

    def _pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        n, h, w, c = x.size()
        scale = self.downsample_ratio
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(n, h, int(w / scale), int(c * scale))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(n, int(w / scale), int(h / scale), int(c * (scale * scale)))
        return x

    def visual_tokens(self, dense_visual_features: torch.Tensor) -> torch.Tensor:
        features = dense_visual_features.permute(0, 2, 3, 1).contiguous()
        features = self._pixel_unshuffle(features)
        features = features.reshape(features.shape[0], -1, features.shape[-1])
        return self.visual_to_language(features)

    def inject(self, dense_visual_features: torch.Tensor, language_hidden_states: torch.Tensor) -> torch.Tensor:
        visual_tokens = self.visual_tokens(dense_visual_features)

        if language_hidden_states.size(1) != visual_tokens.size(1):
            target_len = visual_tokens.size(1)
            language_hidden_states = language_hidden_states[:, :target_len, :]
            if language_hidden_states.size(1) < target_len:
                pad_len = target_len - language_hidden_states.size(1)
                language_hidden_states = torch.cat(
                    [language_hidden_states, language_hidden_states[:, -1:, :].repeat(1, pad_len, 1)],
                    dim=1,
                )

        gate = self.fusion_gate(torch.cat([visual_tokens, language_hidden_states], dim=-1))
        fused_tokens = gate * language_hidden_states + (1.0 - gate) * visual_tokens
        projected = self.language_to_visual(fused_tokens)

        spatial_size = int(math.sqrt(projected.shape[1]))
        projected = projected.reshape(projected.shape[0], spatial_size, spatial_size, projected.shape[2])
        projected = self._pixel_shuffle(projected)
        return projected.permute(0, 3, 1, 2).contiguous()

