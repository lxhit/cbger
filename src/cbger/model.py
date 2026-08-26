from __future__ import annotations

import math
import torch
from torch import nn


class CBGER(nn.Module):
    """Exact model used to generate the released three-seed CBGER results.

    Segment scores implement the Where pathway. The paper's final Whether score
    is the arithmetic mean of all segment scores and is applied at evaluation.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256,
                 dropout: float = 0.1, raw_residual: bool = True) -> None:
        super().__init__()
        self.energy_head = False
        self.raw_residual = raw_residual
        block = lambda: nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.user_projection, self.item_projection = block(), block()
        self.compatibility_user, self.compatibility_item = block(), block()
        self.relative_temporal = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.raw_blend_logit = nn.Parameter(torch.tensor(1.0))
        self.temporal_gate_logit = nn.Parameter(torch.tensor(-3.0))

    @staticmethod
    def _relative_features(items: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(items[:1])
        left, right = torch.cat((zeros, items[:-1])), torch.cat((items[1:], zeros))
        boundary = torch.zeros((items.shape[0], 2), device=items.device, dtype=items.dtype)
        boundary[0, 0], boundary[-1, 1] = 1, 1
        return torch.cat((items-left, items-right, duration.unsqueeze(-1), boundary), -1)

    def forward(self, interest_features, interest_types, interest_confidence,
                item_features, temporal_features, use_temporal=True,
                use_routing=False, return_aux=False):
        del interest_types, use_routing
        weight = interest_confidence / interest_confidence.sum().clamp_min(1e-8)
        profile = (interest_features * weight[:, None]).sum(0, keepdim=True)
        raw_user, raw_items = nn.functional.normalize(profile, dim=-1), nn.functional.normalize(item_features, dim=-1)
        user = nn.functional.normalize(self.user_projection(profile), dim=-1)
        items = nn.functional.normalize(self.item_projection(item_features), dim=-1)
        scale = self.logit_scale.exp().clamp(max=100)
        projected = (user @ items.T).squeeze(0) * scale
        raw_similarity = (raw_user @ raw_items.T).squeeze(0) * scale
        if self.raw_residual:
            blend = 0.9 + 0.1 * self.raw_blend_logit.sigmoid()
            semantic = blend * raw_similarity + (1 - blend) * projected
        else:
            blend, semantic = torch.zeros((), device=projected.device), projected
        relative_scores = torch.zeros_like(semantic)
        if use_temporal:
            relative = nn.functional.normalize(
                self.relative_temporal(self._relative_features(items, temporal_features[:, 1])), dim=-1
            )
            relative_scores = (relative * user).sum(-1) * scale
        temporal_gate = 0.02 * self.temporal_gate_logit.sigmoid()
        scores = semantic + temporal_gate * relative_scores
        c_user = nn.functional.normalize(self.compatibility_user(profile), dim=-1)
        c_items = nn.functional.normalize(self.compatibility_item(item_features), dim=-1)
        compatibility = (c_user @ c_items.T).squeeze(0) * scale
        routing = torch.ones((1, len(item_features)), device=item_features.device, dtype=item_features.dtype)
        if not return_aux:
            return scores
        return scores, {"routing": routing, "raw_clip_scores": raw_similarity,
            "projected_scores": projected, "semantic_scores": semantic,
            "relative_scores": relative_scores, "clip_blend": blend,
            "temporal_gate": temporal_gate, "compatibility_scores": compatibility,
            "background_scores": torch.zeros_like(compatibility),
            "evidence_energy": torch.logsumexp(compatibility, 0)}
