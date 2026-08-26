from __future__ import annotations

import math

import torch
from torch import nn


class PBGERv2Retriever(nn.Module):
    """CLIP-residual semantic retrieval plus relative temporal context."""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        energy_head: bool = False,
        shared_compatibility: bool = False,
        raw_residual: bool = True,
    ):
        super().__init__()
        self.energy_head = energy_head
        self.shared_compatibility = shared_compatibility
        self.raw_residual = raw_residual
        self.user_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.item_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.interest_type = nn.Embedding(3, hidden_dim)
        self.router_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.router_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.compatibility_user = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.compatibility_item = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.relative_temporal = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.clip_blend_logit = nn.Parameter(torch.tensor(1.0))
        self.temporal_gate_logit = nn.Parameter(torch.tensor(-3.0))
        if energy_head:
            self.background_item = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.background_item[-1].weight)
            nn.init.zeros_(self.background_item[-1].bias)

    @staticmethod
    def _relative_features(items: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(items[:1])
        left = torch.cat((zeros, items[:-1]), dim=0)
        right = torch.cat((items[1:], zeros), dim=0)
        left_delta = items - left
        right_delta = items - right
        boundary = torch.zeros((items.shape[0], 2), device=items.device, dtype=items.dtype)
        boundary[0, 0] = 1
        boundary[-1, 1] = 1
        return torch.cat((left_delta, right_delta, duration.unsqueeze(-1), boundary), dim=-1)

    def forward(
        self,
        interest_features: torch.Tensor,
        interest_types: torch.Tensor,
        interest_confidence: torch.Tensor,
        item_features: torch.Tensor,
        temporal_features: torch.Tensor,
        use_temporal: bool = True,
        use_routing: bool = True,
        return_aux: bool = False,
    ):
        raw_interests = nn.functional.normalize(interest_features, dim=-1)
        raw_items = nn.functional.normalize(item_features, dim=-1)
        interests = self.user_projection(interest_features) + self.interest_type(interest_types)
        interests = nn.functional.normalize(interests, dim=-1)
        items = nn.functional.normalize(self.item_projection(item_features), dim=-1)
        confidence_bias = interest_confidence.clamp_min(1e-4).log().unsqueeze(-1)
        if use_routing:
            router_logits = (
                self.router_query(interests) @ self.router_key(items).T
            ) / math.sqrt(items.shape[-1]) + confidence_bias
            routing = router_logits.softmax(dim=0)
        else:
            routing = interest_confidence.unsqueeze(-1).expand(-1, items.shape[0])
            routing = routing / routing.sum(dim=0, keepdim=True).clamp_min(1e-8)
        scale = self.logit_scale.exp().clamp(max=100)
        projected = (routing * (interests @ items.T)).sum(dim=0) * scale
        raw_routing = interest_confidence.unsqueeze(-1).expand(-1, raw_items.shape[0])
        raw_routing = raw_routing / raw_routing.sum(dim=0, keepdim=True).clamp_min(1e-8)
        raw_clip = (raw_routing * (raw_interests @ raw_items.T)).sum(dim=0) * scale
        if self.raw_residual:
            blend = 0.9 + 0.1 * self.clip_blend_logit.sigmoid()
            semantic = blend * raw_clip + (1 - blend) * projected
        else:
            # Heterogeneous frozen encoders (for example BGE text + DINOv2
            # vision) do not share a meaningful raw cosine space.  In that
            # setting the independently learned user/item projections are the
            # semantic scorer; retaining raw_clip would inject a false teacher.
            blend = torch.zeros((), device=projected.device, dtype=projected.dtype)
            semantic = projected
        relative_scores = torch.zeros_like(semantic)
        if use_temporal:
            relative = self.relative_temporal(
                self._relative_features(items, temporal_features[:, 1])
            )
            routed_user = routing.T @ interests
            routed_user = nn.functional.normalize(routed_user, dim=-1)
            relative = nn.functional.normalize(relative, dim=-1)
            relative_scores = (routed_user * relative).sum(dim=-1) * scale
        temporal_gate = 0.02 * self.temporal_gate_logit.sigmoid()
        scores = semantic + temporal_gate * relative_scores
        if self.shared_compatibility:
            compatibility_interests = interests
            compatibility_items = items
        else:
            compatibility_interests = nn.functional.normalize(
                self.compatibility_user(interest_features), dim=-1
            )
            compatibility_items = nn.functional.normalize(
                self.compatibility_item(item_features), dim=-1
            )
        compatibility_scores = (
            routing.detach() * (compatibility_interests @ compatibility_items.T)
        ).sum(dim=0) * scale
        background_scores = torch.zeros_like(compatibility_scores)
        if self.energy_head:
            background_scores = self.background_item(item_features).squeeze(-1)
        compatibility_energy = torch.logsumexp(compatibility_scores, dim=0)
        evidence_energy = compatibility_energy
        if self.energy_head:
            evidence_energy = compatibility_energy - torch.logsumexp(
                background_scores, dim=0
            )
        if not return_aux:
            return scores
        return scores, {
            "routing": routing,
            "raw_clip_scores": raw_clip,
            "projected_scores": projected,
            "semantic_scores": semantic,
            "relative_scores": relative_scores,
            "clip_blend": blend,
            "temporal_gate": temporal_gate,
            "compatibility_scores": compatibility_scores,
            "background_scores": background_scores,
            "evidence_energy": evidence_energy,
        }


def distillation_loss(student: torch.Tensor, teacher: torch.Tensor, temperature: float = 1.0):
    teacher_prob = (teacher / temperature).softmax(dim=0)
    return nn.functional.kl_div(
        (student / temperature).log_softmax(dim=0), teacher_prob, reduction="batchmean"
    ) * temperature**2


def v2_losses(
    scores: torch.Tensor,
    evidence_index: int,
    aux: dict[str, torch.Tensor],
    counterfactual_scores: torch.Tensor | None = None,
    counterfactual_compatibility: torch.Tensor | None = None,
    wrong_profile_scores: torch.Tensor | None = None,
    relocated_scores: torch.Tensor | None = None,
    relocated_index: int | None = None,
    counterfactual_energy: torch.Tensor | None = None,
    wrong_profile_energy: torch.Tensor | None = None,
    relocated_energy: torch.Tensor | None = None,
    margin: float = 0.2,
) -> dict[str, torch.Tensor]:
    device = scores.device
    target = torch.tensor([evidence_index], device=device)
    distractors = torch.cat((scores[:evidence_index], scores[evidence_index + 1 :]))
    result = {
        "retrieval": nn.functional.cross_entropy(scores.unsqueeze(0), target),
        "distill": distillation_loss(scores, aux["raw_clip_scores"].detach()),
        "necessity": nn.functional.relu(
            margin - (torch.logsumexp(scores, 0) - torch.logsumexp(distractors, 0))
        ),
        "sufficiency": nn.functional.relu(
            margin - scores[evidence_index]
            + torch.logsumexp(distractors, 0) - math.log(distractors.numel())
        ),
        "pair": torch.zeros((), device=device),
        "replacement": torch.zeros((), device=device),
        "shared_invariance": torch.zeros((), device=device),
        "personalization": torch.zeros((), device=device),
        "relocation": torch.zeros((), device=device),
        "energy_content": torch.zeros((), device=device),
        "energy_behavior": torch.zeros((), device=device),
        "energy_relocation": torch.zeros((), device=device),
        "router_entropy": -(
            aux["routing"] * aux["routing"].clamp_min(1e-8).log()
        ).sum(dim=0).mean() / math.log(max(2, aux["routing"].shape[0])),
    }
    if counterfactual_scores is not None:
        positive_compatibility = torch.logsumexp(aux["compatibility_scores"], dim=0)
        negative_compatibility = torch.logsumexp(
            counterfactual_compatibility
            if counterfactual_compatibility is not None
            else counterfactual_scores,
            dim=0,
        )
        result["pair"] = nn.functional.relu(
            margin - positive_compatibility + negative_compatibility
        )
        if counterfactual_compatibility is not None:
            result["replacement"] = nn.functional.relu(
                margin
                - aux["compatibility_scores"][evidence_index]
                + counterfactual_compatibility[evidence_index]
            )
        if scores.numel() > 1:
            shared_mask = torch.ones(scores.numel(), dtype=torch.bool, device=device)
            shared_mask[evidence_index] = False
            result["shared_invariance"] = nn.functional.l1_loss(
                scores[shared_mask], counterfactual_scores[shared_mask]
            )
    if wrong_profile_scores is not None:
        result["personalization"] = nn.functional.relu(
            margin - scores[evidence_index] + wrong_profile_scores[evidence_index]
        )
    if relocated_scores is not None and relocated_index is not None:
        relocation_retrieval = nn.functional.cross_entropy(
            relocated_scores.unsqueeze(0), torch.tensor([relocated_index], device=device)
        )
        score_consistency = nn.functional.smooth_l1_loss(
            scores[evidence_index], relocated_scores[relocated_index]
        )
        result["relocation"] = relocation_retrieval + score_consistency
    if counterfactual_energy is not None:
        result["energy_content"] = nn.functional.relu(
            margin - aux["evidence_energy"] + counterfactual_energy
        )
    if wrong_profile_energy is not None:
        result["energy_behavior"] = nn.functional.relu(
            margin - aux["evidence_energy"] + wrong_profile_energy
        )
    if relocated_energy is not None:
        result["energy_relocation"] = nn.functional.smooth_l1_loss(
            aux["evidence_energy"], relocated_energy
        )
    return result
