from __future__ import annotations

import math

import torch
from torch import nn


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None


def gradient_reverse(value: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(value, scale)


class MultiInterestTemporalRetriever(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.user_projection = nn.Linear(input_dim, hidden_dim)
        self.item_projection = nn.Linear(input_dim, hidden_dim)
        self.interest_type = nn.Embedding(3, hidden_dim)
        self.temporal_projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.router_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.router_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.position_adversary = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.temperature = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(
        self,
        interest_features: torch.Tensor,
        interest_types: torch.Tensor,
        interest_confidence: torch.Tensor,
        item_features: torch.Tensor,
        temporal_features: torch.Tensor,
        use_temporal: bool = True,
        single_interest: bool = False,
        candidate_routing: bool = True,
        suppress_absolute_position: bool = False,
        return_aux: bool = False,
        adversarial_scale: float = 1.0,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        interests = self.user_projection(interest_features)
        interests = interests + self.interest_type(interest_types)
        interests = nn.functional.normalize(interests, dim=-1)
        if single_interest:
            weights = interest_confidence / interest_confidence.sum().clamp_min(1e-6)
            interests = (interests * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
            interest_confidence = torch.ones_like(interest_confidence[:1])

        items = self.item_projection(item_features)
        if use_temporal:
            temporal_input = temporal_features
            if suppress_absolute_position:
                temporal_input = torch.stack(
                    (torch.zeros_like(temporal_features[:, 0]), temporal_features[:, 1]),
                    dim=-1,
                )
            items = items + self.temporal_projection(temporal_input)
        items = self.temporal_encoder(items.unsqueeze(0)).squeeze(0)
        items = nn.functional.normalize(items, dim=-1)
        scale = self.temperature.exp().clamp(max=100)
        similarities = interests @ items.transpose(0, 1) * scale
        confidence_bias = interest_confidence.clamp_min(1e-4).log().unsqueeze(-1)
        if candidate_routing and not single_interest:
            router_logits = (
                self.router_query(interests) @ self.router_key(items).transpose(0, 1)
            ) / math.sqrt(items.shape[-1])
            router_logits = router_logits + confidence_bias
            routing = router_logits.softmax(dim=0)
            scores = (routing * similarities).sum(dim=0)
        else:
            routing = (similarities + confidence_bias).softmax(dim=0)
            scores = torch.logsumexp(similarities + confidence_bias, dim=0)
        if not return_aux:
            return scores
        position_logits = self.position_adversary(
            gradient_reverse(items, adversarial_scale)
        )
        return scores, {
            "routing": routing,
            "position_logits": position_logits,
            "item_embeddings": items,
        }


def joint_retrieval_loss(
    positive_scores: torch.Tensor,
    evidence_index: int,
    counterfactual_scores: torch.Tensor,
    pair_weight: float = 0.5,
    removal_weight: float = 0.5,
    sparse_weight: float = 0.02,
    margin: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    retrieval = nn.functional.cross_entropy(
        positive_scores.unsqueeze(0),
        torch.tensor([evidence_index], device=positive_scores.device),
    )
    pair = nn.functional.relu(
        margin
        - torch.logsumexp(positive_scores, dim=0)
        + torch.logsumexp(counterfactual_scores, dim=0)
    )
    distractors = torch.cat(
        (positive_scores[:evidence_index], positive_scores[evidence_index + 1 :])
    )
    removal = nn.functional.relu(
        margin - positive_scores[evidence_index] + distractors.max()
    )
    probabilities = positive_scores.softmax(dim=0)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum()
    entropy = entropy / math.log(max(2, positive_scores.numel()))
    loss = retrieval + pair_weight * pair + removal_weight * removal + sparse_weight * entropy
    return loss, {
        "retrieval": retrieval.item(),
        "pair": pair.item(),
        "removal": removal.item(),
        "entropy": entropy.item(),
    }


def counterfactual_grounding_loss(
    positive_scores: torch.Tensor,
    evidence_index: int,
    counterfactual_scores: torch.Tensor,
    routing: torch.Tensor,
    position_logits: torch.Tensor | None = None,
    position_labels: torch.Tensor | None = None,
    relocated_scores: torch.Tensor | None = None,
    relocated_evidence_index: int | None = None,
    wrong_profile_scores: torch.Tensor | None = None,
    pair_weight: float = 0.5,
    necessity_weight: float = 0.5,
    sufficiency_weight: float = 0.5,
    sparse_weight: float = 0.02,
    position_weight: float = 0.2,
    invariance_weight: float = 0.5,
    personalization_weight: float = 0.5,
    margin: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Joint objective for retrieval, evidence faithfulness and position invariance."""
    target = torch.tensor([evidence_index], device=positive_scores.device)
    retrieval = nn.functional.cross_entropy(positive_scores.unsqueeze(0), target)
    evidence_score = positive_scores[evidence_index]
    distractors = torch.cat(
        (positive_scores[:evidence_index], positive_scores[evidence_index + 1 :])
    )
    pair = nn.functional.relu(margin - evidence_score + counterfactual_scores.max())

    full_utility = torch.logsumexp(positive_scores, dim=0)
    deleted_utility = torch.logsumexp(distractors, dim=0)
    necessity = nn.functional.relu(margin - (full_utility - deleted_utility))
    distractor_mean = torch.logsumexp(distractors, dim=0) - math.log(distractors.numel())
    sufficiency = nn.functional.relu(margin - evidence_score + distractor_mean)

    router_entropy = -(routing * routing.clamp_min(1e-8).log()).sum(dim=0).mean()
    router_entropy = router_entropy / math.log(max(2, routing.shape[0]))
    position = torch.zeros((), device=positive_scores.device)
    if position_logits is not None and position_labels is not None:
        position = nn.functional.cross_entropy(position_logits, position_labels)
    invariance = torch.zeros((), device=positive_scores.device)
    relocated_retrieval = torch.zeros((), device=positive_scores.device)
    if relocated_scores is not None and relocated_evidence_index is not None:
        original_probability = positive_scores.softmax(dim=0)[evidence_index]
        relocated_probability = relocated_scores.softmax(dim=0)[relocated_evidence_index]
        relocated_retrieval = nn.functional.cross_entropy(
            relocated_scores.unsqueeze(0),
            torch.tensor([relocated_evidence_index], device=positive_scores.device),
        )
        invariance = nn.functional.smooth_l1_loss(
            positive_scores[evidence_index], relocated_scores[relocated_evidence_index]
        ) + nn.functional.smooth_l1_loss(original_probability, relocated_probability)
    personalization = torch.zeros((), device=positive_scores.device)
    if wrong_profile_scores is not None:
        personalization = nn.functional.relu(
            margin - positive_scores[evidence_index] + wrong_profile_scores[evidence_index]
        )

    loss = (
        retrieval
        + pair_weight * pair
        + necessity_weight * necessity
        + sufficiency_weight * sufficiency
        + sparse_weight * router_entropy
        + position_weight * position
        + invariance_weight * (invariance + relocated_retrieval)
        + personalization_weight * personalization
    )
    return loss, {
        "retrieval": retrieval.item(),
        "pair": pair.item(),
        "necessity": necessity.item(),
        "sufficiency": sufficiency.item(),
        "router_entropy": router_entropy.item(),
        "position_adversary": position.item(),
        "position_invariance": invariance.item(),
        "relocated_retrieval": relocated_retrieval.item(),
        "personalization": personalization.item(),
    }
