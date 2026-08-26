from __future__ import annotations

import math

import torch
from torch import nn


def _weighted_user(features, confidence):
    weights = confidence / confidence.sum().clamp_min(1e-8)
    return (weights.unsqueeze(-1) * features).sum(0)


class PRNetAdapter(nn.Module):
    """Candidate-conditioned multi-interest baseline inspired by PRNet."""

    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.user = nn.Linear(input_dim, hidden_dim)
        self.item = nn.Linear(input_dim, hidden_dim)
        self.scale = nn.Parameter(torch.tensor(math.log(1 / .07)))

    def forward(self, interests, types, confidence, items, temporal):
        users = nn.functional.normalize(self.user(interests), dim=-1)
        videos = nn.functional.normalize(self.item(items), dim=-1)
        similarity = users @ videos.T
        attention = (similarity + confidence.clamp_min(1e-8).log().unsqueeze(1)).softmax(0)
        return (attention * similarity).sum(0) * self.scale.exp().clamp(max=100)


class PRNetPaperFaithful(nn.Module):
    """Feature-adapted reproduction of PR-Net's equations 2--9.

    PBGER interest-slot embeddings stand in for the paper's historical-highlight
    embeddings, and PBGER segment embeddings stand in for frame embeddings.  The
    preference reasoning and bidirectional contrastive loss follow the paper,
    while its C3D/U-Net feature extractor is intentionally not reproduced.
    """

    def __init__(self, input_dim=512, hidden_dim=256, inverse_temperature=9.):
        super().__init__()
        self.history = nn.Linear(input_dim, hidden_dim)
        self.item = nn.Linear(input_dim, hidden_dim)
        self.context = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
        )
        self.generic_preference = nn.Parameter(torch.randn(hidden_dim) / math.sqrt(hidden_dim))
        self.nonhighlight_preference = nn.Parameter(
            torch.randn(hidden_dim) / math.sqrt(hidden_dim)
        )
        self.inverse_temperature = inverse_temperature

    def score_components(self, interests, items):
        history = nn.functional.normalize(self.history(interests), dim=-1)
        item = self.item(items)
        item = item + self.context(item.T.unsqueeze(0)).squeeze(0).T
        item = nn.functional.normalize(item, dim=-1)

        history_attention = (self.inverse_temperature * (item @ history.T)).softmax(-1)
        user_preference = nn.functional.normalize(history_attention @ history, dim=-1)
        generic = nn.functional.normalize(self.generic_preference, dim=-1)
        generic = generic.unsqueeze(0).expand_as(user_preference)
        user_weight = torch.exp(
            self.inverse_temperature * (item * user_preference).sum(-1)
        )
        generic_weight = torch.exp(self.inverse_temperature * (item * generic).sum(-1))
        denominator = (user_weight + generic_weight).clamp_min(1e-8)
        comprehensive = (
            user_weight.unsqueeze(-1) * user_preference
            + generic_weight.unsqueeze(-1) * generic
        ) / denominator.unsqueeze(-1)
        highlight = (item * comprehensive).sum(-1)
        nonhighlight = item @ nn.functional.normalize(
            self.nonhighlight_preference, dim=-1
        )
        return highlight, nonhighlight

    def forward(self, interests, types, confidence, items, temporal):
        del types, confidence, temporal
        highlight, nonhighlight = self.score_components(interests, items)
        return highlight - nonhighlight

    def bidirectional_contrastive_loss(
        self, interests, types, confidence, items, temporal, positive_index
    ):
        del types, confidence, temporal
        highlight, nonhighlight = self.score_components(interests, items)
        relative_highlight = highlight.softmax(0)
        relative_nonhighlight = nonhighlight.softmax(0)
        positive = torch.as_tensor([positive_index], device=items.device)
        negative_pool = torch.tensor(
            [index for index in range(len(items)) if index != positive_index],
            device=items.device,
        )
        k = min(5 * len(positive), len(negative_pool))
        negative = negative_pool[
            relative_nonhighlight[negative_pool].topk(k).indices
        ]
        positive_term = torch.log(
            relative_highlight[positive] + relative_nonhighlight[positive] + 1e-8
        ).sum()
        negative_term = torch.log(
            relative_highlight[negative] + relative_nonhighlight[negative] + 1e-8
        ).sum()
        attraction = relative_highlight[positive].sum() + relative_nonhighlight[negative].sum()
        return positive_term + negative_term - attraction


class SASRecAdapter(nn.Module):
    """History/self-attention baseline with a SASRec-style causal encoder."""

    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.user = nn.Linear(input_dim, hidden_dim)
        self.item = nn.Linear(input_dim, hidden_dim)
        self.type_embedding = nn.Embedding(3, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim * 2,
                                           dropout=.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.scale = nn.Parameter(torch.tensor(math.log(1 / .07)))

    def forward(self, interests, types, confidence, items, temporal):
        sequence = self.user(interests) + self.type_embedding(types)
        mask = torch.triu(
            torch.ones(len(sequence), len(sequence), device=sequence.device), 1
        ).bool()
        encoded = self.encoder(sequence.unsqueeze(0), mask=mask).squeeze(0)
        user = nn.functional.normalize(_weighted_user(encoded, confidence), dim=-1)
        videos = nn.functional.normalize(self.item(items), dim=-1)
        return (videos @ user) * self.scale.exp().clamp(max=100)


class QDDETRAdapter(nn.Module):
    """Query-dependent candidate decoder adapted from QD-DETR's core mechanism."""

    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.user = nn.Linear(input_dim, hidden_dim)
        self.item = nn.Linear(input_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, interests, types, confidence, items, temporal):
        query = _weighted_user(self.user(interests), confidence).view(1, 1, -1)
        videos = self.item(items).unsqueeze(0)
        conditioned, _ = self.cross_attention(videos, query, query)
        fused = self.norm(videos + conditioned + query)
        return self.head(fused).squeeze(0).squeeze(-1)


class UniVTGAdapter(nn.Module):
    """Local/global temporal grounding baseline adapted from UniVTG."""

    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.user = nn.Linear(input_dim, hidden_dim)
        self.item = nn.Linear(input_dim, hidden_dim)
        self.local = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.global_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.scale = nn.Parameter(torch.tensor(math.log(1 / .07)))

    def forward(self, interests, types, confidence, items, temporal):
        user = nn.functional.normalize(_weighted_user(self.user(interests), confidence), dim=-1)
        videos = self.item(items)
        local = self.local(videos.T.unsqueeze(0)).squeeze(0).T
        global_video = videos.mean(0).expand_as(videos)
        gate = self.global_gate(torch.cat((local, global_video), dim=-1))
        fused = gate * local + (1 - gate) * global_video + videos
        fused = nn.functional.normalize(fused, dim=-1)
        return (fused @ user) * self.scale.exp().clamp(max=100)


BASELINES = {
    "prnet_adapter": PRNetAdapter,
    "prnet_paper_faithful": PRNetPaperFaithful,
    "sasrec_adapter": SASRecAdapter,
    "qddetr_adapter": QDDETRAdapter,
    "univtg_adapter": UniVTGAdapter,
}
