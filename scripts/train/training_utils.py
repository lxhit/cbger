"""Train the position-robust candidate-conditioned multi-interest retriever."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from cbger.io import read_jsonl, write_jsonl
from cbger.paper_reference_model import MultiInterestTemporalRetriever, counterfactual_grounding_loss
from cbger.relocation import _relocate_evidence


def sample_tensors(sample, profiles, items, device):
    slots = profiles[str(sample["user_id"])]["slots"]
    cached = profiles[str(sample["user_id"])].get("_tensor_cache")
    if cached is None:
        interests = torch.tensor([slot["feature"] for slot in slots], device=device)
        types = torch.tensor([slot["type_id"] for slot in slots], dtype=torch.long, device=device)
        confidence = torch.tensor([slot["confidence"] for slot in slots], device=device)
    else:
        interests, types, confidence = cached
    item_tensor = torch.stack([items[row["segment_id"]] for row in sample["timeline"]]).to(device)
    total = max(float(row["target_interval"][1]) for row in sample["timeline"])
    temporal = torch.tensor(
        [[float(row["target_interval"][0]) / total,
          (float(row["target_interval"][1]) - float(row["target_interval"][0])) / total]
         for row in sample["timeline"]], device=device)
    return interests, types, confidence, item_tensor, temporal


def evidence_index(sample):
    return next(i for i, row in enumerate(sample["timeline"]) if row["role"] == "evidence")


def position_bucket(sample):
    index = evidence_index(sample)
    total = max(float(row["target_interval"][1]) for row in sample["timeline"])
    value = float(sample["timeline"][index]["target_interval"][0]) / max(total, 1e-8)
    return min(2, int(value * 3))


def position_balanced_order(samples, rng):
    buckets = defaultdict(list)
    for sample in samples:
        buckets[position_bucket(sample)].append(sample)
    target = max(len(bucket) for bucket in buckets.values())
    ordered = []
    for bucket_id in range(3):
        bucket = buckets[bucket_id]
        if not bucket:
            continue
        ordered.extend(bucket)
        ordered.extend(rng.choices(bucket, k=target - len(bucket)))
    rng.shuffle(ordered)
    return ordered


def score(model, sample, profiles, items, device, args, return_aux=False):
    return model(
        *sample_tensors(sample, profiles, items, device),
        use_temporal=not args.no_temporal,
        single_interest=args.single_interest,
        candidate_routing=not args.no_routing,
        suppress_absolute_position=not args.allow_absolute_position,
        return_aux=return_aux,
        adversarial_scale=args.adversarial_scale,
    )


def validation_mrr(model, samples, profiles, items, device, args):
    values = []
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            scores = score(model, sample, profiles, items, device, args)
            order = scores.argsort(descending=True).tolist()
            values.append(1 / (order.index(evidence_index(sample)) + 1))
    return sum(values) / max(1, len(values))


def write_predictions(model, samples, profiles, items, device, args, output):
    records = []
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            scores, aux = score(model, sample, profiles, items, device, args, True)
            scores = scores.cpu()
            candidates = [
                {**row, "score": float(scores[index])}
                for index, row in enumerate(sample["timeline"])
            ]
            candidates.sort(key=lambda row: (-row["score"], row["segment_id"]))
            total = max(row["target_interval"][1] for row in sample["timeline"])
            selected = candidates[0]["target_interval"]
            records.append({
                "sample_id": sample["sample_id"], "split": sample["split"],
                "user_id": sample["user_id"], "relevance": sample["relevance"],
                "counterfactual_of": sample.get("counterfactual_of"),
                "ranked_candidates": candidates,
                "sample_score": float(torch.logsumexp(scores, dim=0)),
                "selected_duration_fraction": (selected[1] - selected[0]) / total,
                "mean_router_entropy": float(
                    -(aux["routing"] * aux["routing"].clamp_min(1e-8).log())
                    .sum(dim=0).mean().cpu()
                ),
                "representation": "counterfactual_interest_router",
            })
    write_jsonl(output, records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--pair-weight", type=float, default=0.5)
    parser.add_argument("--necessity-weight", type=float, default=0.5)
    parser.add_argument("--sufficiency-weight", type=float, default=0.5)
    parser.add_argument("--sparse-weight", type=float, default=0.02)
    parser.add_argument("--position-weight", type=float, default=0.2)
    parser.add_argument("--invariance-weight", type=float, default=0.5)
    parser.add_argument("--personalization-weight", type=float, default=1.0)
    parser.add_argument("--adversarial-scale", type=float, default=1.0)
    parser.add_argument("--single-interest", action="store_true")
    parser.add_argument("--no-temporal", action="store_true")
    parser.add_argument("--no-routing", action="store_true")
    parser.add_argument("--no-position-balanced", action="store_true")
    parser.add_argument("--no-position-loss", action="store_true")
    parser.add_argument("--no-invariance-loss", action="store_true")
    parser.add_argument("--allow-absolute-position", action="store_true")
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = list(read_jsonl(args.dataset))
    paired = {row["counterfactual_of"]: row for row in rows if row.get("counterfactual_of")}
    profiles = {str(row["user_id"]): row for row in read_jsonl(args.profiles)}
    items = {row["segment_id"]: torch.tensor(row["feature"], dtype=torch.float32)
             for row in read_jsonl(args.features)}
    train = [row for row in rows if row["split"] == "train" and row["relevance"] == 1]
    validation = [row for row in rows if row["split"] == "validation" and row["relevance"] == 1]
    model = MultiInterestTemporalRetriever(
        len(next(iter(items.values()))), args.hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_mrr, best_epoch = -1.0, 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    profile_users = sorted(profiles)
    for epoch in range(1, args.epochs + 1):
        ordered = list(train)
        if args.no_position_balanced:
            rng.shuffle(ordered)
        else:
            ordered = position_balanced_order(ordered, rng)
        totals = defaultdict(float)
        model.train()
        for positive in ordered:
            positive_scores, aux = score(model, positive, profiles, items, device, args, True)
            counterfactual_scores = score(
                model, paired[positive["sample_id"]], profiles, items, device, args
            )
            ev_index = evidence_index(positive)
            relocated_scores = None
            relocated_index = None
            if not args.no_invariance_loss:
                targets = [0, len(positive["timeline"]) // 2, len(positive["timeline"]) - 1]
                current = ev_index
                choices = [target for target in targets if target != current] or targets
                relocated = _relocate_evidence(positive, rng.choice(choices))
                relocated_scores = score(model, relocated, profiles, items, device, args)
                relocated_index = evidence_index(relocated)
            temporal = sample_tensors(positive, profiles, items, device)[-1]
            position_labels = (temporal[:, 0] * 3).long().clamp(max=2)
            wrong_user = rng.choice(profile_users)
            while wrong_user == str(positive["user_id"]):
                wrong_user = rng.choice(profile_users)
            wrong_profile_sample = {**positive, "user_id": wrong_user}
            wrong_profile_scores = score(
                model, wrong_profile_sample, profiles, items, device, args
            )
            loss, parts = counterfactual_grounding_loss(
                positive_scores=positive_scores,
                evidence_index=ev_index,
                counterfactual_scores=counterfactual_scores,
                routing=aux["routing"],
                position_logits=None if args.no_position_loss else aux["position_logits"],
                position_labels=None if args.no_position_loss else position_labels,
                relocated_scores=relocated_scores,
                relocated_evidence_index=relocated_index,
                wrong_profile_scores=wrong_profile_scores,
                pair_weight=args.pair_weight,
                necessity_weight=args.necessity_weight,
                sufficiency_weight=args.sufficiency_weight,
                sparse_weight=args.sparse_weight,
                position_weight=args.position_weight,
                invariance_weight=args.invariance_weight,
                personalization_weight=args.personalization_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for key, value in parts.items():
                totals[key] += value
        val_mrr = validation_mrr(model, validation, profiles, items, device, args)
        record = {"epoch": epoch, **{k: v / max(1, len(ordered)) for k, v in totals.items()},
                  "validation_mrr": val_mrr, "epoch_samples": len(ordered)}
        history.append(record)
        print(json.dumps(record))
        if val_mrr > best_mrr:
            best_mrr, best_epoch = val_mrr, epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    write_predictions(model, rows, profiles, items, device, args, args.output)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "history": history,
                "best_epoch": best_epoch, "best_validation_mrr": best_mrr,
                "input_dim": len(next(iter(items.values()))), "hidden_dim": args.hidden_dim,
                "args": vars(args)}, args.checkpoint)
    args.output.with_suffix(".training.json").write_text(
        json.dumps({"history": history, "best_epoch": best_epoch,
                    "best_validation_mrr": best_mrr}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
