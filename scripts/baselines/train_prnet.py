from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import torch
from training_utils import evidence_index, sample_tensors

from cbger.io import read_jsonl, write_jsonl
from cbger.baselines import BASELINES


def preload_feature_tensors(profiles, items, device):
    """Move immutable feature tensors once without changing training math."""
    items = {key: value.to(device) for key, value in items.items()}
    for profile in profiles.values():
        slots = profile["slots"]
        profile["_tensor_cache"] = (
            torch.tensor([slot["feature"] for slot in slots], device=device),
            torch.tensor(
                [slot["type_id"] for slot in slots], dtype=torch.long, device=device
            ),
            torch.tensor([slot["confidence"] for slot in slots], device=device),
        )
    return profiles, items


def predict(model, rows, profiles, items, device, name, output):
    records = []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            scores = model(*sample_tensors(row, profiles, items, device)).cpu()
            ranked = [{**candidate, "score": float(scores[index])}
                      for index, candidate in enumerate(row["timeline"])]
            ranked.sort(key=lambda value: (-value["score"], value["segment_id"]))
            total = max(value["target_interval"][1] for value in row["timeline"])
            selected = ranked[0]["target_interval"]
            records.append({
                "sample_id": row["sample_id"], "split": row["split"],
                "user_id": row["user_id"], "relevance": row["relevance"],
                "counterfactual_of": row.get("counterfactual_of"),
                "ranked_candidates": ranked,
                "sample_score": float(torch.logsumexp(scores, 0)),
                "selected_duration_fraction": (selected[1] - selected[0]) / total,
                "representation": name,
            })
    write_jsonl(output, records)


def validation_mrr(model, rows, profiles, items, device):
    model.eval()
    values = []
    with torch.inference_mode():
        for row in rows:
            scores = model(*sample_tensors(row, profiles, items, device))
            order = scores.argsort(descending=True).tolist()
            values.append(1 / (order.index(evidence_index(row)) + 1))
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(BASELINES), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--pair-weight", type=float, default=.5)
    parser.add_argument("--paper-loss-weight", type=float, default=1.)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-validation", type=int, default=0)
    parser.add_argument("--max-predict", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--preload-features", action="store_true")
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = list(read_jsonl(args.dataset))
    profiles = {str(row["user_id"]): row for row in read_jsonl(args.profiles)}
    items = {row["segment_id"]: torch.tensor(row["feature"], dtype=torch.float32)
             for row in read_jsonl(args.features)}
    if args.preload_features:
        profiles, items = preload_feature_tensors(profiles, items, device)
    paired = {row["counterfactual_of"]: row for row in rows if row.get("counterfactual_of")}
    train = [row for row in rows if row["split"] == "train" and row["relevance"] == 1]
    validation = [row for row in rows
                  if row["split"] == "validation" and row["relevance"] == 1]
    if args.max_train:
        train = train[:args.max_train]
    if args.max_validation:
        validation = validation[:args.max_validation]
    dim = len(next(iter(items.values())))
    model = BASELINES[args.model](dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best, best_state, history = -1., None, []
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train)
        model.train()
        total = 0.
        for positive in train:
            positive_scores = model(*sample_tensors(positive, profiles, items, device))
            negative_scores = model(*sample_tensors(
                paired[positive["sample_id"]], profiles, items, device))
            retrieval = torch.nn.functional.cross_entropy(
                positive_scores.unsqueeze(0),
                torch.tensor([evidence_index(positive)], device=device))
            pair = torch.nn.functional.relu(
                .2 - torch.logsumexp(positive_scores, 0) + torch.logsumexp(negative_scores, 0))
            loss = retrieval + args.pair_weight * pair
            if hasattr(model, "bidirectional_contrastive_loss"):
                paper_loss = model.bidirectional_contrastive_loss(
                    *sample_tensors(positive, profiles, items, device),
                    evidence_index(positive),
                )
                loss = loss + args.paper_loss_weight * paper_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())
        mrr = validation_mrr(model, validation, profiles, items, device)
        record = {"epoch": epoch, "loss": total / len(train), "validation_mrr": mrr}
        history.append(record)
        print(json.dumps(record), flush=True)
        if mrr > best:
            best, best_state = mrr, copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if args.early_stop_patience and stale_epochs >= args.early_stop_patience:
            print(json.dumps({"early_stop_epoch": epoch, "best_validation_mrr": best}),
                  flush=True)
            break
    model.load_state_dict(best_state)
    prediction_rows = rows[:args.max_predict] if args.max_predict else rows
    predict(model, prediction_rows, profiles, items, device, args.model, args.output)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": dim,
                "hidden_dim": args.hidden_dim, "model": args.model,
                "best_validation_mrr": best, "history": history}, args.checkpoint)


if __name__ == "__main__":
    main()
