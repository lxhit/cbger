"""Exact three-stage CBGER training used for the released main results."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from training_utils import evidence_index, position_balanced_order, sample_tensors

from cbger.io import read_jsonl, write_jsonl
from cbger.model import CBGER
from cbger.losses import v2_losses
from cbger.relocation import _relocate_evidence


def score(model, sample, profiles, items, device, args, return_aux=False):
    tensors = list(sample_tensors(sample, profiles, items, device))
    if getattr(args, "single_interest", False):
        features, types, confidence = tensors[:3]
        weights = confidence / confidence.sum().clamp_min(1e-8)
        tensors[0] = (features * weights[:, None]).sum(0, keepdim=True)
        tensors[1] = torch.full((1,), 2, dtype=types.dtype, device=types.device)
        tensors[2] = torch.ones((1,), dtype=confidence.dtype, device=confidence.device)
    return model(
        *tensors,
        use_temporal=not args.no_temporal,
        use_routing=not args.no_routing,
        return_aux=return_aux,
    )


def existence_score(aux):
    """Return the intervention-consistent energy when the head is enabled."""
    return aux.get("evidence_energy", torch.logsumexp(aux["compatibility_scores"], 0))


def load_disagreement_manifest(path):
    if path is None:
        return {}
    return {row["sample_id"]: row for row in read_jsonl(path)}


def disagreement_augmented_order(
    samples, manifest, top_fraction, extra_repeats, rng
):
    ordered = list(samples)
    if not manifest or extra_repeats <= 0:
        return ordered
    scored = [sample for sample in samples if sample["sample_id"] in manifest]
    scored.sort(
        key=lambda sample: manifest[sample["sample_id"]]["disagreement_score"],
        reverse=True,
    )
    count = max(1, round(len(scored) * top_fraction))
    hard = scored[:count]
    for _ in range(extra_repeats):
        ordered.extend(hard)
    rng.shuffle(ordered)
    return ordered


def hard_user_map(profiles: dict[str, dict], top_k: int) -> dict[str, list[str]]:
    users = sorted(profiles)
    vectors = []
    for user in users:
        slots = profiles[user]["slots"]
        features = torch.tensor([slot["feature"] for slot in slots])
        confidence = torch.tensor([slot["confidence"] for slot in slots])
        confidence = confidence / confidence.sum().clamp_min(1e-8)
        vector = (features * confidence[:, None]).sum(0)
        vectors.append(torch.nn.functional.normalize(vector, dim=0))
    matrix = torch.stack(vectors)
    similarities = matrix @ matrix.T
    similarities.fill_diagonal_(-1e9)
    neighbours = similarities.topk(min(top_k, len(users) - 1), dim=1).indices
    return {user: [users[index] for index in neighbours[row].tolist()]
            for row, user in enumerate(users)}


def preload_feature_tensors(profiles, items, device):
    """Move immutable feature tensors once without changing training math."""
    items = {key: value.to(device) for key, value in items.items()}
    for profile in profiles.values():
        slots = profile["slots"]
        profile["_tensor_cache"] = (
            torch.tensor([slot["feature"] for slot in slots], device=device),
            torch.tensor([slot["type_id"] for slot in slots], dtype=torch.long, device=device),
            torch.tensor([slot["confidence"] for slot in slots], device=device),
        )
    return profiles, items


def validation_metrics(model, samples, pairs, profiles, items, device, args):
    values = []
    retrieval_hits = []
    wrong_profile_hits = []
    pair_hits = []
    position_hits = defaultdict(list)
    model.eval()
    users = sorted(profiles)
    evaluate_profile_gain = args.personalization_weight > 0
    wrong_user = ({user: users[(index + 1) % len(users)]
                   for index, user in enumerate(users)}
                  if evaluate_profile_gain else {})
    with torch.inference_mode():
        for sample in samples:
            scores, aux = score(model, sample, profiles, items, device, args, True)
            order = scores.argsort(descending=True).tolist()
            values.append(1 / (order.index(evidence_index(sample)) + 1))
            retrieval_hits.append(int(order.index(evidence_index(sample)) == 0))
            if evaluate_profile_gain:
                wrong_scores = score(
                    model,
                    {**sample, "user_id": wrong_user[str(sample["user_id"])]},
                    profiles,
                    items,
                    device,
                    args,
                )
                wrong_order = wrong_scores.argsort(descending=True).tolist()
                wrong_profile_hits.append(
                    int(wrong_order.index(evidence_index(sample)) == 0)
                )
            _, cf_aux = score(
                model, pairs[sample["sample_id"]], profiles, items, device, args, True
            )
            pair_hits.append(int(
                existence_score(aux) > existence_score(cf_aux)
            ))
            targets = {"beginning": 0, "middle": len(sample["timeline"]) // 2,
                       "end": len(sample["timeline"]) - 1}
            for name, target in targets.items():
                moved = _relocate_evidence(sample, target)
                moved_scores = score(model, moved, profiles, items, device, args)
                moved_order = moved_scores.argsort(descending=True).tolist()
                position_hits[name].append(
                    int(moved_order.index(evidence_index(moved)) == 0)
                )
    mrr = sum(values) / max(1, len(values))
    pair_accuracy = sum(pair_hits) / max(1, len(pair_hits))
    position_r1 = {name: sum(hits) / max(1, len(hits))
                   for name, hits in position_hits.items()}
    position_gap = max(position_r1.values()) - min(position_r1.values())
    profile_gain = (sum(retrieval_hits) / max(1, len(retrieval_hits))
                    - sum(wrong_profile_hits) / len(wrong_profile_hits)
                    if wrong_profile_hits else 0.0)
    excess_gap = max(0.0, position_gap - args.selection_gap_threshold)
    composite = (
        mrr
        + args.selection_pair_weight * pair_accuracy
        - args.selection_gap_weight * excess_gap
    )
    return {"validation_mrr": mrr, "validation_pair_accuracy": pair_accuracy,
            "validation_position_gap": position_gap,
            "validation_worst_position_r1": min(position_r1.values()),
            "validation_profile_gain": profile_gain,
            "validation_composite": composite}


def _variant_path(path: Path, label: str) -> Path:
    return path.with_name(f"{path.stem}.{label}{path.suffix}")


def _pareto_epochs(history: list[dict]) -> list[int]:
    keys = (
        "validation_mrr",
        "validation_pair_accuracy",
        "validation_worst_position_r1",
    )
    selected = []
    for index, candidate in enumerate(history):
        dominated = False
        for other_index, other in enumerate(history):
            if index == other_index:
                continue
            weakly_better = all(other[key] >= candidate[key] for key in keys)
            strictly_better = any(other[key] > candidate[key] for key in keys)
            gap_better = other["validation_position_gap"] <= candidate["validation_position_gap"]
            gap_strict = other["validation_position_gap"] < candidate["validation_position_gap"]
            if weakly_better and gap_better and (strictly_better or gap_strict):
                dominated = True
                break
        if not dominated:
            selected.append(int(candidate["epoch"]))
    return selected


def _best_stage_epoch(history: list[dict], stage: str, gap_threshold: float) -> int:
    rows = [row for row in history if row["stage"] == stage]
    if not rows:
        raise ValueError(f"No epochs for stage {stage}")
    if stage == "semantic":
        best = max(rows, key=lambda row: (
            row["validation_mrr"], row["validation_pair_accuracy"]
        ))
    elif stage == "counterfactual":
        best = max(rows, key=lambda row: (
            row["validation_pair_accuracy"], row["validation_mrr"]
        ))
    else:
        feasible = [row for row in rows
                    if row["validation_position_gap"] <= gap_threshold]
        pool = feasible or rows
        best = max(pool, key=lambda row: (
            row["validation_worst_position_r1"],
            row["validation_pair_accuracy"],
            row["validation_mrr"],
            -row["validation_position_gap"],
        ))
    return int(best["epoch"])


def _checkpoint_payload(model, args, input_dim, history, epoch, selection):
    row = next(record for record in history if record["epoch"] == epoch)
    return {
        "model_type": "pbger_lite",
        "state_dict": copy.deepcopy(model.state_dict()),
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "args": vars(args),
        "history": history,
        "best_epoch": epoch,
        "selection": selection,
        "selection_metrics": row,
    }


def _backward_with_projected_distillation(
    model: torch.nn.Module,
    primary_loss: torch.Tensor,
    distill_loss: torch.Tensor,
    distill_weight: float,
) -> tuple[float, bool]:
    """Project only a conflicting distillation gradient off the primary gradient."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    primary_grads = torch.autograd.grad(
        primary_loss, parameters, retain_graph=True, allow_unused=True
    )
    distill_grads = torch.autograd.grad(
        distill_loss, parameters, allow_unused=True
    )
    dot = torch.zeros((), device=primary_loss.device)
    primary_norm = torch.zeros((), device=primary_loss.device)
    distill_norm = torch.zeros((), device=primary_loss.device)
    for primary, distill in zip(primary_grads, distill_grads, strict=True):
        if primary is not None:
            primary_norm = primary_norm + primary.square().sum()
        if distill is not None:
            distill_norm = distill_norm + distill.square().sum()
        if primary is not None and distill is not None:
            dot = dot + (primary * distill).sum()
    denominator = (primary_norm.sqrt() * distill_norm.sqrt()).clamp_min(1e-12)
    cosine = float((dot / denominator).detach())
    projected = bool(dot.detach() < 0 and primary_norm.detach() > 0)
    coefficient = dot / primary_norm.clamp_min(1e-12) if projected else dot.new_zeros(())
    for parameter, primary, distill in zip(
        parameters, primary_grads, distill_grads, strict=True
    ):
        gradient = torch.zeros_like(parameter)
        if primary is not None:
            gradient = gradient + primary
        if distill is not None:
            adjusted = distill
            if projected and primary is not None:
                adjusted = adjusted - coefficient * primary
            gradient = gradient + distill_weight * adjusted
        parameter.grad = gradient
    return cosine, projected


def predict(model, rows, profiles, items, device, args, output):
    records = []
    model.eval()
    with torch.inference_mode():
        for sample in rows:
            scores, aux = score(model, sample, profiles, items, device, args, True)
            scores = scores.cpu()
            candidates = [{**entry, "score": float(scores[i])}
                          for i, entry in enumerate(sample["timeline"])]
            candidates.sort(key=lambda row: (-row["score"], row["segment_id"]))
            total = max(row["target_interval"][1] for row in sample["timeline"])
            selected = candidates[0]["target_interval"]
            records.append({
                "sample_id": sample["sample_id"], "split": sample["split"],
                "user_id": sample["user_id"], "relevance": sample["relevance"],
                "counterfactual_of": sample.get("counterfactual_of"),
                "ranked_candidates": candidates,
                "sample_score": float(existence_score(aux).cpu()),
                "selected_duration_fraction": (selected[1] - selected[0]) / total,
                "clip_blend": float(aux["clip_blend"].cpu()),
                "temporal_gate": float(aux["temporal_gate"].cpu()),
                "representation": (
                    "pbger_lite_global_profile_decoupled_where_whether"
                    if getattr(model, "raw_residual", True)
                    else "pbger_lite_projected_global_profile"
                ),
            })
    write_jsonl(output, records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--counterfactual-epochs", type=int, default=2)
    parser.add_argument("--robust-epochs", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--counterfactual-distill-scale", type=float, default=1.0)
    parser.add_argument("--robust-distill-scale", type=float, default=1.0)
    parser.add_argument("--personalization-weight", type=float, default=1.0)
    parser.add_argument("--pair-weight", type=float, default=0.5)
    parser.add_argument("--replacement-weight", type=float, default=0.0)
    parser.add_argument("--shared-weight", type=float, default=0.0)
    parser.add_argument("--energy-head", action="store_true")
    parser.add_argument("--energy-content-weight", type=float, default=0.0)
    parser.add_argument("--energy-behavior-weight", type=float, default=0.0)
    parser.add_argument("--energy-relocation-weight", type=float, default=0.0)
    parser.add_argument("--disagreement-manifest", type=Path)
    parser.add_argument("--disagreement-top-fraction", type=float, default=0.25)
    parser.add_argument("--disagreement-extra-repeats", type=int, default=0)
    parser.add_argument("--project-conflicting-distill", action="store_true")
    parser.add_argument("--preload-features", action="store_true")
    parser.add_argument("--necessity-weight", type=float, default=0.5)
    parser.add_argument("--sufficiency-weight", type=float, default=0.5)
    parser.add_argument("--relocation-weight", type=float, default=0.5)
    parser.add_argument("--sparse-weight", type=float, default=0.01)
    parser.add_argument("--hard-users", type=int, default=16)
    parser.add_argument("--selection-pair-weight", type=float, default=0.2)
    parser.add_argument("--selection-gap-weight", type=float, default=0.3)
    parser.add_argument("--selection-gap-threshold", type=float, default=0.05)
    parser.add_argument("--no-routing", action="store_true")
    parser.add_argument("--no-temporal", action="store_true")
    parser.add_argument("--single-interest", action="store_true")
    parser.add_argument("--shared-compatibility", action="store_true")
    parser.add_argument("--disable-raw-residual", action="store_true")
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = list(read_jsonl(args.dataset))
    disagreement = load_disagreement_manifest(args.disagreement_manifest)
    pairs = {row["counterfactual_of"]: row for row in rows if row.get("counterfactual_of")}
    profiles = {str(row["user_id"]): row for row in read_jsonl(args.profiles)}
    items = {row["segment_id"]: torch.tensor(row["feature"], dtype=torch.float32)
             for row in read_jsonl(args.features)}
    train = [row for row in rows if row["split"] == "train" and row["relevance"] == 1]
    validation = [row for row in rows if row["split"] == "validation" and row["relevance"] == 1]
    hard_users = (hard_user_map(profiles, args.hard_users)
                  if args.personalization_weight > 0 else {})
    if args.preload_features:
        profiles, items = preload_feature_tensors(profiles, items, device)
    model = CBGER(
        len(next(iter(items.values()))), args.hidden_dim,
        raw_residual=not args.disable_raw_residual,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    stages = (["semantic"] * args.warmup_epochs
              + ["counterfactual"] * args.counterfactual_epochs
              + ["robust"] * args.robust_epochs)
    history, epoch_states = [], {}
    best_state, best_score, best_epoch = copy.deepcopy(model.state_dict()), -1e9, 0
    for epoch, stage in enumerate(stages, 1):
        base_order = position_balanced_order(train, rng) if stage == "robust" else list(train)
        ordered = disagreement_augmented_order(
            base_order,
            disagreement if stage != "semantic" else {},
            args.disagreement_top_fraction,
            args.disagreement_extra_repeats,
            rng,
        )
        rng.shuffle(ordered)
        totals = defaultdict(float)
        model.train()
        for positive in ordered:
            scores, aux = score(model, positive, profiles, items, device, args, True)
            wrong_scores, wrong_aux = None, None
            if args.personalization_weight > 0:
                user = str(positive["user_id"])
                wrong_pool = hard_users[user]
                if not wrong_pool:
                    wrong_pool = [candidate for candidate in profiles if candidate != user]
                wrong_user = rng.choice(wrong_pool)
                wrong_scores = score(
                    model, {**positive, "user_id": wrong_user}, profiles, items,
                    device, args
                )
            cf_scores, cf_compatibility, cf_aux = None, None, None
            relocated_scores, relocated_index, relocated_aux = None, None, None
            if stage != "semantic":
                cf_scores, cf_aux = score(
                    model, pairs[positive["sample_id"]], profiles, items, device, args, True
                )
                cf_compatibility = cf_aux["compatibility_scores"]
            if stage == "robust":
                targets = [0, len(positive["timeline"]) // 2, len(positive["timeline"]) - 1]
                relocated = _relocate_evidence(positive, rng.choice(targets))
                relocated_result = score(
                    model, relocated, profiles, items, device, args, args.energy_head
                )
                if args.energy_head:
                    relocated_scores, relocated_aux = relocated_result
                else:
                    relocated_scores, relocated_aux = relocated_result, None
                relocated_index = evidence_index(relocated)
            losses = v2_losses(
                scores, evidence_index(positive), aux, cf_scores,
                cf_compatibility, wrong_scores, relocated_scores, relocated_index,
                existence_score(cf_aux) if args.energy_head and cf_scores is not None else None,
                existence_score(wrong_aux) if args.energy_head else None,
                existence_score(relocated_aux)
                if args.energy_head and relocated_aux is not None else None,
            )
            distill_scale = (1.0 if stage == "semantic" else
                             args.counterfactual_distill_scale
                             if stage == "counterfactual" else
                             args.robust_distill_scale)
            weights = {
                "retrieval": 1.0,
                "distill": args.distill_weight * distill_scale,
                "personalization": args.personalization_weight,
                "router_entropy": args.sparse_weight,
                "pair": args.pair_weight if stage != "semantic" else 0.0,
                "replacement": args.replacement_weight if stage != "semantic" else 0.0,
                "shared_invariance": args.shared_weight if stage != "semantic" else 0.0,
                "necessity": args.necessity_weight if stage != "semantic" else 0.0,
                "sufficiency": args.sufficiency_weight if stage != "semantic" else 0.0,
                "relocation": args.relocation_weight if stage == "robust" else 0.0,
                "energy_content": (
                    args.energy_content_weight if stage != "semantic" else 0.0
                ),
                "energy_behavior": (
                    args.energy_behavior_weight if stage != "semantic" else 0.0
                ),
                "energy_relocation": (
                    args.energy_relocation_weight if stage == "robust" else 0.0
                ),
            }
            primary_loss = sum(
                weights[name] * value
                for name, value in losses.items()
                if name != "distill"
            )
            distill_weight = weights["distill"]
            loss = primary_loss + distill_weight * losses["distill"]
            optimizer.zero_grad()
            if args.project_conflicting_distill and distill_weight > 0:
                gradient_cosine, projected = _backward_with_projected_distillation(
                    model, primary_loss, losses["distill"], distill_weight
                )
                totals["distill_gradient_cosine"] += gradient_cosine
                totals["distill_projection_rate"] += float(projected)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for name, value in losses.items():
                totals[name] += float(value.detach())
        validation_result = validation_metrics(
            model, validation, pairs, profiles, items, device, args
        )
        record = {"epoch": epoch, "stage": stage, **validation_result,
                  "epoch_samples": len(ordered),
                  **{name: value / len(ordered) for name, value in totals.items()},
                  "clip_blend": float(
                      (0.9 + 0.1 * model.raw_blend_logit.sigmoid()).detach()
                  ) if model.raw_residual else 0.0,
                  "temporal_gate": float((0.02 * model.temporal_gate_logit.sigmoid()).detach())}
        history.append(record)
        epoch_states[epoch] = copy.deepcopy(model.state_dict())
        print(json.dumps(record))
        if validation_result["validation_composite"] > best_score:
            best_score, best_epoch = validation_result["validation_composite"], epoch
            best_state = copy.deepcopy(model.state_dict())
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    input_dim = len(next(iter(items.values())))
    stage_best = {}
    for stage in ("semantic", "counterfactual", "robust"):
        if not any(row["stage"] == stage for row in history):
            continue
        epoch = _best_stage_epoch(history, stage, args.selection_gap_threshold)
        stage_best[stage] = epoch
        model.load_state_dict(epoch_states[epoch])
        stage_output = _variant_path(args.output, stage)
        stage_checkpoint = _variant_path(args.checkpoint, stage)
        predict(model, rows, profiles, items, device, args, stage_output)
        torch.save(
            _checkpoint_payload(model, args, input_dim, history, epoch, stage),
            stage_checkpoint,
        )
    pareto_epochs = _pareto_epochs(history)
    model.load_state_dict(best_state)
    predict(model, rows, profiles, items, device, args, args.output)
    payload = _checkpoint_payload(
        model, args, input_dim, history, best_epoch, "composite"
    )
    payload["best_validation_composite"] = best_score
    payload["stage_best_epochs"] = stage_best
    payload["pareto_epochs"] = pareto_epochs
    torch.save(payload, args.checkpoint)
    args.output.with_suffix(".training.json").write_text(
        json.dumps({"history": history, "best_epoch": best_epoch,
                    "best_validation_composite": best_score,
                    "stage_best_epochs": stage_best,
                    "pareto_epochs": pareto_epochs}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
