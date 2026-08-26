"""Export official QD-DETR saliency scores in canonical PBGER prediction format."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from cbger.io import read_jsonl, write_jsonl


def load_features(paths: list[Path]) -> dict[str, torch.Tensor]:
    features = {}
    for path in paths:
        for row in read_jsonl(path):
            features[row["segment_id"]] = torch.tensor(
                row["feature"], dtype=torch.float32
            )
    return features


def profile_mapping(profiles: dict[str, dict], mode: str, seed: int) -> dict[str, str]:
    users = sorted(profiles)
    if mode == "none":
        return {user: user for user in users}
    replacements = list(users)
    if mode == "random":
        random.Random(seed).shuffle(replacements)
        if any(
            left == right
            for left, right in zip(users, replacements, strict=True)
        ):
            replacements = replacements[1:] + replacements[:1]
    elif mode == "shuffled":
        replacements = replacements[1:] + replacements[:1]
    else:
        raise ValueError(mode)
    return dict(zip(users, replacements, strict=True))


def prepare_sample(sample, profiles, features, mapping, device):
    mapped_user = mapping[str(sample["user_id"])]
    slots = profiles[mapped_user]["slots"][:32]
    query = torch.tensor(
        [slot["feature"] for slot in slots], dtype=torch.float32, device=device
    )
    video = torch.stack(
        [features[row["segment_id"]] for row in sample["timeline"]]
    ).to(device)
    query = F.normalize(query, dim=-1)
    video = F.normalize(video, dim=-1)
    length = len(video)
    starts = torch.arange(length, dtype=video.dtype, device=device) / length
    temporal_endpoint = torch.stack((starts, starts + 1.0 / length), dim=-1)
    return query, torch.cat((video, temporal_endpoint), dim=-1)


def predict_batch(model, rows, profiles, features, mapping, device):
    prepared = [
        prepare_sample(row, profiles, features, mapping, device) for row in rows
    ]
    queries = [item[0] for item in prepared]
    videos = [item[1] for item in prepared]
    query_lengths = torch.tensor([len(item) for item in queries], device=device)
    video_lengths = torch.tensor([len(item) for item in videos], device=device)
    query_batch = pad_sequence(queries, batch_first=True)
    video_batch = pad_sequence(videos, batch_first=True)
    query_mask = (
        torch.arange(query_batch.shape[1], device=device)[None, :]
        < query_lengths[:, None]
    )
    video_mask = (
        torch.arange(video_batch.shape[1], device=device)[None, :]
        < video_lengths[:, None]
    )
    outputs = model(
        src_txt=query_batch,
        src_txt_mask=query_mask,
        src_vid=video_batch,
        src_vid_mask=video_mask,
    )
    return [
        outputs["saliency_scores"][index, :length].detach().cpu()
        for index, length in enumerate(video_lengths.tolist())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--profile-mode", choices=("none", "random", "shuffled"), default="none")
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    official = args.root / "third_party/qd_detr_official_src"
    sys.path.insert(0, str(official))
    from qd_detr.model import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = checkpoint["opt"]
    model_args.device = device
    model, _ = build_model(model_args)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    rows = list(read_jsonl(args.dataset))
    profiles = {
        str(row["user_id"]): row for row in read_jsonl(args.profiles)
    }
    features = load_features(args.features)
    mapping = profile_mapping(profiles, args.profile_mode, args.seed)
    missing = sorted(
        {
            entry["segment_id"]
            for row in rows
            for entry in row["timeline"]
            if entry["segment_id"] not in features
        }
    )
    if missing:
        raise KeyError(f"Missing {len(missing)} segment features, first={missing[:5]}")

    records = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            batch_scores = predict_batch(
                model, batch_rows, profiles, features, mapping, device
            )
            for sample, scores in zip(batch_rows, batch_scores, strict=True):
                candidates = [
                    {**entry, "score": float(scores[index])}
                    for index, entry in enumerate(sample["timeline"])
                ]
                candidates.sort(key=lambda row: (-row["score"], row["segment_id"]))
                total = max(row["target_interval"][1] for row in sample["timeline"])
                selected = candidates[0]["target_interval"]
                records.append(
                    {
                        "sample_id": sample["sample_id"],
                        "split": sample["split"],
                        "user_id": sample["user_id"],
                        "relevance": sample["relevance"],
                        "counterfactual_of": sample.get("counterfactual_of"),
                        "ranked_candidates": candidates,
                        "sample_score": float(torch.logsumexp(scores, dim=0)),
                        "selected_duration_fraction": (
                            selected[1] - selected[0]
                        ) / total,
                        "representation": "official_qd_detr",
                        "profile_mode": args.profile_mode,
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, records)
    print(
        json.dumps(
            {
                "records": len(records),
                "profile_mode": args.profile_mode,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
