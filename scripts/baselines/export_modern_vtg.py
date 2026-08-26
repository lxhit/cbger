"""Export TR-DETR, FlashVTG, or MQVTG scores in canonical PBGER format."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from cbger.io import read_jsonl, write_jsonl


def load_model(kind: str, root: Path, checkpoint: Path, device):
    if kind == "tr_detr":
        repo, module = root / "third_party/tr_detr_official", "tr_detr.model"
    elif kind == "mqvtg":
        repo, module = root / "third_party/mqvtg_paper_repro", "tr_detr.model"
    else:
        repo, module = root / "third_party/flashvtg_official", "FlashVTG.model"
    sys.path.insert(0, str(repo))
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    args = checkpoint_data["opt"]
    args.device = device
    if kind == "flashvtg":
        args.mq_codebook_init = None
        build = __import__(module, fromlist=["build_model1"]).build_model1
    else:
        # Checkpoints should be self-contained and not depend on the prior file.
        if hasattr(args, "mq_codebook_init"):
            args.mq_codebook_init = None
        build = __import__(module, fromlist=["build_model"]).build_model
    model, _ = build(args)
    model.load_state_dict(checkpoint_data["model"])
    return model.to(device).eval()


def profile_map(profiles, mode, seed):
    users = sorted(profiles)
    if mode == "none":
        return dict(zip(users, users, strict=True))
    replacements = list(users)
    random.Random(seed).shuffle(replacements)
    if any(a == b for a, b in zip(users, replacements, strict=True)):
        replacements = replacements[1:] + replacements[:1]
    return dict(zip(users, replacements, strict=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=("tr_detr", "flashvtg", "mqvtg"), required=True)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--profiles", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile-mode", choices=("none", "random"), default="none")
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument("--split", default="test")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.kind, a.root, a.checkpoint, device)
    profiles = {str(x["user_id"]): x for x in read_jsonl(a.profiles)}
    features = {
        x["segment_id"]: torch.tensor(x["feature"], dtype=torch.float32)
        for x in read_jsonl(a.features)
    }
    mapping = profile_map(profiles, a.profile_mode, a.seed)
    records = []
    with torch.inference_mode():
        selected_rows = [x for x in read_jsonl(a.dataset) if not a.split or x.get("split") == a.split]
        for qid, sample in enumerate(selected_rows, 1):
            mapped = mapping[str(sample["user_id"])]
            query = torch.tensor(
                [x["feature"] for x in profiles[mapped]["slots"][:32]],
                dtype=torch.float32, device=device,
            )
            video = torch.stack([features[x["segment_id"]] for x in sample["timeline"]]).to(device)
            query, video = F.normalize(query, dim=-1), F.normalize(video, dim=-1)
            length = len(video)
            starts = torch.arange(length, device=device, dtype=video.dtype) / length
            tef = torch.stack((starts, starts + 1.0 / length), dim=-1)
            kwargs = dict(
                src_txt=query[None], src_txt_mask=torch.ones(1, len(query), dtype=torch.bool, device=device),
                src_vid=torch.cat((video, tef), dim=-1)[None],
                src_vid_mask=torch.ones(1, length, dtype=torch.bool, device=device),
            )
            if a.kind == "flashvtg":
                out = model(**kwargs, vid=[sample["sample_id"]], qid=[qid], targets={})
            else:
                out = model(**kwargs)
            scores = out["saliency_scores"][0, :length].float().cpu()
            candidates = [{**x, "score": float(scores[i])} for i, x in enumerate(sample["timeline"])]
            candidates.sort(key=lambda x: (-x["score"], x["segment_id"]))
            selected = candidates[0]["target_interval"]
            total = max(x["target_interval"][1] for x in sample["timeline"])
            records.append({
                "sample_id": sample["sample_id"], "split": sample["split"],
                "user_id": sample["user_id"], "relevance": sample["relevance"],
                "counterfactual_of": sample.get("counterfactual_of"),
                "ranked_candidates": candidates,
                "sample_score": float(torch.logsumexp(scores, 0)),
                "selected_duration_fraction": (selected[1] - selected[0]) / total,
                "representation": a.kind, "profile_mode": a.profile_mode,
            })
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(a.output, records)
    print(json.dumps({"kind": a.kind, "records": len(records), "output": str(a.output)}, indent=2))


if __name__ == "__main__":
    main()
