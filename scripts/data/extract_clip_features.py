"""Extract mean multi-frame CLIP features for segment manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from cbger.io import read_jsonl


def sample_frames(path: Path, start: float, end: float, count: int) -> list[Image.Image]:
    container = av.open(str(path))
    stream = container.streams.video[0]
    times = np.linspace(start, end, count + 2)[1:-1]
    frames: list[Image.Image] = []
    for timestamp in times:
        container.seek(int(timestamp / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if float(frame.pts * stream.time_base) + 1e-3 >= timestamp:
                frames.append(frame.to_image().convert("RGB"))
                break
    container.close()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--batch-segments", type=int, default=24)
    parser.add_argument("--max-segments", type=int)
    parser.add_argument(
        "--empty-frame-fallback",
        action="store_true",
        help="Retry empty segments using frames near the file boundaries.",
    )
    args = parser.parse_args()

    completed: set[str] = set()
    if args.output.exists():
        completed = {row["segment_id"] for row in read_jsonl(args.output)}
    segments = [
        row
        for path in args.segments
        for row in read_jsonl(path)
        if row["segment_id"] not in completed
    ]
    if args.max_segments:
        segments = segments[: args.max_segments]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(args.model, local_files_only=True)
    model = CLIPModel.from_pretrained(args.model, local_files_only=True).to(device).eval()

    with args.output.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(segments), args.batch_segments):
            batch = segments[offset : offset + args.batch_segments]
            images: list[Image.Image] = []
            owners: list[int] = []
            for index, segment in enumerate(batch):
                decoded = sample_frames(
                    Path(segment["path"]),
                    float(segment["start"]),
                    float(segment["end"]),
                    args.frames,
                )
                used_fallback = False
                if not decoded and args.empty_frame_fallback:
                    # Some source manifests slightly overrun the decoded stream.
                    # Seeking near zero and near the nominal end preserves a
                    # deterministic visual fallback instead of silently dropping
                    # the segment.
                    fallback_end = max(float(segment["end"]), 1.0)
                    decoded = sample_frames(
                        Path(segment["path"]),
                        0.0,
                        min(fallback_end, 1.0),
                        args.frames,
                    )
                    used_fallback = bool(decoded)
                segment["_feature_fallback"] = used_fallback
                images.extend(decoded)
                owners.extend([index] * len(decoded))
            if not images:
                print(
                    "skipped empty batch: "
                    + ",".join(segment["segment_id"] for segment in batch)
                )
                continue
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.inference_mode():
                image_features = model.get_image_features(pixel_values=pixel_values)
                if not isinstance(image_features, torch.Tensor):
                    image_features = image_features.pooler_output
                image_features = torch.nn.functional.normalize(image_features, dim=-1)
            for index, segment in enumerate(batch):
                positions = [position for position, owner in enumerate(owners) if owner == index]
                if not positions:
                    continue
                feature = image_features[positions].mean(dim=0)
                feature = torch.nn.functional.normalize(feature, dim=0).cpu().tolist()
                handle.write(
                    json.dumps(
                        {
                            "segment_id": segment["segment_id"],
                            "feature": feature,
                            "model": args.model.name,
                            "frames": len(positions),
                            "fallback": bool(segment.get("_feature_fallback", False)),
                        }
                    )
                    + "\n"
                )
            handle.flush()
            print(f"processed {min(offset + len(batch), len(segments))}/{len(segments)}")


if __name__ == "__main__":
    main()
