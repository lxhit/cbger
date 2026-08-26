"""Build a flat, deterministic segment manifest from PBGER timelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cbger.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    segments: dict[str, dict] = {}
    conflicts: list[str] = []
    for dataset in args.dataset:
        for row in read_jsonl(dataset):
            for candidate in row["timeline"]:
                segment_id = str(candidate["segment_id"])
                source_id = str(candidate["source_id"])
                start, end = map(float, candidate["source_interval"])
                record = {
                    "segment_id": segment_id,
                    "source_id": source_id,
                    "path": str(args.video_dir / f"{source_id}.mp4"),
                    "start": start,
                    "end": end,
                }
                previous = segments.get(segment_id)
                if previous is not None and previous != record:
                    conflicts.append(segment_id)
                else:
                    segments[segment_id] = record

    if conflicts:
        raise ValueError(f"Conflicting metadata for {len(set(conflicts))} segment IDs")
    missing = [row["segment_id"] for row in segments.values() if not Path(row["path"]).is_file()]
    if missing and not args.allow_missing:
        raise FileNotFoundError(
            f"Missing source videos for {len(missing)} segments; first IDs: {missing[:5]}"
        )
    records = [segments[key] for key in sorted(segments)]
    write_jsonl(args.output, records)
    report = {
        "datasets": [str(path) for path in args.dataset],
        "video_dir": str(args.video_dir),
        "segments": len(records),
        "sources": len({row["source_id"] for row in records}),
        "missing_segments": len(missing),
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
