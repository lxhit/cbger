#!/usr/bin/env python3
"""Gate Qwen3-VL-8B for PBGER-v0.7 automatic segment annotation."""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import tempfile
import time
from pathlib import Path

import torch
import imageio_ffmpeg
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration


PROMPT = """Analyze this short video segment for a personalized video evidence benchmark.
Return JSON only with exactly these keys:
objects (list of short strings), actions (list of short strings),
event_sequence (list of objects with order and event), outcome (short string),
topics (list of short strings), uncertainty (number from 0 to 1).
Describe only visible evidence. Do not infer user preference.
Keep the complete JSON compact and under 120 words."""


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object")
    obj = json.loads(text[start : end + 1])
    required = {"objects", "actions", "event_sequence", "outcome", "topics", "uncertainty"}
    if set(obj) != required:
        raise ValueError(f"schema keys differ: {sorted(obj)}")
    if not all(isinstance(obj[x], list) for x in ("objects", "actions", "event_sequence", "topics")):
        raise ValueError("list field has wrong type")
    uncertainty = float(obj["uncertainty"])
    if not 0 <= uncertainty <= 1:
        raise ValueError("uncertainty outside [0,1]")
    return obj


def clip_video(row: dict, output: Path, duration: float) -> None:
    start = float(row["start"])
    available = max(0.25, float(row["end"]) - start)
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start),
         "-i", row["path"], "-t", str(min(duration, available)), "-an",
         "-vf", "scale=448:-2", "-c:v", "libx264", "-preset", "veryfast", str(output)],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--max-pixels", type=int, default=320 * 320)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--precision", choices=("nf4", "bf16"), default="nf4")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    output = args.output or args.root / f"outputs/v07/qwen3_vl_gate_{args.count}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in ("train", "validation", "test"):
        for row in read_jsonl(args.root / f"data/interim/v0_6/segments_{split}.jsonl"):
            if Path(row["path"]).exists():
                row["split"] = split
                rows.append(row)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    # Prefer distinct sources so decoding failures are not hidden by repeated videos.
    selected, seen = [], set()
    for row in rows:
        if row["source_id"] not in seen:
            selected.append(row); seen.add(row["source_id"])
        if len(selected) == args.count:
            break

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True) if args.precision == "nf4" else None
    load_start = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, quantization_config=quant,
        device_map="auto" if args.precision == "nf4" else {"": 0},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    load_seconds = time.perf_counter() - load_start
    torch.cuda.reset_peak_memory_stats()

    completed = []
    with tempfile.TemporaryDirectory(prefix="qwen3_vl_gate_") as td:
        td = Path(td)
        for idx, row in enumerate(selected):
            started = time.perf_counter()
            raw = None
            record = {"index": idx, "segment_id": row["segment_id"], "source_id": row["source_id"],
                      "split": row["split"], "ok": False}
            try:
                clip = td / f"{idx:04d}.mp4"
                clip_video(row, clip, args.duration)
                messages = [{"role": "user", "content": [
                    {"type": "video", "video": str(clip), "fps": args.fps,
                     "max_pixels": args.max_pixels},
                    {"type": "text", "text": PROMPT},
                ]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                # Transformers 5.x natively decodes paths and returns VideoMetadata.
                # Avoid pre-sampling with qwen-vl-utils, whose tuple interface targets
                # older Transformers releases.
                inputs = processor(
                    text=[text], videos=[str(clip)], padding=True, return_tensors="pt",
                    do_sample_frames=True, fps=args.fps,
                ).to(model.device)
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                               do_sample=False, use_cache=True)
                trimmed = generated[:, inputs.input_ids.shape[1]:]
                raw = processor.batch_decode(trimmed, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)[0]
                record.update({"ok": True, "annotation": parse_json(raw), "raw": raw,
                               "input_tokens": int(inputs.input_ids.shape[1]),
                               "output_tokens": int(trimmed.shape[1])})
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                if raw is not None:
                    record["raw"] = raw
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            record["seconds"] = round(time.perf_counter() - started, 3)
            completed.append(record)
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps({k: record[k] for k in ("index", "segment_id", "ok", "seconds")}), flush=True)

    successful = [r for r in completed if r["ok"]]
    latencies = sorted(r["seconds"] for r in successful)
    summary = {
        "model": args.model,
        "precision": "NF4-4bit" if args.precision == "nf4" else "BF16",
        "quantization": "NF4-4bit" if args.precision == "nf4" else "none",
        "count": len(completed),
        "successes": len(successful), "success_rate": len(successful) / max(1, len(completed)),
        "json_valid_rate": len(successful) / max(1, len(completed)),
        "load_seconds": round(load_seconds, 3),
        "mean_seconds": round(sum(latencies) / max(1, len(latencies)), 3),
        "median_seconds": latencies[len(latencies)//2] if latencies else None,
        "p95_seconds": latencies[min(len(latencies)-1, int(len(latencies)*.95))] if latencies else None,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "gate": {"no_oom": not any("out of memory" in r.get("error", "").lower() for r in completed),
                 "success_ge_98pct": len(successful) / max(1, len(completed)) >= .98,
                 "json_ge_95pct": len(successful) / max(1, len(completed)) >= .95,
                 "peak_vram_le_23gb": torch.cuda.max_memory_allocated() / 2**30 <= 23},
    }
    summary["qualified"] = all(summary["gate"].values())
    report = output.with_suffix(".report.json")
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
