#!/usr/bin/env python3
"""Resume-safe Qwen3-VL annotation for CBGER-10K construction."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from qwen3vl_utils import PROMPT, clip_video, parse_json, read_jsonl


def write_report(path: Path, *, total: int, completed: int, successes: int,
                 failures: int, started: float, peak_vram: float, fps: float,
                 precision: str, final: bool = False) -> None:
    elapsed = time.time() - started
    rate = completed / elapsed if elapsed else 0.0
    report = {
        "version": "PBGER-v0.7",
        "constructor": "Qwen/Qwen3-VL-8B-Instruct",
        "precision": "BF16" if precision == "bf16" else "NF4-4bit",
        "quantization": "none" if precision == "bf16" else "NF4-4bit",
        "fps": fps,
        "prompt_version": "qwen3vl_event_v1",
        "total_segments": total,
        "completed": completed,
        "successes": successes,
        "failures": failures,
        "success_rate": successes / completed if completed else None,
        "elapsed_hours": round(elapsed / 3600, 4),
        "segments_per_hour": round(rate * 3600, 2),
        "eta_hours": round((total - completed) / rate / 3600, 4) if rate else None,
        "peak_vram_gb": round(peak_vram, 3),
        "complete": final and completed >= total,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--precision", choices=("nf4", "bf16"), default="bf16")
    ap.add_argument("--max-duration", type=float, default=12.0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--retry", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    output = args.output or args.root / "data/interim/v0_7/qwen3_vl_segment_annotations_4fps_bf16.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_suffix(".report.json")

    by_id = {}
    for split in ("train", "validation", "test"):
        for row in read_jsonl(args.root / f"data/interim/v0_6/segments_{split}.jsonl"):
            if Path(row["path"]).exists():
                row = dict(row); row["split"] = split
                by_id.setdefault(row["segment_id"], row)
    rows = [by_id[key] for key in sorted(by_id)]
    if args.limit:
        rows = rows[:args.limit]

    done = {}
    if output.exists():
        for record in read_jsonl(output):
            if record.get("ok"):
                done[record["segment_id"]] = record
    pending = [row for row in rows if row["segment_id"] not in done]

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True) if args.precision == "nf4" else None
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, quantization_config=quant,
        device_map="auto" if args.precision == "nf4" else {"": 0},
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    torch.cuda.reset_peak_memory_stats()
    started = time.time(); successes = len(done); failures = 0

    with tempfile.TemporaryDirectory(prefix="pbger_v07_qwen3vl_") as td:
        td = Path(td)
        for index, row in enumerate(pending):
            result = None
            for attempt in range(args.retry + 1):
                tic = time.perf_counter(); raw = None
                record = {
                    "segment_id": row["segment_id"], "source_id": row["source_id"],
                    "split": row["split"], "source_interval": [row["start"], row["end"]],
                    "model": args.model,
                    "precision": "BF16" if args.precision == "bf16" else "NF4-4bit",
                    "quantization": "none" if args.precision == "bf16" else "NF4-4bit",
                    "prompt_version": "qwen3vl_event_v1", "fps": args.fps,
                    "attempt": attempt + 1, "ok": False,
                }
                try:
                    clip = td / "segment.mp4"
                    clip_video(row, clip, args.max_duration)
                    messages = [{"role": "user", "content": [
                        {"type": "video", "video": str(clip)},
                        {"type": "text", "text": PROMPT},
                    ]}]
                    text = processor.apply_chat_template(messages, tokenize=False,
                                                         add_generation_prompt=True)
                    inputs = processor(text=[text], videos=[str(clip)], padding=True,
                                       return_tensors="pt", do_sample_frames=True,
                                       fps=args.fps).to(model.device)
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                                   do_sample=False, use_cache=True)
                    trimmed = generated[:, inputs.input_ids.shape[1]:]
                    raw = processor.batch_decode(trimmed, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)[0]
                    record.update({"ok": True, "annotation": parse_json(raw),
                                   "input_tokens": int(inputs.input_ids.shape[1]),
                                   "output_tokens": int(trimmed.shape[1])})
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    if raw is not None:
                        record["raw"] = raw
                    torch.cuda.empty_cache()
                record["seconds"] = round(time.perf_counter() - tic, 3)
                result = record
                if record["ok"]:
                    break
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            if result["ok"]: successes += 1
            else: failures += 1
            completed = successes + failures
            if completed % 25 == 0 or index == len(pending) - 1:
                peak = torch.cuda.max_memory_allocated() / 2**30
                write_report(report_path, total=len(rows), completed=completed,
                             successes=successes, failures=failures, started=started,
                             peak_vram=peak, fps=args.fps, precision=args.precision)
                print(json.dumps({"completed": completed, "total": len(rows),
                                  "successes": successes, "failures": failures,
                                  "last_seconds": result["seconds"]}), flush=True)

    write_report(report_path, total=len(rows), completed=successes + failures,
                 successes=successes, failures=failures, started=started,
                 peak_vram=torch.cuda.max_memory_allocated() / 2**30,
                 fps=args.fps, precision=args.precision, final=True)


if __name__ == "__main__":
    main()
