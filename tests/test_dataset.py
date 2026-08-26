import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def test_cbger10k_structure():
    data = list(rows(ROOT / "data/cbger10k/cbger10k.jsonl"))
    positives = {row["sample_id"] for row in data if row["relevance"] == 1}
    counterfactuals = [row for row in data if row["relevance"] == 0]
    assert len(data) == 10_000
    assert len(positives) == 5_000
    assert all(row["counterfactual_of"] in positives for row in counterfactuals)
    assert all(len(row["timeline"]) == 9 for row in data)
    assert all(
        sum(segment["role"] == "evidence" for segment in row["timeline"]) == 1
        for row in data
        if row["relevance"] == 1
    )

