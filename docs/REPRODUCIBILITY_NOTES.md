# Reproducibility notes

## Final naming map

| Paper/release | Historical project name |
|---|---|
| CBGER-10K | PBGER-v0.8 / `pbger_v0_8_lite` |
| CBGER | PBGER-Lite + Structured CF + mean pooling |
| Where | segment localization |
| Whether | mean-pooled factual/counterfactual ordering |
| Intervention | focal-slot score response |

Legacy sample IDs and checkpoint metadata are intentionally not rewritten because doing so would break compatibility and change frozen hashes.

## Final Whether implementation

The release evaluator defaults to parameter-free arithmetic mean pooling over the nine segment scores (`--aggregation mean`) to reproduce PairAcc `.6977±.0070`.

## Baseline scope

Frozen CLIP, PR-Net, QD-DETR, TR-DETR, FlashVTG and MQVTG summaries are included. Upstream code is not redistributed. MQVTG and QD-DETR adapter provenance should be completed before a public reproducibility claim.
