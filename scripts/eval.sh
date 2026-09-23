#!/bin/bash
# Evaluate a trained prompt against the frozen model without it, on the test split.
#
#   bash scripts/eval.sh qwen2.5-omni checkpoints/qwen2.5-omni/pool.pt sakura
#   bash scripts/eval.sh qwen2.5-omni checkpoints/qwen2.5-omni/pool.pt mmau
#   LIMIT=2 bash scripts/eval.sh af3 checkpoints/af3/pool.pt mmar       # smoke test
#
# Conditions: clean + noise 10/0/-20 dB, masking 40/60/100 %, text injection
# (+ adversarial speech injection on SAKURA). CONDS="..." overrides, e.g. CONDS=unseen
# for the out-of-distribution transforms (reverb, choice permutation, ...).
# Writes <out>/softpool_report.json, per-condition eval_report.json, and per_item.csv.
set -euo pipefail
MODEL="${1:?usage: bash scripts/eval.sh <model> <pool.pt> <sakura|mmau|mmar>}"
POOL="${2:?pool.pt}"; BENCH="${3:?benchmark}"
FMT=()  # format is read from the checkpoint
python -m src.expel.run_softpool --model "$MODEL" --pool "$POOL" --benchmark "$BENCH" \
    --split "${SPLIT:-test}" --limit "${LIMIT:-0}" --conditions ${CONDS:-paper} "${FMT[@]}" \
    --seed 0 --out "${OUT:-output/eval_${MODEL}_${BENCH}}"
