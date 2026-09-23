#!/bin/bash
# Train one AnchorPrompt soft prompt (L = 8 vectors) for a frozen audio-language model.
#
#   bash scripts/train.sh qwen2.5-omni            # Qwen2.5-Omni-7B   (~4 h on one 48 GB GPU)
#   bash scripts/train.sh qwen3-omni              # Qwen3-Omni-30B-A3B (~10 h, 80 GB GPU)
#   bash scripts/train.sh af3                     # Audio Flamingo 3   (~4 h, 48 GB GPU)
#   LIMIT=2 EPOCHS=1 bash scripts/train.sh qwen2.5-omni   # smoke test, a few minutes
#
# Training data: SAKURA train split (200 recordings/track) + 67/track MMAU + 199 MMAR,
# each rendered clean and under 12 perturbations. Targets are the frozen model's own
# answers on the clean audio (no labels); CANNOT DETERMINE on the two evidence-free levels.
set -euo pipefail
MODEL="${1:?usage: bash scripts/train.sh <qwen2.5-omni|qwen3-omni|af3>}"
LIMIT="${LIMIT:-200}"; EPOCHS="${EPOCHS:-3}"
FEWSHOT=(--fewshot mmau:67 mmar:199)
[ "$LIMIT" -lt 200 ] && FEWSHOT=(--fewshot mmau:"$LIMIT" mmar:"$LIMIT")
FMT=(--format direct_abstain)
[ "$MODEL" = af3 ] && FMT=(--format af3_native --abstain-anyway)
python -m src.expel.train_softpool --model "$MODEL" "${FMT[@]}" "${FEWSHOT[@]}" \
    --pool-type single --prompt-len 8 --top-k 1 --lr 1e-3 \
    --train-limit "$LIMIT" --epochs "$EPOCHS" --seed 0 \
    --out "${OUT:-output/train_${MODEL}}"
