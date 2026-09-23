# AnchorPrompt

**Self-Distilled Soft Prompts for Robust Audio-Language Models**
Pooneh Mousavi, Amir Ivry, Mirco Ravanelli, Cem Subakan — submitted to ICASSP 2027

[Project page](https://poonehmousavi.github.io/anchorprompt/) · Paper (link to come)

AnchorPrompt keeps an audio-language model frozen and learns one block of **8 prompt
vectors**, inserted in the decoder input between the audio and the question embeddings.
The vectors are trained by self-distillation: on clean and perturbed inputs, the target
is the model's own answer on the clean recording, or `CANNOT DETERMINE` when the audio
carries no evidence (100% masking, −20 dB SNR). No ground-truth labels are used as
targets, and no perturbation detector is needed at inference.

The model weights are frozen, but training is **not gradient-free**: gradients flow
through the frozen model into the 8 vectors.

## Install

Python 3.10, one CUDA GPU (sm_75 or newer).

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

| model | `--model` | GPU memory |
|---|---|---|
| `Qwen/Qwen2.5-Omni-7B` | `qwen2.5-omni` | ~19 GB |
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | `qwen3-omni` | ~60 GB (80 GB card) |
| `nvidia/audio-flamingo-3-hf` | `af3` | ~17 GB |

## Data

Each root is read from an environment variable, with a default under `data/`.

| variable | default | contents |
|---|---|---|
| `REPROMPT_SAKURA_ROOT` | `data/sakura` | SAKURA: `data/<Track>/metadata.json` and wavs |
| `REPROMPT_MMAU_AUDIO` | `data/mmau_audio` | MMAU test-mini audio (metadata ships in `data/benchmarks/mmau_meta.json`) |
| `REPROMPT_MMAR_ROOT` | `data/mmar` | MMAR: `MMAR-meta.json` and `audio/` |

Noise and masking are rendered on the fly (cached in `data/benchmarks/_rendered`).
Adversarial audio injection reuses the SAKURA renders of Mousavi et al. (2026),
expected under `data/variants/audio_variants/`. Text injection is generated in code.

Recordings are split per track into 20% train, 16% validation and 64% test
(seed-fixed, by recording). All reported numbers are on the test split.

## Trained prompts

`checkpoints/<model>/pool.pt` holds the released prompt for each model, together with
`train_report.json` (the exact configuration) and `train_log.jsonl`.

## Evaluate

```bash
bash scripts/eval.sh qwen2.5-omni checkpoints/qwen2.5-omni/pool.pt sakura   # or mmau, mmar
LIMIT=2 bash scripts/eval.sh qwen2.5-omni checkpoints/qwen2.5-omni/pool.pt mmau   # quick check
CONDS=unseen bash scripts/eval.sh ...    # unseen perturbations (reverb, choice permutation, ...)
```

Each run scores the frozen model with and without the prompt and writes
`softpool_report.json`, a per-condition `eval_report.json` and `per_item.csv`.

## Train

```bash
bash scripts/train.sh qwen2.5-omni        # ~4 h on one 48 GB GPU
LIMIT=2 EPOCHS=1 bash scripts/train.sh qwen2.5-omni   # quick check
```

Training data: 200 SAKURA recordings per track (train split), 67 per track from MMAU and
199 from MMAR, each clean and under 12 perturbations (noise 20/10/0/−10/−20 dB, masking
20/40/60/80/100%, adversarial audio and text injection). The first pass records the
frozen model's clean answers (`teacher.jsonl`); these are the targets.

## Results

`results/summary_table.csv` (main and hallucination numbers) and
`results/ood_summary.csv` (unseen perturbations) hold every number in the paper.
`results/scaling/` holds the prompt-length and data-size studies (validation split).

## Tests

```bash
python -m pytest tests
```

## Citation

```bibtex
@inproceedings{mousavi2027anchorprompt,
  title     = {AnchorPrompt: Self-Distilled Soft Prompts for Robust Audio-Language Models},
  author    = {Mousavi, Pooneh and Ivry, Amir and Ravanelli, Mirco and Subakan, Cem},
  booktitle = {Submitted to ICASSP 2027},
  year      = {2027}
}
```
