"""Context-preserving interventions, per arXiv:2509.22363 Sec. 3.

The governing constraint from that paper: interventions are "context-preserving" —
built to avoid the OOD artifacts audio editing tools introduce, and NONE of them may
change the correct answer. The model should still be able to listen to the right part.
src/verify_admissibility.py checks that mechanically rather than trusting it.

  permute      seeded derangement of choice order (no existing implementation anywhere
               in the prior repos; only a commented-out prepare_jumbled_choices)
  noise        Gaussian white noise over the whole signal, SNR in {20,10,0,-10,-20} dB
  mask         random distributed-chunk masking, ratio in {20,40,60,80,100}%
  adv_correct  TTS uttering the CORRECT answer, power <= source   (pre-generated)
  adv_wrong    TTS uttering a WRONG answer, power <= source       (pre-generated)

Published operating points from that paper: both models hold to 0 dB SNR and 60%
masking, then collapse. At mask 100% / SNR -20 dB the answer is genuinely
unrecoverable and the correct behaviour is abstention, not an answer.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import replace

import numpy as np

NOISE_SNRS = (20, 10, 0, -10, -20)
MASK_RATIOS = (20, 40, 60, 80, 100)
# Conditions where a faithful model should decline rather than answer.
UNANSWERABLE = {"noise_-20dB", "mask_100"}

# --- intervention families -------------------------------------------------------
# Phase 1 scope is AUDIO robustness via the corrective prompt. The instruction-side
# interventions (choice permutation, and the parked stem rephrasing) are a separate
# axis of the proposal and are deliberately NOT optimized against here: mixing them
# in would make a repair of audio damage indistinguishable from a repair of
# instruction sensitivity.
INSTRUCTION_VARIANTS = {"permute"}

# Variants that ALTER THE DATA and are therefore never legal targets, whatever a
# manifest happens to contain. data/phase0/items_v2.jsonl still holds 600 `rephrase`
# rows from before this rule; excluding them here means an old manifest cannot quietly
# reintroduce them.
ILLEGAL_VARIANTS = {"rephrase"}

# THE DATA IS IMMUTABLE. The question text and the choice TEXTS come from the benchmark
# and may never be altered — only the ORDER of the choices may change (permute). Anything
# we add around them (system prompt, corrective prompt, extra context) is ours to edit.
# `rephrase` rewrote the question stem and is therefore NOT a legal intervention; the
# helper stays for reference but is not emitted into any manifest.


def question_preserved(clean_stem: str, clean_choices: dict, stem: str,
                       choices: dict) -> str | None:
    """None if the data survived intact, else a description of what was altered.

    Checks the stem verbatim and the choice TEXTS as a multiset, so reordering passes
    and any edit, drop or addition fails. Enforced by tests over the built manifest,
    because a silently reworded question invalidates every accuracy number computed
    from it without raising anything.
    """
    if stem.strip() != clean_stem.strip():
        return f"stem altered: {clean_stem[:60]!r} -> {stem[:60]!r}"
    a = sorted(t.strip() for t in clean_choices.values())
    b = sorted(t.strip() for t in choices.values())
    if a != b:
        return f"choice texts altered: {a} -> {b}"
    return None


def adversarial_variants(all_variants) -> list[str]:
    """CURRENT PHASE 1 SCOPE: adversarial injection only.

    Narrower than audio_variants() on purpose. adv_wrong is the largest and cleanest
    repair target the baseline found — pooled accuracy falls 77.9 -> 36.8 (Animal) and
    72.4 -> 40.2 (Language) — and unlike noise/mask it is a BEHAVIOURAL failure: the
    model follows an injected spoken label instead of the audio it was asked about.
    That is the kind of failure a prompt can plausibly repair, whereas a degraded
    percept (heavy noise, heavy masking) may simply not be there to recover.

    adv_correct is NOT a repair target, but it is not benign either: see
    FAITHFULNESS_VARIANTS below. It is optimized under an inverted objective.
    """
    return sorted(v for v in all_variants if v == "adv_wrong")


# Injecting the CORRECT answer. Accuracy RISES, but a faithful model answers from the
# source audio, so a rise is susceptibility, not skill. Measured on the baseline:
# 82.4% / 95.3% / 68.5% / 83.0% (Animal/Emotion/Gender/Language) of clean-WRONG items
# flip to right once the answer is spoken aloud.
#
# These variants must NEVER be scored with the repair objective. There, `recovered`
# counts exactly those unfaithful flips as success and the optimizer is rewarded for
# making the model MORE injection-susceptible. They are scored with the sign inverted
# (src/metrics.py:compute_faithfulness, agent_optimize.score_candidate).
FAITHFULNESS_VARIANTS = {"adv_correct"}


def audio_variants(all_variants) -> list[str]:
    """Audio interventions only, excluding clean and the unanswerable regime.

    `adv_correct` is excluded as an OPTIMIZATION target: injecting the correct answer
    raises accuracy, so it is not a degradation to repair. It is still reported.
    """
    return sorted(v for v in all_variants
                  if v not in INSTRUCTION_VARIANTS
                  and v not in ILLEGAL_VARIANTS
                  and v not in UNANSWERABLE
                  and v not in ("clean", "adv_correct"))


def stable_item_seed(global_seed: int, item_id: str) -> int:
    """Hash-stable per-item seed (ported idea from faith_new/.../add_noise.py).

    Python's hash() is salted per process, so it cannot be used: the same item would
    get different noise across workers and across reruns.
    """
    h = hashlib.sha256(f"{global_seed}:{item_id}".encode()).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


# --------------------------------------------------------------------------- choice permutation
def permute_choices(choices: dict[str, str], gold: str, seed: int) -> tuple[dict[str, str], str]:
    """Derange the choice ORDER; the gold TEXT is unchanged, only its letter moves.

    Returns (new_choices, new_gold_letter). A derangement (no option keeps its slot)
    rather than a shuffle, so 'permuted' always actually permutes — a random shuffle
    returns the identity 1/n! of the time and would silently weaken the intervention.
    """
    letters = list(choices.keys())
    texts = [choices[L] for L in letters]
    n = len(letters)
    if n < 2:
        return dict(choices), gold
    rng = random.Random(seed)
    for _ in range(1000):
        perm = list(range(n))
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            break
    else:                                    # pragma: no cover - astronomically unlikely
        perm = list(range(1, n)) + [0]
    # slot i of the new layout receives the text that used to be at perm[i]
    new_choices = {letters[i]: texts[perm[i]] for i in range(n)}
    gold_idx = letters.index(gold)
    new_gold = letters[[i for i in range(n) if perm[i] == gold_idx][0]]
    return new_choices, new_gold


class NotParaphrasable(ValueError):
    """Raised when an item's stem pool is not a set of paraphrases."""


def rephrase_stem(stem: str, pool: list[str], seed: int, hop: str = "single") -> str:
    """Swap the question wording for another SAKURA phrasing of the same question.

    This is the proposal's instruction-rephrasing axis, and SAKURA supplies it directly:
    each track ships 8-10 human-authored phrasings of its single-hop question over the
    same audio with the same gold answer ("Can you select the animal from the provided
    options that matches the sound?" / "Identify the animal most likely responsible for
    the sound in this audio clip"). Using those beats generating paraphrases with an
    LLM: no meaning drift to police, and the wording is the benchmark's own.

    SINGLE-HOP ONLY, and this is a correctness constraint, not a convention. Multi-hop
    stems are NOT paraphrases of each other — Animal alone has 70, asking about Linnaean
    classification, locomotion, physical traits, and care requirements. They are
    different questions with different answers over the same audio, so swapping them
    changes the gold and silently breaks the context-preserving rule that no
    intervention may change the answer. Passing hop != "single" raises.
    """
    if hop != "single":
        raise NotParaphrasable(
            f"rephrase_stem got hop={hop!r}; only single-hop stems are paraphrases. "
            "Multi-hop stems ask different questions over the same audio, so swapping "
            "them would change the correct answer."
        )
    others = sorted({p for p in pool if p != stem})
    if not others:
        return stem                      # single-phrasing track: nothing to swap to
    return others[stable_item_seed(seed, stem) % len(others)]


def render_choices(choices: dict[str, str], upper: bool = False) -> str:
    """{'a': 'dog', 'b': 'cow'} -> '(a) dog (b) cow'.

    `upper` renders '(A) dog (B) cow', which is what the reference pipeline fed the
    model (faith_new/pooneh_version/sakura_prepare.py:45 uppercases the letters). Gold
    labels stay lowercase and parsing lowercases whatever it extracts, so this changes
    only what the model sees — which is the point of the replication arm.
    """
    return " ".join(f"({L.upper() if upper else L}) {t}" for L, t in choices.items())


# --------------------------------------------------------------------------- audio: noise
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)) + 1e-12)


def add_noise_at_snr(audio: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """Overlay Gaussian white noise on the WHOLE signal at the requested SNR.

    Power-domain, matching faith_new/.../add_noise.py:
        noise_power = signal_power / 10**(snr_db/10)
    """
    rng = np.random.default_rng(seed)
    sig_pow = float(np.mean(np.square(audio, dtype=np.float64))) + 1e-12
    noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_pow), size=audio.shape).astype(audio.dtype)
    return safe_clip(audio + noise)


def measure_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Realized SNR in dB, gain-invariant, for the admissibility check.

    add_noise_at_snr may apply a global gain via safe_clip when the sum peaks above
    1.0. A global gain PRESERVES the true SNR (signal and noise scale together), but a
    naive `noisy - clean` residual does not: it leaves (k-1)*clean mixed into the
    "noise" term and reports a badly wrong value (-20 dB requested measured as -8 dB).
    So project `noisy` onto `clean` first and treat the orthogonal remainder as noise.
    """
    n = min(len(clean), len(noisy))
    c, y = clean[:n].astype(np.float64), noisy[:n].astype(np.float64)
    denom = float(np.dot(c, c)) + 1e-12
    alpha = float(np.dot(y, c)) / denom            # best-fit gain applied to the signal
    resid = y - alpha * c
    sp = float(np.mean(np.square(alpha * c))) + 1e-12
    npow = float(np.mean(np.square(resid))) + 1e-12
    return 10.0 * np.log10(sp / npow)


# --------------------------------------------------------------------------- audio: masking
def block_mask(audio: np.ndarray, ratio_pct: float, seed: int, n_blocks: int = 10) -> np.ndarray:
    """Zero out `ratio_pct`% of the signal in DISTRIBUTED chunks.

    Distributed rather than contiguous, per the paper: spreading the masks over the
    whole duration stops the model leaning on one localized 'shortcut' segment and
    forces it to attend across the clip.
    """
    out = audio.copy()
    n = len(audio)
    if ratio_pct <= 0 or n == 0:
        return out
    if ratio_pct >= 100:
        return np.zeros_like(out)
    rng = random.Random(seed)
    block_len = max(1, n // n_blocks)
    n_mask_blocks = int(round(n_blocks * ratio_pct / 100.0))
    idx = list(range(n_blocks))
    rng.shuffle(idx)
    for b in idx[:n_mask_blocks]:
        s = b * block_len
        e = n if b == n_blocks - 1 else min(n, s + block_len)
        out[s:e] = 0.0
    return out


def measure_mask_ratio(audio: np.ndarray) -> float:
    """Fraction of samples that are exactly zero, as a percentage."""
    if len(audio) == 0:
        return 0.0
    return 100.0 * float(np.count_nonzero(audio == 0.0)) / len(audio)


# --------------------------------------------------------------------------- shared
def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim > 1 else x


def safe_clip(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return (x / peak * 0.99).astype(np.float32) if peak > 1.0 else x.astype(np.float32)


def variant_name(kind: str, level=None) -> str:
    if kind == "noise":
        return f"noise_{level}dB"
    if kind == "mask":
        return f"mask_{level}"
    return kind


def measure_mask_ratio_relative(clean: np.ndarray, masked: np.ndarray) -> float:
    """Percentage of ORIGINALLY NON-SILENT samples that the mask zeroed.

    measure_mask_ratio() counts every zero sample, so a clip that already contains
    natural silence (leading/trailing padding is common in SAKURA) measures far above
    the applied ratio — a 20% mask on a clip that is 30% silent reads as ~44%. That is
    a property of the source, not a mislabelled intervention.
    """
    n = min(len(clean), len(masked))
    c, m = clean[:n], masked[:n]
    nonzero_src = c != 0.0
    denom = int(np.count_nonzero(nonzero_src))
    if denom == 0:
        return 0.0
    newly_zeroed = int(np.count_nonzero(nonzero_src & (m == 0.0)))
    return 100.0 * newly_zeroed / denom
