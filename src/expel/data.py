"""SAKURA loading and the train/dev split for ExpeL.

Three roles, and they are not interchangeable:

  TRAIN  20% of SAKURA, further filtered to items the frozen model already answers
         correctly on CLEAN audio. Adversarial failure collection and pool learning.
  DEV    the remaining 80% of SAKURA. Every tuning decision -- retrieval, prompt
         format, robustness knobs -- is made here and nowhere else.
  TEST   MMAU and MMAR (src/expel/benchmarks.py). Zero-shot transfer, touched only
         after dev is settled. Nothing may be tuned on it.

Split is over WAVS, stratified by track, so both hop types of a wav land on the same
side and a question whose audio mined knowledge cannot reappear in dev.

The clean-correct prefilter matters more than it looks: an item the model already gets
wrong on clean audio cannot demonstrate that a perturbation broke anything, so mining it
teaches the pool about task difficulty rather than robustness. Measured on the first
runs, roughly 28% of training pairs fell in that bucket.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from src.expel.types import Task
from src.parsing import parse_options_raw, split_instruction

from src.paths import SAKURA_ROOT  # configurable; see src/paths.py
TRACKS = ("Animal", "Emotion", "Gender", "Language")
HOPS = ("single", "multi")


def _gold_letter(answer: str, choices: dict) -> str | None:
    a = answer.strip()
    if a.startswith("(") and len(a) > 2 and a[1].lower() in "abcd":
        return a[1].lower()
    norm = a.lower().strip()
    for L, t in choices.items():
        if t.lower().strip() == norm:
            return L
    return None


def load_track(track: str, root: Path = SAKURA_ROOT, hops=HOPS) -> list[Task]:
    meta = json.loads((root / "data" / track / "metadata.json").read_text())
    out: list[Task] = []
    for rel, d in meta.items():
        wav_id = f"{track}/{Path(rel).stem}"
        audio = str((root / rel).resolve())
        for hop in hops:
            instr = d.get(f"{hop}_instruction")
            if not instr:
                continue
            choices = parse_options_raw(instr)
            gold = _gold_letter(d.get(f"{hop}_answer", ""), choices)
            if not choices or gold is None:
                continue
            stem, _ = split_instruction(instr)
            out.append(Task(id=f"{wav_id}:{hop}", track=track, hop=hop, audio_path=audio,
                            stem=stem, choices=choices, gold=gold))
    return out


def load_sakura(tracks=TRACKS, root: Path = SAKURA_ROOT, hops=HOPS) -> list[Task]:
    return [t for tr in tracks for t in load_track(tr, root, hops)]


def wav_id(task: Task) -> str:
    return task.id.split(":")[0]


def split_train_dev(tasks: list[Task], train_frac: float = 0.2,
                    seed: int = 0) -> tuple[list[Task], list[Task]]:
    """Per-track wav-level split. Returns (train, dev)."""
    if not 0.10 <= train_frac <= 0.50:
        raise ValueError(f"train_frac must lie in [0.10, 0.50] for this phase, got {train_frac}")
    by_track: dict[str, list[str]] = {}
    for t in tasks:
        by_track.setdefault(t.track, [])
        if wav_id(t) not in by_track[t.track]:
            by_track[t.track].append(wav_id(t))
    rng = random.Random(seed)
    train_wavs: set[str] = set()
    for track, wavs in by_track.items():
        wavs = sorted(wavs)
        rng.shuffle(wavs)
        n = max(1, round(len(wavs) * train_frac))
        train_wavs.update(wavs[:n])
    train = [t for t in tasks if wav_id(t) in train_wavs]
    dev = [t for t in tasks if wav_id(t) not in train_wavs]
    return train, dev


VAL_FRAC = 0.2   # of the dev side (16% of the benchmark); by recording, per track, seed-fixed


def split_val_test(dev: list[Task], val_frac: float = VAL_FRAC, seed: int = 0) -> tuple[list[Task], list[Task]]:
    """Split the DEV side into a VALIDATION slice (model selection: prompt length, data
    size, RL vs gradient -- read only here) and a TEST slice (reported numbers). Per track,
    by recording (both SAKURA hops of a wav stay together), deterministic in `seed`.
    Returns (val, test)."""
    if not 0.05 <= val_frac <= 0.5:
        raise ValueError(f"val_frac must lie in [0.05, 0.5], got {val_frac}")
    by_track: dict[str, list[str]] = {}
    for t in dev:
        by_track.setdefault(t.track, [])
        if wav_id(t) not in by_track[t.track]:
            by_track[t.track].append(wav_id(t))
    rng = random.Random(seed + 101)
    val_wavs: set[str] = set()
    for track in sorted(by_track):
        wavs = sorted(by_track[track])
        rng.shuffle(wavs)
        n = max(1, round(len(wavs) * val_frac))
        val_wavs.update(wavs[:n])
    val = [t for t in dev if wav_id(t) in val_wavs]
    test = [t for t in dev if wav_id(t) not in val_wavs]
    return val, test


def select_split(dev: list[Task], split: str, seed: int = 0) -> list[Task]:
    """`dev` (everything), `val` or `test`."""
    if split == "dev":
        return dev
    val, test = split_val_test(dev, seed=seed)
    return val if split == "val" else test


# Kept so older call sites keep working; the second element is the DEV split.
split_train_test = split_train_dev


def take(tasks: list[Task], limit: int | None, seed: int = 0) -> list[Task]:
    """Balanced per-track subsample; `limit` is per track. None = everything."""
    if limit is None:
        return tasks
    rng = random.Random(seed + 7)
    by_track: dict[str, list[Task]] = {}
    for t in tasks:
        by_track.setdefault(t.track, []).append(t)
    out: list[Task] = []
    for track in sorted(by_track):
        group = sorted(by_track[track], key=lambda t: t.id)
        rng.shuffle(group)
        out.extend(group[:limit])
    return sorted(out, key=lambda t: t.id)
