"""SAKURA loading and the Phase 0 item sample.

SAKURA lives at listen-to-reason/dataset/SAKURA (clean audio, intact: 500 wav x 4
tracks). Each track's metadata.json is keyed by relative wav path and carries BOTH a
single-hop and a multi-hop question over the same audio:

    "data/Animal/audio/dog28.wav": {
      "attribute_label": "dog",
      "single_instruction": "...which animal...? (a) dog (b) crow (c) cow (d) rooster",
      "single_answer": "(a) dog",
      "multi_instruction": "What physical feature does the animal have? (a) Feathers ...",
      "multi_answer": "(c) Claws" }

That pairing is why Phase 0 uses SAKURA: prompts are induced on single-hop and tested
on multi-hop, so the headline result asks whether repairing perception transfers to a
task that depends on it.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from src.parsing import parse_options, parse_options_raw, split_instruction

from src.paths import SAKURA_ROOT  # configurable; see src/paths.py
TRACKS = ("Animal", "Emotion", "Gender", "Language")


@dataclass
class Item:
    id: str                     # "Animal/dog28"
    track: str                  # "Animal"
    hop: str                    # "single" | "multi"
    audio_path: str             # absolute path to the (possibly intervened) wav
    instruction: str            # full question incl. inline "(a) .. (b) .."
    stem: str                   # question text without the choices
    choices_raw: dict           # {"a": "dog", ...} original casing
    gold: str                   # "a"
    attribute_label: str
    variant: str = "clean"      # clean | permute | adv_correct | adv_wrong | noise_XdB | mask_XX
    # Train-side only. NEVER serialized into a prompt; see src/checker.py and tests/test_no_leak.py.
    intervention: dict = field(default_factory=dict)


def _gold_letter(answer: str, choices: dict[str, str]) -> str | None:
    """SAKURA answers look like '(a) dog'. Fall back to matching on the text."""
    a = answer.strip()
    if a.startswith("(") and len(a) > 2 and a[1].lower() in "abcd":
        return a[1].lower()
    norm = a.lower().strip()
    for L, t in choices.items():
        if t.lower().strip() == norm:
            return L
    return None


def load_track(track: str, root: Path = SAKURA_ROOT) -> list[Item]:
    """All items for one track, both hop types (2 Items per wav)."""
    meta_path = root / "data" / track / "metadata.json"
    meta = json.loads(meta_path.read_text())
    items: list[Item] = []
    for rel, d in meta.items():
        audio = (root / rel).resolve()
        stem_id = f"{track}/{Path(rel).stem}"
        for hop in ("single", "multi"):
            instr = d[f"{hop}_instruction"]
            raw_choices = parse_options_raw(instr)
            gold = _gold_letter(d[f"{hop}_answer"], raw_choices)
            if gold is None or not raw_choices:
                continue                                    # unparseable metadata row; skip loudly below
            q_stem, _ = split_instruction(instr)
            items.append(Item(
                id=stem_id, track=track, hop=hop, audio_path=str(audio),
                instruction=instr, stem=q_stem, choices_raw=raw_choices, gold=gold,
                attribute_label=d.get("attribute_label", ""),
            ))
    return items


def load_sakura(tracks=TRACKS, root: Path = SAKURA_ROOT) -> list[Item]:
    out: list[Item] = []
    for t in tracks:
        out.extend(load_track(t, root))
    return out


def sample_items(items: list[Item], per_track: int, seed: int = 0) -> list[Item]:
    """Balanced per-track sample. Sampling is over WAVS, so both hop types of a chosen
    wav are kept together — the single->multi split requires the same audio on both sides."""
    by_track: dict[str, list[str]] = {}
    for it in items:
        by_track.setdefault(it.track, [])
        if it.id not in by_track[it.track]:
            by_track[it.track].append(it.id)
    rng = random.Random(seed)
    keep: set[str] = set()
    for track, ids in by_track.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        keep.update(ids[:per_track])
    return [it for it in items if it.id in keep]


def split_dev_test(items: list[Item], seed: int = 0) -> tuple[set[str], set[str]]:
    """Split by WAV id, stratified by track, so no item appears on both sides."""
    by_track: dict[str, list[str]] = {}
    for it in items:
        by_track.setdefault(it.track, [])
        if it.id not in by_track[it.track]:
            by_track[it.track].append(it.id)
    rng = random.Random(seed + 1)
    dev: set[str] = set()
    test: set[str] = set()
    for track, ids in by_track.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        half = len(ids) // 2
        dev.update(ids[:half])
        test.update(ids[half:])
    return dev, test


def to_json(it: Item) -> dict:
    d = it.__dict__.copy()
    return d


def from_json(d: dict) -> Item:
    return Item(**d)
