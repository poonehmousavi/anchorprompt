"""MMAU and MMAR — the Phase 3 transfer sets. Zero-shot; nothing here is tuned.

Both are loaded into the same `Task` the SAKURA path uses, so every downstream stage
(retrieval, evaluation, abstention scoring) works unchanged. Question and choice strings
are carried through verbatim: the constraint that applies to SAKURA applies here too.

Neither benchmark ships adversarial injection, which is why Phase 3 is noise and masking
only. MMAR ships no corrupted audio either, so its variants are rendered locally with
src/expel/render.py using the same recipe as SAKURA's — a different recipe would make
the transfer number measure the recipe.

Availability is checked and reported, never assumed. A benchmark whose audio is missing
raises with what is missing and where, rather than silently evaluating on a short prefix.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

from src.expel.types import Task

from src.paths import MMAR_ROOT, MMAU_AUDIO_ROOT  # configurable; see src/paths.py
# The full MMAU test-mini (1000 items, sound/music/speech), not the 281-item speech
# subset. Metadata is extracted once from the HF cache by scripts/prepare_mmau.py --
# offline, and with the audio column dropped, since decoding it needs torchcodec and the
# wavs already exist. listen-to-reason decoded them during its own run; that is the only
# clean MMAU audio on this cluster, so it is the source of truth rather than a
# convenience.
MMAU_META = Path("data/benchmarks/mmau_meta.json")
RENDER_CACHE = Path("data/benchmarks/_rendered")

LETTERS = "abcd"


def _as_choices(raw) -> dict:
    """MMAU stores {'A': ...}; MMAR stores a stringified list. Normalise to a..d."""
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {}
    if isinstance(raw, dict):
        return {k.lower(): str(v) for k, v in raw.items() if k.lower() in LETTERS}
    if isinstance(raw, (list, tuple)):
        return {LETTERS[i]: str(v) for i, v in enumerate(raw) if i < len(LETTERS)}
    return {}


def _gold(answer, choices: dict) -> str | None:
    a = str(answer).strip()
    if len(a) == 1 and a.lower() in choices:
        return a.lower()
    if a.startswith("(") and len(a) > 2 and a[1].lower() in choices:
        return a[1].lower()
    norm = " ".join(a.lower().split())
    for L, t in choices.items():
        if " ".join(str(t).lower().split()) == norm:
            return L
    return None


def load_mmar(root: Path = MMAR_ROOT, limit: int | None = None) -> list[Task]:
    meta_path = root / "MMAR-meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"MMAR metadata not found at {meta_path}")
    rows = json.loads(meta_path.read_text())
    out: list[Task] = []
    missing = 0
    for d in rows:
        choices = _as_choices(d.get("choices"))
        gold = _gold(d.get("answer"), choices)
        audio = (root / str(d["audio_path"]).lstrip("./")).resolve()
        if not choices or gold is None:
            continue
        if not audio.exists():
            missing += 1
            continue
        out.append(Task(id=f"mmar/{d['id']}", track="MMAR", hop="single",
                        audio_path=str(audio), stem=str(d["question"]).strip(),
                        choices=choices, gold=gold))
        if limit and len(out) >= limit:
            break
    if not out:
        raise FileNotFoundError(f"MMAR loaded 0 usable items ({missing} missing audio)")
    return out


def load_mmau(meta: Path = MMAU_META, audio_root: Path = MMAU_AUDIO_ROOT,
              limit: int | None = None) -> list[Task]:
    if not meta.exists():
        raise FileNotFoundError(
            f"MMAU metadata not found at {meta}. Run scripts/prepare_mmau.py "
            f"(offline, reads the HF cache).")
    rows = json.loads(meta.read_text())
    out, missing = [], 0
    for d in rows:
        choices = _as_choices(d.get("choices"))
        gold = d.get("gold") if d.get("gold") in choices else _gold(d.get("answer"), choices)
        rel = str(d.get("speech_path") or d.get("audio_path") or f"{d['id']}.wav")
        audio = Path(rel.replace("{data_root}/", "").replace("./", ""))
        audio = (audio_root / audio.name).resolve()
        if not choices or gold is None:
            continue
        if not audio.exists():
            missing += 1
            continue
        out.append(Task(id=f"mmau/{d['id']}", track=f"MMAU-{d.get('task', 'all')}",
                        hop="single",
                        audio_path=str(audio), stem=str(d["question"]).strip(),
                        choices=choices, gold=gold))
        if limit and len(out) >= limit:
            break
    if not out:
        raise FileNotFoundError(
            f"MMAU has metadata ({len(rows)} rows) but no usable audio: {missing} files "
            f"absent under {audio_root}. Without clean audio there is no baseline to "
            f"measure a transfer against.")
    return out


LOADERS = {"mmar": load_mmar, "mmau": load_mmau}


def load_benchmark(name: str, limit: int | None = None) -> list[Task]:
    if name not in LOADERS:
        raise KeyError(f"unknown benchmark {name!r}; have {sorted(LOADERS)}")
    return LOADERS[name](limit=limit)


def availability() -> dict:
    """What Phase 3 can actually run today. Prints rather than fails."""
    report = {}
    for name in LOADERS:
        try:
            tasks = load_benchmark(name, limit=5)
            report[name] = {"usable": True, "sample": len(tasks)}
        except Exception as e:
            report[name] = {"usable": False, "why": str(e).split("\n")[0][:200]}
    return report
