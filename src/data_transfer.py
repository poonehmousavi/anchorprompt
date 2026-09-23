"""MMAU and MMAR loaders — the ZERO-SHOT TRANSFER targets.

Corrective prompts are learned on SAKURA and applied here unchanged. Nothing in this
module is ever used for learning; both benchmarks are held out entirely.

Why these two:
  MMAU  1000 items, letter-keyed choices already, questions close to SAKURA's framing
        ("identify the source of the speaking voice").
  MMAR  1000 items, free-text choice lists, and genuinely further from SAKURA
        ("Determine what is producing the sound"). It also ships modality / category /
        sub-category / language, so transfer can be broken down by audio type instead of
        reported as one pooled number. That makes it the stronger of the two targets.

THE CATEGORY PROBLEM. src/checker.py picks a corrective prompt by inferring a SAKURA
track (Animal/Emotion/Gender/Language) from the instruction. MMAU and MMAR have no such
tracks, so a per-category checker cannot transfer: infer_track would map every item onto
some SAKURA track and apply an unrelated prompt. It would not error — it would just
produce a meaningless number. Transfer therefore requires a SINGLE GLOBAL prompt, and
these loaders set `track="_global"` so that a per-track prompt library will KeyError
rather than silently mis-key.

Both are read-only benchmarks: no interventions are generated here. Clean transfer is
the first question; perturbing them is a separate exercise (noise/mask are cheap,
adversarial injection needs the TTS pipeline that produced the SAKURA injections).

Usage:
  python -m src.data_transfer --benchmark mmau --out data/transfer/mmau.jsonl
  python -m src.data_transfer --benchmark mmar --out data/transfer/mmar.jsonl --limit 20
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.data import Item, to_json
from src.interventions import permute_choices, render_choices, stable_item_seed
from src.tasks import gold_task_mmar, gold_task_mmau

import os

from src.paths import MMAR_ROOT, MMAU_AUDIO_ROOT

MMAU_MANIFEST = Path(os.environ.get("REPROMPT_MMAU_MANIFEST", "data/mmau/mmau_manifest.jsonl"))
MMAU_AUDIO = MMAU_AUDIO_ROOT
MMAR_META = MMAR_ROOT / "MMAR-meta.json"
MMAR_AUDIO = MMAR_ROOT / "audio"

# Items carry no SAKURA track. A per-track prompt library must fail loudly on this key
# rather than quietly resolve to Animal.
GLOBAL_TRACK = "_global"

_LETTERS = "abcdefghij"
_PAREN = re.compile(r"\(([a-jA-J])\)\s*([^()]+)")


def _choices_from_text(blob: str) -> dict[str, str]:
    """'(A) Man (B) Woman' -> {'a': 'Man', 'b': 'Woman'}."""
    return {m.group(1).lower(): m.group(2).strip() for m in _PAREN.finditer(blob)}


def _choices_from_list(items) -> dict[str, str]:
    """['Owl','Robot'] -> {'a': 'Owl', 'b': 'Robot'}. MMAR ships bare lists."""
    return {_LETTERS[i]: str(t).strip() for i, t in enumerate(items) if i < len(_LETTERS)}


def _gold_from_text(answer: str, choices: dict[str, str]) -> str | None:
    """Resolve the gold LETTER. MMAR answers are free text, not '(a) x'."""
    a = str(answer).strip()
    # A BARE letter first: MMAU's `true_letter` is just "B", with no parenthesis. A
    # pattern requiring "(B)" silently drops the whole benchmark to a text match that
    # cannot succeed, which is how this dropped 996/1000 items without erroring.
    if len(a) == 1 and a.lower() in choices:
        return a.lower()
    m = re.match(r"^\(?([a-jA-J])[\).\s]", a)
    if m and m.group(1).lower() in choices:
        return m.group(1).lower()
    norm = a.casefold().strip().rstrip(".")
    for L, t in choices.items():
        if t.casefold().strip().rstrip(".") == norm:
            return L
    return None


def _render(stem: str, choices: dict[str, str]) -> str:
    return f"{stem} " + " ".join(f"({L}) {t}" for L, t in choices.items())


def load_mmau(manifest: Path = MMAU_MANIFEST, audio_dir: Path = MMAU_AUDIO) -> list[Item]:
    out, dropped = [], 0
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        choices = _choices_from_text(str(r.get("choices", "")))
        gold = _gold_from_text(r.get("true_letter") or r.get("answer", ""), choices)
        wav = audio_dir / Path(str(r["audio_path"])).name
        if not (choices and gold and wav.exists()):
            dropped += 1
            continue
        stem = str(r["question"]).strip()
        out.append(Item(id=f"mmau/{r['id']}", track=GLOBAL_TRACK, hop="single",
                        audio_path=str(wav), instruction=_render(stem, choices),
                        stem=stem, choices_raw=choices, gold=gold,
                        attribute_label="", variant="clean"))
    if dropped:
        print(f"[mmau] dropped {dropped} items (missing audio / unparseable choices or gold)")
    return out


def load_mmar(meta: Path = MMAR_META, audio_dir: Path = MMAR_AUDIO) -> list[Item]:
    out, dropped = [], 0
    for r in json.loads(meta.read_text()):
        raw = r.get("choices")
        choices = _choices_from_list(raw) if isinstance(raw, list) else _choices_from_text(str(raw))
        gold = _gold_from_text(r.get("answer", ""), choices)
        wav = audio_dir / Path(str(r["audio_path"])).name
        if not (choices and gold and wav.exists()):
            dropped += 1
            continue
        stem = str(r["question"]).strip()
        it = Item(id=f"mmar/{r['id']}", track=GLOBAL_TRACK, hop="single",
                  audio_path=str(wav), instruction=_render(stem, choices),
                  stem=stem, choices_raw=choices, gold=gold,
                  attribute_label=str(r.get("category", "")), variant="clean")
        # Train-side metadata only — used for the per-modality breakdown at report time,
        # never serialized into a prompt.
        it.intervention = {"modality": r.get("modality"), "category": r.get("category"),
                           "sub_category": r.get("sub-category"), "language": r.get("language")}
        out.append(it)
    if dropped:
        print(f"[mmar] dropped {dropped} items (missing audio / unparseable choices or gold)")
    return out


LOADERS = {"mmau": load_mmau, "mmar": load_mmar}


def split_dev_test(items: list[Item], dev_frac: float = 0.4, seed: int = 0) -> dict:
    """Deterministic per-item dev/test assignment, keyed on the item id.

    Hash-based rather than positional so the split is stable across reruns and does not
    depend on manifest order. Dev is the smaller half: these benchmarks are the TEST
    targets, and only a slice is spent on learning.
    """
    return {it.id: ("dev" if stable_item_seed(seed, it.id) % 100 < dev_frac * 100
                    else "test") for it in items}


def emit_rows(items: list[Item], split: dict, seed: int = 0, permute: bool = True):
    """clean + permute for every item. Only the choice ORDER changes; the question and
    the choice TEXTS are data and stay verbatim (src/interventions.question_preserved).
    No audio interventions here — MMAU/MMAR have no variant audio."""
    for it in items:
        base = dict(to_json(it), split=split[it.id], unanswerable=False, variant="clean")
        yield base
        if not permute:
            continue
        sd = stable_item_seed(seed, f"{it.id}:permute")
        new_ch, new_gold = permute_choices(it.choices_raw, it.gold, sd)
        yield dict(base, variant="permute", choices_raw=new_ch, gold=new_gold,
                   instruction=f"{it.stem} {render_choices(new_ch)}",
                   intervention={"kind": "permute", "seed": sd})


def _gold_tasks(benchmark: str) -> dict:
    """Gold task labels for REPORTING slices. Never used to route a prompt."""
    if benchmark == "mmau":
        per = json.loads(Path(os.environ.get("REPROMPT_MMAU_GROUPS",
                                      "data/mmau/eval_mmau_omni.json")).read_text())["per_sample"]
        return {p["id"]: gold_task_mmau(p.get("group")) for p in per}
    meta = json.loads(MMAR_META.read_text())
    return {str(m["id"]): gold_task_mmar(m.get("modality"), m.get("sub-category"))
            for m in meta}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=sorted(LOADERS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dev-frac", type=float, default=0.4,
                    help="fraction held for LEARNING; the rest is the transfer test set")
    ap.add_argument("--no-permute", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    items = LOADERS[args.benchmark]()
    if args.limit:
        items = items[: args.limit]
    split = split_dev_test(items, args.dev_frac)
    gold = _gold_tasks(args.benchmark)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w") as fh:
        for d in emit_rows(items, split, permute=not args.no_permute):
            # gold task label: REPORTING ONLY. Never routed on, never in a prompt.
            d["gold_task"] = gold.get(d["id"].split("/", 1)[1], "sound")
            fh.write(json.dumps(d) + "\n"); n += 1
    import collections
    rows = [json.loads(l) for l in args.out.read_text().splitlines()]
    print(f"[{args.benchmark}] wrote {n} rows ({len(items)} items) -> {args.out}")
    print(f"[{args.benchmark}] split  : {dict(collections.Counter(r['split'] for r in rows))}")
    print(f"[{args.benchmark}] variant: {dict(collections.Counter(r['variant'] for r in rows))}")
    print(f"[{args.benchmark}] task   : {dict(collections.Counter(r['gold_task'] for r in rows))}")

if __name__ == "__main__":
    main()
