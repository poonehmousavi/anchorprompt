"""Evaluate a trained soft prompt pool (approach 11) on the SAKURA dev split.

Arms, both in the direct-answer format:
    no_pool   frozen LALM, no soft prompts            (the control)
    pool      soft prompts selected from the audio    (the method)
Clean and attacked are scored for every condition, so clean collateral is visible.
`selection.json` holds the confusion matrix of the top-slot condition against the true
one and the CLEAN FALSE-ALARM RATE -- that is the encoder-space router result and is
reported whether or not the pool helps.

Usage:
    python -m src.expel.run_softpool --pool output/softpool/pool.pt --limit 100 --out output/softpool_eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.expel.attacks import UNSEEN_CONDITIONS, get_attack
from src.expel.data import load_sakura, select_split, split_train_dev, take, wav_id

# Zero-shot transfer sets. Neither ships adversarial injection, so `adv_wrong` cannot be
# evaluated there (noise/masking render on the fly with SAKURA's recipe at EVERY level,
# including the two unanswerable ones; text_inject needs no files). Refused up front
# rather than crashing after the clean pass.
TRANSFER_CONDITIONS = ("noise_20dB", "noise_10dB", "noise_0dB", "noise_-10dB", "noise_-20dB",
                       "mask_20", "mask_40", "mask_60", "mask_80", "mask_100", "text_inject",
                       # answer-preserving channel transforms + choice permutation: render
                       # on the fly / need no files, so they run on every benchmark
                       *UNSEEN_CONDITIONS)

ARMS = ("no_pool", "pool")

# The paper's evaluation set (fixed 2026-09-10 so every prompt -- soft, hand-written, any
# length, any training size -- is scored on the same cells): three noise levels spanning
# answerable -> unanswerable, three mask levels likewise, text injection, and audio
# injection where it exists. `--conditions paper` expands to this.
PAPER_EVAL_CONDITIONS = ("noise_10dB", "noise_0dB", "noise_-20dB",
                         "mask_40", "mask_60", "mask_100", "text_inject")


def paper_conditions(benchmark: str) -> list[str]:
    conds = list(PAPER_EVAL_CONDITIONS)
    if benchmark == "sakura":
        conds.append("adv_wrong")
    return conds


def resume_condition(cond_dir: Path, n_tasks: int, arms=ARMS) -> tuple[dict, list[dict]] | None:
    """A finished condition folder (report + one row per task per arm on the attacked side)
    is reused instead of re-generated, so a requeued job (preemptible partitions) picks up
    where it stopped. Anything partial is redone."""
    rep, rows = cond_dir / "eval_report.json", cond_dir / "eval_rows.jsonl"
    if not (rep.exists() and rows.exists()):
        return None
    loaded = [json.loads(l) for l in open(rows) if l.strip()]
    attacked = [r for r in loaded if r.get("condition") != "clean"]
    for arm in arms:
        if sum(r.get("arm") == arm for r in attacked) != n_tasks:
            return None
    return json.loads(rep.read_text()), loaded


def seed_clean_cache(rows: list[dict], cache: dict) -> int:
    """Clean rows of a finished condition feed the in-memory clean cache, so a resumed run
    does not regenerate the clean arm either."""
    n = 0
    for r in rows:
        if r.get("condition") == "clean" and (r["task_id"], r["arm"]) not in cache:
            cache[(r["task_id"], r["arm"])] = dict(r)
            n += 1
    return n


PER_ITEM_COLUMNS = ("benchmark", "model", "split", "condition", "arm", "task_id", "recording_id",
                    "track", "hop", "gold", "pred", "outcome", "correct", "abstained",
                    "answerable", "lure_letter", "raw_full", "choices")


def write_per_item_csv(out_dir: Path, rows_by_cond: dict[str, list[dict]], benchmark: str,
                       split: str, model: str, name: str = "per_item.csv") -> Path:
    """One flat CSV per eval dir: every generation with its recording id, prediction, gold,
    condition and arm, so results can be analysed outside this repo. Clean rows are cached
    across conditions and repeated per folder; they are written once (condition "clean")."""
    import csv
    path = out_dir / name
    seen_clean: set = set()
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PER_ITEM_COLUMNS)
        w.writeheader()
        for cond, rows in rows_by_cond.items():
            for r in rows:
                condition = r.get("condition")
                if condition == "clean":
                    key = (r["task_id"], r["arm"])
                    if key in seen_clean:
                        continue
                    seen_clean.add(key)
                else:
                    condition = cond
                w.writerow({"benchmark": benchmark, "model": model, "split": split,
                            "condition": condition, "arm": r.get("arm"), "task_id": r.get("task_id"),
                            "recording_id": str(r.get("task_id", "")).split(":")[0],
                            "track": r.get("track"), "hop": r.get("hop"), "gold": r.get("gold"),
                            "pred": r.get("pred"), "outcome": r.get("outcome"),
                            "correct": r.get("correct"), "abstained": r.get("abstained"),
                            "answerable": r.get("answerable"), "lure_letter": r.get("lure_letter"),
                            "raw_full": r.get("raw_full", r.get("raw", "")),
                            "choices": json.dumps(r.get("choices")) if r.get("choices") else ""})
    return path


def _load(benchmark: str):
    if benchmark == "sakura":
        return load_sakura()
    from src.expel.benchmarks import load_benchmark
    return load_benchmark(benchmark)
from src.expel.evaluate import evaluate
from src.expel.softpool import (CONDITIONS, FULL_CONDITIONS, FORMATS, SoftPoolActor, SoftPoolHandle,
                                SoftPromptPool, SELECTION_MODES)

# Conditions that leave the AUDIO untouched: indistinguishable from clean by the selector.
AUDIO_IDENTICAL_TO_CLEAN = {"clean", "text_inject", "uic", "text_ec", "text_sc", "text_woo", "text_hw"}


def _answer_key(r: dict):
    """What the row chose, as the option TEXT when the row carries its choice set, else the
    letter. Under `permute` the letters legitimately move, so letter equality would score
    a perfectly consistent model as inconsistent; for every other condition the two keys
    are equivalent. A decline (`pred` None) is its own key, so two declines COMPARE EQUAL:
    declining on the clean twin and again on the attacked item is the same behaviour."""
    pred = r.get("pred")
    if pred is None:
        return ("abstain",)
    ch = r.get("choices")
    if ch and pred in ch:
        return ("text", ch[pred])
    return ("letter", pred)


TRANSITIONS = ("same_answer", "both_decline", "answer_to_decline", "decline_to_answer", "answer_changed")


def _transition(clean_key, att_key) -> str:
    c_dec, a_dec = clean_key == ("abstain",), att_key == ("abstain",)
    if c_dec and a_dec:
        return "both_decline"
    if c_dec:
        return "decline_to_answer"
    if a_dec:
        return "answer_to_decline"
    return "same_answer" if clean_key == att_key else "answer_changed"


def consistency_report(rows: list[dict], arms=ARMS) -> dict:
    """Per arm: share of attacked items whose BEHAVIOUR equals that ARM's behaviour on the
    clean twin (the training objective, measured on eval rows): the same option, or a
    decline on both. Any switch between answering and declining, in either direction, is a
    behaviour change and counts as inconsistent. `transitions_pct` breaks the whole set into
    same_answer / both_decline / answer_to_decline / decline_to_answer / answer_changed
    (`consistency_pct` = same_answer + both_decline). On the two unanswerable levels a LOW
    consistency is the goal (the clean answer is wrong there) and the decline rate is the
    number to read. Answers are compared as option TEXT when the rows carry `choices`
    (see `_answer_key`)."""
    out = {}
    for arm in arms:
        clean = {r["task_id"]: r for r in rows if r["arm"] == arm and r["condition"] == "clean"}
        att = [r for r in rows if r["arm"] == arm and r["condition"] != "clean" and r["task_id"] in clean]
        if not att:
            out[arm] = {"consistency_pct": None, "decline_pct": None, "n": 0}
            continue
        counts = {t: 0 for t in TRANSITIONS}
        for r in att:
            counts[_transition(_answer_key(clean[r["task_id"]]), _answer_key(r))] += 1
        same = counts["same_answer"] + counts["both_decline"]
        dec = sum(r.get("outcome") == "abstain" for r in att)
        out[arm] = {"consistency_pct": round(100.0 * same / len(att), 2),
                    "decline_pct": round(100.0 * dec / len(att), 2), "n": len(att),
                    "transitions_pct": {t: round(100.0 * v / len(att), 2) for t, v in counts.items()}}
    return out


def selection_report(rows_by_condition: dict[str, list[dict]], conditions) -> dict:
    """Confusion of predicted (top-slot) condition vs the true one, pool arm only.
    Clean rows appear under every condition run; they are counted ONCE (first seen)."""
    conf: dict[str, dict[str, int]] = {}
    seen_clean: set[str] = set()
    for cond, rows in rows_by_condition.items():
        for r in rows:
            if r["arm"] != "pool" or r.get("route") is None:
                continue
            if r["condition"] == "clean":
                if r["task_id"] in seen_clean:
                    continue
                seen_clean.add(r["task_id"])
                true = "clean"
            else:
                true = cond
            conf.setdefault(true, {}).setdefault(r["route"], 0)
            conf[true][r["route"]] += 1
    clean = conf.get("clean", {})
    n_clean = sum(clean.values())
    false_alarm = round(100.0 * (n_clean - clean.get("clean", 0)) / n_clean, 2) if n_clean else None
    # The query is audio-only, so a clean item and its text_inject twin are the SAME
    # input to the selector; routing a clean item to a text_inject slot is not an alarm
    # about the audio. This is the number to read for audio attacks.
    audio_alarm = sum(v for c, v in clean.items() if c not in AUDIO_IDENTICAL_TO_CLEAN)
    audio_false_alarm = round(100.0 * audio_alarm / n_clean, 2) if n_clean else None
    recall = {c: (round(100.0 * conf[c].get(c, 0) / sum(conf[c].values()), 2) if sum(conf[c].values()) else None)
              for c in conf}
    return {"confusion": conf, "clean_false_alarm_pct": false_alarm,
            "clean_audio_false_alarm_pct": audio_false_alarm, "recall_pct": recall,
            "conditions": list(conditions)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, required=False, help="pool.pt from train_softpool")
    ap.add_argument("--text-prompt", type=Path, default=None, metavar="FILE",
                    help="hand-written baseline: a text file whose content is inserted where the "
                         "soft block goes; no pool is loaded. Mutually exclusive with --pool")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="default: every paper noise/mask level + text_inject, and adv_wrong on "
                         "SAKURA. The single word `paper` = PAPER_EVAL_CONDITIONS (+ adv_wrong "
                         "on SAKURA), the fixed cell set every reported prompt is scored on. "
                         "`unseen` = UNSEEN_CONDITIONS (answer-preserving transforms never trained on)")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS),
                    help="default both; `no_pool` alone = the frozen model only (pilot: does a "
                         "transformation move the model at all before spending prompt evals on it)")
    ap.add_argument("--limit", type=int, default=15, help="dev items per track; 0 = the whole dev split")
    ap.add_argument("--split", default="dev", choices=("dev", "val", "test"),
                    help="dev = the whole 80%% held-out side; val = its 20%% model-selection slice "
                         "(scaling studies read ONLY this); test = the remaining 80%% (reported numbers)")
    # PER TRACK: sakura 4 tracks, mmau 3 (sound/music/speech), mmar 1. Match N, not the flag.
    ap.add_argument("--benchmark", default="sakura", choices=("sakura", "mmau", "mmar"),
                    help="zero-shot transfer: the pool is trained on SAKURA only")
    ap.add_argument("--train-frac", type=float, default=0.2)
    ap.add_argument("--top-k", type=int, default=None, help="override the trained top-k")
    ap.add_argument("--format", default=None, choices=sorted(FORMATS),
                    help="default: the format the pool was trained with")
    ap.add_argument("--selection", default="audio",
                    help="audio (the method) | random (placebo: k random slots per item) | "
                         "fixed:<condition> (placebo: that condition's slots for every item)")
    ap.add_argument("--model", default=None, help="qwen2.5-omni (default) | qwen3-omni | af3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("output/softpool_eval"))
    ap.add_argument("--dry-run", action="store_true", help="print the resolved config and exit")
    a = ap.parse_args(argv)
    if (a.pool is None) == (a.text_prompt is None):
        raise SystemExit("give exactly one of --pool POOL.PT or --text-prompt FILE")
    if a.conditions is None:
        a.conditions = [c for c in FULL_CONDITIONS if c != "clean"
                        and (a.benchmark == "sakura" or c in TRANSFER_CONDITIONS)]
    elif a.conditions == ["paper"]:
        a.conditions = paper_conditions(a.benchmark)
    elif a.conditions == ["unseen"]:
        a.conditions = list(UNSEEN_CONDITIONS)
    if a.benchmark != "sakura":
        bad = [c for c in a.conditions if c not in TRANSFER_CONDITIONS]
        if bad:
            raise SystemExit(f"{a.benchmark} has no renders for {bad}; transfer conditions are "
                             f"{list(TRANSFER_CONDITIONS)}")
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    print(json.dumps(cfg, indent=2), flush=True)
    if a.dry_run:
        return

    a.out.mkdir(parents=True, exist_ok=True)
    _, dev = split_train_dev(_load(a.benchmark), a.train_frac, a.seed)
    dev = select_split(dev, a.split, a.seed)
    tasks = take(dev, a.limit or None, a.seed)

    from src.model import env_report, load_model
    print(json.dumps(env_report(), indent=2), flush=True)
    model, processor = load_model(a.model)
    model.eval()
    if a.text_prompt is not None:
        # Hand-written baseline: no pool. The "pool" arm carries the text instruction.
        from src.expel.softpool import TextPromptHandle
        text = Path(a.text_prompt).read_text()
        pool = None
        top_k = 0
        fmt = a.format or "direct_abstain"
        handle = TextPromptHandle(text)
        pool_hyper = {"pool_type": "text_prompt", "text": handle.text, "file": str(a.text_prompt),
                      "conditions": list(FULL_CONDITIONS)}
        print(f"text prompt ({len(handle.text)} chars): {handle.text!r}", flush=True)
    else:
        pool = SoftPromptPool.load(a.pool).to(model.device)
        pool.eval()
        top_k = a.top_k or pool.top_k
        fmt = a.format or getattr(pool, "loaded_extra", {}).get("config", {}).get("format", "direct")
        handle = SoftPoolHandle(pool)
        pool_hyper = pool.hyper()
        print(f"pool: {pool.hyper()}  eval top_k={top_k}  extra={getattr(pool, 'loaded_extra', {})}", flush=True)
    from src.expel.softpool import NATIVE_FORMATS
    max_new = 512 if fmt == "af3_think" else (16 if fmt in NATIVE_FORMATS else (12 if fmt == "direct_abstain" else 8))
    actor = SoftPoolActor(model, processor, top_k=top_k or None, selection=a.selection, fmt=fmt,
                          max_new_tokens=max_new)
    print(f"format: {fmt}", flush=True)

    clean_cache: dict = {}
    summary, rows_by_cond = {}, {}
    for cond in a.conditions:
        print(f"== {cond}", flush=True)
        done = resume_condition(a.out / cond, len(tasks), arms=tuple(a.arms))
        if done is not None:
            rep, prior_rows = done
            print(f"   resumed from {a.out / cond} ({seed_clean_cache(prior_rows, clean_cache)} clean rows cached)", flush=True)
        else:
            rep = evaluate(actor, tasks, get_attack(cond), handle, seed=a.seed,
                           out_dir=a.out / cond, arms=tuple(a.arms), clean_cache=clean_cache)
        summary[cond] = {"accuracy": rep["accuracy"], "attacked": rep["attacked"],
                         "abstention_rate_unanswerable": rep["abstention_rate_unanswerable"],
                         "false_abstention_rate_answerable": rep["false_abstention_rate_answerable"],
                         "clean_collateral": rep["clean_collateral"],
                         "induced": rep["attack_induced_failures"],
                         "failures_repaired_pct": rep["failures_repaired_pct"],
                         "lure_follow_rate": rep["lure_follow_rate"], "by_track": rep["by_track"]}
        rows_by_cond[cond] = [json.loads(l) for l in open(a.out / cond / "eval_rows.jsonl")]
        summary[cond]["consistency"] = consistency_report(rows_by_cond[cond], arms=tuple(a.arms))
        print(json.dumps(summary[cond], indent=1), flush=True)
    sel = selection_report(rows_by_cond, pool_hyper["conditions"])
    if pool_hyper.get("pool_type") == "text_prompt":
        sel["note"] = "text prompt baseline: no selection; route is the constant 'text_prompt'"
    if pool_hyper.get("pool_type") == "free":
        sel["note"] = ("free pool: slot tags are POST-HOC (argmax of the training usage "
                       "histogram), so recall/false alarm read 'the slot this condition "
                       "used most fired', not a learned label")
    (a.out / "selection.json").write_text(json.dumps(sel, indent=2))
    write_per_item_csv(a.out, rows_by_cond, a.benchmark, a.split,
                       getattr(model, "_reprompt_name", "qwen2.5-omni"))
    (a.out / "softpool_report.json").write_text(json.dumps(
        {"config": cfg, "benchmark": a.benchmark, "model": getattr(model, "_reprompt_name", "qwen2.5-omni"),
         "pool": pool_hyper, "top_k": top_k,
         "n_items": len(tasks),
         "conditions": summary, "selection": sel, "format": fmt,
         "selection_mode": a.selection,
         "note": "frozen weights, label-free targets, NOT gradient-free"}, indent=2))
    print(json.dumps(sel, indent=2), flush=True)


if __name__ == "__main__":
    main()
