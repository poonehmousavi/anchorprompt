"""Re-score a finished `--split dev` eval per VAL / TEST slice, on CPU.

A dev run scores every held-out item, so the validation slice (model selection) and the
test slice (reported numbers) can both be read from it after the fact. Writes
`softpool_report_val.json` and `softpool_report_test.json` (same `conditions` layout as
`softpool_report.json`, plus `n_items`, `split`) and `per_item.csv` with a `split` column.

    python -m src.expel.softpool_split_report output/softpool_eval_single_L4_full_mmau --benchmark mmau
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def split_ids(benchmark: str, seed: int = 0, train_frac: float = 0.2) -> dict[str, set[str]]:
    from src.expel.data import split_train_dev, split_val_test
    from src.expel.run_softpool import _load
    _, dev = split_train_dev(_load(benchmark), train_frac, seed)
    val, test = split_val_test(dev, seed=seed)
    return {"val": {t.id for t in val}, "test": {t.id for t in test}}


def report_for(rows_by_cond: dict[str, list[dict]], keep: set[str]) -> dict:
    from src.expel.evaluate import summarize
    from src.expel.run_softpool import consistency_report
    out = {}
    for cond, rows in rows_by_cond.items():
        sub = [r for r in rows if r["task_id"] in keep]
        if not sub:
            continue
        rep = summarize(sub)
        out[cond] = {"accuracy": rep["accuracy"], "attacked": rep["attacked"],
                     "abstention_rate_unanswerable": rep["abstention_rate_unanswerable"],
                     "false_abstention_rate_answerable": rep["false_abstention_rate_answerable"],
                     "clean_collateral": rep["clean_collateral"], "induced": rep["attack_induced_failures"],
                     "failures_repaired_pct": rep["failures_repaired_pct"],
                     "lure_follow_rate": rep["lure_follow_rate"], "by_track": rep["by_track"],
                     "consistency": consistency_report(sub)}
    return out


def split_dir(run_dir: Path, benchmark: str, seed: int = 0, ids: dict[str, set[str]] | None = None) -> dict:
    from src.expel.run_softpool import write_per_item_csv
    ids = ids or split_ids(benchmark, seed)
    rows_by_cond = {}
    for cond_dir in sorted(d for d in run_dir.iterdir() if d.is_dir() and (d / "eval_rows.jsonl").exists()):
        rows_by_cond[cond_dir.name] = [json.loads(l) for l in open(cond_dir / "eval_rows.jsonl") if l.strip()]
    base = json.loads((run_dir / "softpool_report.json").read_text()) if (run_dir / "softpool_report.json").exists() else {}
    summary = {}
    for split in ("val", "test"):
        conds = report_for(rows_by_cond, ids[split])
        n = len({r["task_id"] for rows in rows_by_cond.values() for r in rows if r["task_id"] in ids[split]})
        rep = {k: v for k, v in base.items() if k not in ("conditions", "selection")}
        rep.update({"split": split, "n_items": n, "conditions": conds,
                    "note": "re-scored from a --split dev run; val = model selection, test = reported"})
        (run_dir / f"softpool_report_{split}.json").write_text(json.dumps(rep, indent=2))
        summary[split] = {"n_items": n, "conditions": sorted(conds)}
    # per-item CSV with the split of every row
    model = base.get("model", "qwen2.5-omni")
    tagged = {c: [dict(r, split=("val" if r["task_id"] in ids["val"] else "test")) for r in rows]
              for c, rows in rows_by_cond.items()}
    _write_csv_with_split(run_dir, tagged, benchmark, model)
    return summary


def _write_csv_with_split(run_dir: Path, rows_by_cond: dict, benchmark: str, model: str) -> None:
    from src.expel.run_softpool import PER_ITEM_COLUMNS
    import csv
    seen_clean: set = set()
    with open(run_dir / "per_item.csv", "w", newline="") as fh:
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
                w.writerow({"benchmark": benchmark, "model": model, "split": r["split"],
                            "condition": condition, "arm": r.get("arm"), "task_id": r.get("task_id"),
                            "recording_id": str(r.get("task_id", "")).split(":")[0],
                            "track": r.get("track"), "hop": r.get("hop"), "gold": r.get("gold"),
                            "pred": r.get("pred"), "outcome": r.get("outcome"),
                            "correct": r.get("correct"), "abstained": r.get("abstained"),
                            "answerable": r.get("answerable"), "lure_letter": r.get("lure_letter"),
                            "raw_full": r.get("raw_full", r.get("raw", "")),
                            "choices": json.dumps(r.get("choices")) if r.get("choices") else ""})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", type=Path, nargs="+")
    ap.add_argument("--benchmark", required=True, choices=("sakura", "mmau", "mmar"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    ids = split_ids(a.benchmark, a.seed)
    for d in a.run_dirs:
        print(json.dumps({"run_dir": str(d), **split_dir(d, a.benchmark, a.seed, ids)}))


if __name__ == "__main__":
    main()
