"""Re-score existing soft-pool eval dirs for CONSISTENCY with the clean answer (no GPU).

    python -m src.expel.softpool_consistency output/softpool_eval_joint_k67_mmau [more dirs]
    python -m src.expel.softpool_consistency --write output/softpool_eval_*          # rewrite reports

Per condition and arm: share of attacked items behaving exactly as that arm behaved on
the clean twin (same option, or a decline on both), the decline rate, and the transition
breakdown. `--write` replaces `conditions.<cond>.consistency` in `softpool_report.json`
and every `softpool_report_{val,test}.json` next to it (the split reports are re-scored on
their own slice via the `split` column of `per_item.csv` when present). Nothing else in
the reports is touched.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from src.expel.run_softpool import consistency_report
from src.expel.softpool import PAPER_LEVELS


def _rows(d: Path, cond: str) -> list[dict]:
    f = d / cond / "eval_rows.jsonl"
    return [json.loads(l) for l in open(f)] if f.exists() else []


def _split_of(d: Path) -> dict[str, str]:
    """task_id -> split, from per_item.csv (older runs carry no split column: empty)."""
    f = d / "per_item.csv"
    if not f.exists():
        return {}
    out = {}
    with open(f) as fh:
        for r in csv.DictReader(fh):
            if r.get("split"):
                out[r["task_id"]] = r["split"]
    return out


def rescore(d: Path, write: bool = False) -> dict:
    report_f = d / "softpool_report.json"
    report = json.load(open(report_f)) if report_f.exists() else {"conditions": {}}
    conds = [c for c in report.get("conditions", {})] or [p.name for p in d.iterdir() if (p / "eval_rows.jsonl").exists()]
    split_of = _split_of(d)
    split_reports = {s: json.load(open(f)) for s in ("val", "test")
                     for f in [d / f"softpool_report_{s}.json"] if f.exists()}
    out = {}
    for cond in conds:
        rows = _rows(d, cond)
        if not rows:
            continue
        out[cond] = consistency_report(rows)
        report.setdefault("conditions", {}).setdefault(cond, {})["consistency"] = out[cond]
        for s, rep in split_reports.items():
            sub = [r for r in rows if split_of.get(r["task_id"]) == s]
            if sub and cond in rep.get("conditions", {}):
                rep["conditions"][cond]["consistency"] = consistency_report(sub)
    if write:
        report["consistency_rule"] = "both_decline_consistent"
        json.dump(report, open(report_f, "w"), indent=2)
        for s, rep in split_reports.items():
            rep["consistency_rule"] = "both_decline_consistent"
            json.dump(rep, open(d / f"softpool_report_{s}.json", "w"), indent=2)
    return out


def main(argv: list[str] | None = None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])
    write = "--write" in args
    dirs = [Path(a) for a in args if a != "--write"]
    if not dirs:
        raise SystemExit(__doc__)
    for d in dirs:
        if not (d / "softpool_report.json").exists():
            continue
        print(f"== {d}")
        print(f"{'condition':12} {'baseline consistent':>20} {'baseline decline':>17} "
              f"{'pool consistent':>16} {'pool decline':>13}")
        res = rescore(d, write=write)
        for cond in (*PAPER_LEVELS, "adv_wrong", "text_inject", *[c for c in res if c not in PAPER_LEVELS and c not in ("adv_wrong", "text_inject")]):
            if cond not in res:
                continue
            b, p = res[cond].get("no_pool", {}), res[cond].get("pool", {})
            print(f"{cond:12} {b.get('consistency_pct')!s:>20} {b.get('decline_pct')!s:>17} "
                  f"{p.get('consistency_pct')!s:>16} {p.get('decline_pct')!s:>13}")


if __name__ == "__main__":
    main()
