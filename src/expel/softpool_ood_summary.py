"""Build `results/ood_summary.csv` from the OOD eval dirs (no GPU).

    python -m src.expel.softpool_ood_summary [--out results/ood_summary.csv]

One row per model x benchmark x transform x arm: clean accuracy, perturbed accuracy,
consistency (same behaviour as the arm's own clean twin: same option text, or a decline on
both), decline rate, and the transition breakdown. Rows are scored from `eval_rows.jsonl`
with `consistency_report`, so the file always reflects the current rule.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.expel.run_softpool import TRANSITIONS, consistency_report

MODELS = (("qwen2.5-omni", ""), ("af3", "af3_"), ("qwen3-omni", "qwen3-omni_"))
DIRS = {"sakura": ["ood_permute_sakura_test", "ood_reverb_1.0s_sakura_test"],
        "mmau": ["ood_mmau_test"], "mmar": ["ood_mmar_test"]}
ARM_LABEL = {"no_pool": "none", "pool": "soft_L8_full"}


def summary_rows(root: Path = Path("output")) -> list[dict]:
    out = []
    for model, pfx in MODELS:
        for bench, dirs in DIRS.items():
            for tag in dirs:
                d = root / f"softpool_eval_{pfx}{tag}"
                if not (d / "softpool_report.json").exists():
                    continue
                for cond_dir in sorted(p for p in d.iterdir() if (p / "eval_rows.jsonl").exists()):
                    rows = [json.loads(l) for l in open(cond_dir / "eval_rows.jsonl")]
                    cons = consistency_report(rows)
                    for arm in ("no_pool", "pool"):
                        clean = [r for r in rows if r["arm"] == arm and r["condition"] == "clean"]
                        att = [r for r in rows if r["arm"] == arm and r["condition"] != "clean"]
                        if not att:
                            continue
                        c = cons[arm]
                        out.append({"model": model, "bench": bench, "condition": cond_dir.name,
                                    "prompt": ARM_LABEL[arm], "n": len(att),
                                    "clean_acc": round(100 * sum(r["correct"] for r in clean) / len(clean), 2),
                                    "perturbed_acc": round(100 * sum(r["correct"] for r in att) / len(att), 2),
                                    "consistency": c["consistency_pct"], "decline_pct": c["decline_pct"],
                                    **{t: c["transitions_pct"][t] for t in TRANSITIONS}})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/ood_summary.csv"))
    ap.add_argument("--root", type=Path, default=Path("output"))
    a = ap.parse_args(argv)
    rows = summary_rows(a.root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
