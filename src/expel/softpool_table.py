"""Per-category table for a soft-pool eval run.

Reads `<run_dir>/<condition>/eval_rows.jsonl` and prints, for every track x condition,
the attacked accuracy without and with the pool, net = repaired - damaged (paired per
item), and, on unanswerable conditions, the abstention rate. Clean accuracy per track is
printed once. The pooled number hides which categories survive; this does not.

    python -m src.expel.softpool_table output/softpool_eval_levels_mmau_k3
    python -m src.expel.softpool_table <dir> --json   # machine-readable
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_rows(run_dir: Path) -> list[dict]:
    """Rows label attacked items `condition="attacked"`; the level is the folder name.
    Clean rows are cached and repeated in every folder, so they are kept once."""
    rows: list[dict] = []
    seen_clean: set[tuple] = set()
    for f in sorted(run_dir.glob("*/eval_rows.jsonl")):
        level = f.parent.name
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("condition") == "clean":
                    k = (r["task_id"], r["arm"])
                    if k in seen_clean:
                        continue
                    seen_clean.add(k)
                else:
                    r["condition"] = level
                rows.append(r)
    return rows


def _score(r: dict) -> bool:
    """Correct, or a decline on an unanswerable item (the faithful answer there)."""
    if not r.get("answerable", True):
        return bool(r.get("abstained"))
    return bool(r.get("correct"))


def per_category(rows: list[dict]) -> dict:
    """{track: {condition: {n, none, pool, repaired, damaged, net, abstain_none, abstain_pool}}}
    Accuracies in percent. Items are paired by (task_id, condition) across arms."""
    by_key: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        by_key[(r["track"], r["condition"], r["task_id"])][r["arm"]] = r
    out: dict = defaultdict(dict)
    agg: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "none": 0, "pool": 0, "rep": 0, "dmg": 0,
                                                  "ab_none": 0, "ab_pool": 0, "unans": False})
    for (track, cond, _), arms in by_key.items():
        if "no_pool" not in arms or "pool" not in arms:
            continue
        a, b = arms["no_pool"], arms["pool"]
        s = agg[(track, cond)]
        s["n"] += 1
        sa, sb = _score(a), _score(b)
        s["none"] += sa
        s["pool"] += sb
        s["rep"] += (not sa) and sb
        s["dmg"] += sa and (not sb)
        s["ab_none"] += bool(a.get("abstained"))
        s["ab_pool"] += bool(b.get("abstained"))
        s["unans"] = s["unans"] or (not a.get("answerable", True))
    for (track, cond), s in agg.items():
        n = s["n"]
        out[track][cond] = {
            "n": n,
            "none": round(100.0 * s["none"] / n, 1),
            "pool": round(100.0 * s["pool"] / n, 1),
            "repaired": s["rep"], "damaged": s["dmg"], "net": s["rep"] - s["dmg"],
            "abstain_none": round(100.0 * s["ab_none"] / n, 1),
            "abstain_pool": round(100.0 * s["ab_pool"] / n, 1),
            "unanswerable": s["unans"],
        }
    return dict(out)


def _order(conds: list[str]) -> list[str]:
    def key(c: str):
        if c == "clean":
            return (0, 0)
        if c.startswith("noise_"):
            return (1, -int(c[len("noise_"):-2]))
        if c.startswith("mask_"):
            return (2, int(c[len("mask_"):]))
        return (3, c)
    return sorted(conds, key=key)


def format_table(table: dict) -> str:
    lines = []
    for track in sorted(table):
        conds = _order(list(table[track]))
        lines.append(f"== {track}")
        lines.append(f"  {'condition':<12} {'n':>4} {'none':>6} {'pool':>6} {'net':>5} {'rep/dmg':>8} {'abstain none->pool':>20}")
        for c in conds:
            s = table[track][c]
            ab = f"{s['abstain_none']:.0f} -> {s['abstain_pool']:.0f}" + (" *" if s["unanswerable"] else "")
            lines.append(f"  {c:<12} {s['n']:>4} {s['none']:>6.1f} {s['pool']:>6.1f} {s['net']:>+5d} "
                         f"{s['repaired']:>3}/{s['damaged']:<4} {ab:>20}")
    lines.append("  * unanswerable: 'pool'/'none' count a decline as correct; abstention is the target there")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rows = _load_rows(a.run_dir)
    if not rows:
        raise SystemExit(f"no eval_rows.jsonl under {a.run_dir}")
    table = per_category(rows)
    if a.json:
        print(json.dumps(table, indent=2))
    else:
        print(format_table(table))


if __name__ == "__main__":
    main()
