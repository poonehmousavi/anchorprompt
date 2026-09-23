"""Bundle per-item results for sharing: one CSV + one summary per (model, prompt, benchmark, split).

Merges the `per_item.csv` of every eval dir that belongs to the same run family -- a whole
MMAU/MMAR job, or the eight SAKURA per-condition parts (`_sakura_p1.._p8_test`) -- into
    <out>/<model>__<prompt>__<benchmark>__<split>.csv
with the arm named as a PROMPT (`none` for the no-prompt arm; `v3`, `v4`, `soft_L8_full`, ...
for the other arm). Clean rows, which every SAKURA part recomputes, are kept once. A
`<same stem>.summary.md` holds accuracy / abstention per condition and per track, and
`index.md` lists every bundle.

    python -m src.expel.softpool_bundle --out output/results_test \
        --run qwen2.5-omni:v3:sakura:test=output/softpool_eval_text_v3_sakura_p*_test \
        --run qwen2.5-omni:v3:mmau:test=output/softpool_eval_text_v3_mmau_test
    python -m src.expel.softpool_bundle --out output/results_test --auto   # every *_test dir
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
from collections import defaultdict
from pathlib import Path

COLS = ["benchmark", "model", "split", "prompt", "condition", "task_id", "recording_id", "track",
        "hop", "gold", "pred", "outcome", "correct", "abstained", "answerable", "lure_letter",
        "raw_full", "choices"]

# eval-dir name -> (model, prompt, benchmark, split); SAKURA parts collapse onto one key
_DIR = re.compile(r"^softpool_eval_(?P<pre>(?:qwen3-omni_|af3_)?)(?P<kind>text_|single_)?(?P<tag>.+?)_"
                  r"(?P<bench>sakura|mmau|mmar)(?:_p\d+)?_(?P<split>val|test)$")


def classify(dir_name: str) -> tuple[str, str, str, str] | None:
    m = _DIR.match(dir_name)
    if not m:
        return None
    model = {"": "qwen2.5-omni", "qwen3-omni_": "qwen3-omni", "af3_": "af3"}[m["pre"]]
    tag = m["tag"]
    if "smoke" in tag:
        return None
    prompt = tag if m["kind"] == "text_" else f"soft_{tag}"
    return model, prompt, m["bench"], m["split"]


def bundle(dirs: list[Path], model: str, prompt: str, bench: str, split: str, out: Path) -> Path:
    seen_clean: set[tuple[str, str]] = set()
    rows: list[dict] = []
    for d in sorted(dirs):
        p = d / "per_item.csv"
        if not p.exists():
            continue
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["split"] != split:
                    continue
                r = dict(r)
                r["prompt"] = "none" if r.pop("arm") == "no_pool" else prompt
                if r["condition"] == "clean":
                    key = (r["task_id"], r["prompt"])
                    if key in seen_clean:
                        continue
                    seen_clean.add(key)
                rows.append(r)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{model}__{prompt}__{bench}__{split}"       # model names contain dots: no with_suffix
    csv_path, md_path = out / f"{stem}.csv", out / f"{stem}.summary.md"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    md_path.write_text(summary(rows, model, prompt, bench, split))
    return csv_path


def summary(rows: list[dict], model: str, prompt: str, bench: str, split: str) -> str:
    acc: dict = defaultdict(lambda: [0, 0, 0])   # (condition, prompt, track) -> [correct, abstain, n]
    for r in rows:
        for track in (r["track"], "ALL"):
            c = acc[(r["condition"], r["prompt"], track)]
            c[0] += r["correct"] == "True"; c[1] += r["abstained"] == "True"; c[2] += 1
    conds = sorted({k[0] for k in acc}, key=lambda c: (c != "clean", c))
    prompts = ["none", prompt]
    tracks = sorted({k[2] for k in acc} - {"ALL"}) + ["ALL"]
    lines = [f"# {model} / {prompt} / {bench} / {split}", "",
             "Per condition, per arm: accuracy (%) and abstention rate (%). `correct` on an "
             "unanswerable condition (mask_100, noise_-20dB) means abstained.", ""]
    for track in tracks:
        lines += [f"## {track}", "", "| condition | " + " | ".join(f"{p} acc | {p} abst" for p in prompts) + " | n |",
                  "|---|" + "---|" * (2 * len(prompts) + 1)]
        for c in conds:
            cells = []
            n = 0
            for p in prompts:
                k, a, n = acc.get((c, p, track), [0, 0, 0]), None, None
                n = k[2]
                cells += [f"{100 * k[0] / n:.1f}" if n else "—", f"{100 * k[1] / n:.1f}" if n else "—"]
            lines.append(f"| {c} | " + " | ".join(cells) + f" | {n} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--root", default="output", type=Path)
    ap.add_argument("--run", action="append", default=[], metavar="MODEL:PROMPT:BENCH:SPLIT=GLOB",
                    help="explicit family; GLOB may match several dirs (SAKURA parts)")
    ap.add_argument("--auto", action="store_true", help="bundle every softpool_eval_*_{val,test} dir under --root")
    ap.add_argument("--split", default=None, choices=("val", "test"), help="with --auto: only this split")
    a = ap.parse_args(argv)
    fams: dict[tuple, list[Path]] = defaultdict(list)
    for spec in a.run:
        key, pat = spec.split("=", 1)
        fams[tuple(key.split(":"))] += [Path(p) for p in glob.glob(pat)]
    if a.auto:
        for d in a.root.glob("softpool_eval_*"):
            k = classify(d.name)
            if k and (a.split is None or k[3] == a.split) and (d / "per_item.csv").exists():
                fams[k].append(d)
    written = []
    for (model, prompt, bench, split), dirs in sorted(fams.items()):
        written.append(bundle(dirs, model, prompt, bench, split, a.out))
        print(f"{model:12} {prompt:14} {bench:6} {split:4} <- {len(dirs)} dir(s) -> {written[-1].name}")
    (a.out / "index.md").write_text("# Per-item results\n\n" + "\n".join(
        f"- `{p.name}` (summary: `{p.name[:-4]}.summary.md`)" for p in written) + "\n")


if __name__ == "__main__":
    main()
