"""The detector for the routed arm (B2): which perturbation family is this, from audio alone?

The invariant arm (B1) applies one corrective prompt to every input. That cannot work,
and our own Phase 2 numbers say so: a single prompt has to be right for `mask_100`
(where declining is the correct answer) and `mask_20` (where declining is a wrong
answer) at the same time. It is not, and the result is net +71 on `noise_-20dB` against
net -15 on `mask_20`, with the letter-flip channel at +10 over ~3000 rows -- noise.
Every gain we have is the abstention channel firing indiscriminately.

Routing is what lets it fire only where declining is correct. Training may use the
perturbation label (CLAUDE.md permits intervention descriptions as a TRAINING SIGNAL);
inference may not. So the router is a supervised model over features that are legal at
inference -- the audio and the instruction, nothing else.

THE INVARIANT, ENFORCED STRUCTURALLY. `RouterInput` is frozen with exactly two fields,
and `predict` accepts only that. Handing it a `Task` raises TypeError rather than
quietly reading `task.attack_meta` and reporting oracle accuracy as router accuracy.
This mirrors `src/checker.py:CheckerInput`, for the same reason: a leak here does not
crash, it produces a plausible number that is worthless.

WHY THIS IS CHEAP. `src/expel/retrieval.py:audio_fingerprint` already computes what the
router needs -- 74 dims of log-mel statistics and spectral shape -- and its docstring
records the measurement that makes the task look tractable: at -17 dB SNR the cosine to
the clean source is still 0.985, but spectral flatness moves 0.012 -> 0.559. Silence has
near-zero band energy. The corruptions that matter are exactly the ones this feature
space separates. What was missing is supervision: retrieval uses these vectors
unsupervised (nearest insight), which is not the same as asking which family this is.

ROUTING TARGET. Coarse beats fine. `regime` (answerable vs not) is what actually changes
the prompt; `family` is the next step up; `condition` is 12-way and should be expected to
fit the signature of our own renders rather than anything transferable. All three are
selectable so the cost of granularity is measured rather than assumed.

Usage:
  python -m src.expel.router --target regime --limit 40 --out output/router_smoke
  python -m src.expel.router --target family --out output/router
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.expel.attacks import REGISTRY, UNANSWERABLE, get_attack
from src.expel.data import load_sakura, split_train_dev, take
from src.expel.retrieval import KeyEncoder, instruction_fingerprint
from src.expel.types import Task

# The conditions the routed arm has to tell apart. Deliberately the Phase 2 dev set plus
# clean: routing is only worth building for conditions we actually evaluate on.
DEFAULT_CONDITIONS = ("clean", "adv_wrong", "mask_20", "noise_20dB", "mask_100",
                      "noise_-20dB")

TARGETS = ("regime", "family", "condition")


@dataclass(frozen=True)
class RouterInput:
    """Everything the router is allowed to see at inference. Nothing else may be added.

    Exactly the deployable pair. Not built from `task.__dict__`, which would admit
    `condition` and `attack_meta` the moment Task changes.
    """
    audio_path: str
    instruction: str


def router_input_from_task(task: Task) -> RouterInput:
    return RouterInput(audio_path=task.audio_path,
                       instruction=f"{task.stem}\n{task.choices_block}")


def label_of(task: Task, target: str) -> str:
    """The supervision signal. TRAIN-TIME ONLY -- reads privileged fields on purpose."""
    if target == "condition":
        return task.condition
    if target == "regime":
        return "answerable" if task.answerable else "unanswerable"
    if target == "family":
        if task.condition == "clean":
            return "clean"
        return task.attack_meta.get("family", "unknown")
    raise ValueError(f"unknown target {target!r}; have {TARGETS}")


# --------------------------------------------------------------------------- features
class Featuriser:
    """RouterInput -> feature vector. Audio fingerprints are cached on disk.

    `audio` is the default and is the honest one for audio corruption: the instruction is
    byte-identical between a clean item and its noised twin, so text features cannot
    carry family information and can only memorise which wav this is. Text is worth
    adding only when the router must also catch text injection, where the reverse holds.

    The path is used to READ the waveform and never as a feature. That distinction is
    load-bearing: variant audio lives at `.../mask_100/dog28.wav`, so a single
    path-derived feature would let the router recover the label from the directory name
    and report near-perfect accuracy that measures nothing.
    `tests/test_router.py:test_features_ignore_the_path_string` pins it.
    """

    def __init__(self, mode: str = "audio", cache_path: str | Path | None = None):
        if mode not in ("audio", "audio_text"):
            raise ValueError(f"unknown feature mode {mode!r}")
        self.mode = mode
        self.encoder = KeyEncoder(cache_path=cache_path)

    def __call__(self, ri: RouterInput) -> np.ndarray:
        if not isinstance(ri, RouterInput):
            raise TypeError(
                f"the router featuriser takes RouterInput, got {type(ri).__name__}. "
                "Pass router_input_from_task(task) -- handing it a Task would let "
                "condition/attack_meta reach the inference path.")
        a = self.encoder.audio(ri.audio_path)
        if self.mode == "audio":
            return a
        return np.concatenate([a, instruction_fingerprint(ri.instruction)])

    def save(self) -> None:
        self.encoder.save()


# --------------------------------------------------------------------------- the router
class ConditionRouter:
    """Multinomial logistic regression over the fingerprint. Linear on purpose.

    A stronger model would fit our renders better and tell us less. The question is
    whether the families are linearly separable in a feature space that is cheap at
    inference and not tuned on the corruption recipe; if they are not, a deeper model
    buying separability on synthetic renders is exactly the result we must not believe.
    """

    def __init__(self, target: str = "regime", C: float = 1.0, seed: int = 0):
        if target not in TARGETS:
            raise ValueError(f"unknown target {target!r}; have {TARGETS}")
        self.target = target
        self.C, self.seed = C, seed
        self.model = None
        self.classes_: list[str] = []

    def fit(self, X: np.ndarray, y: list[str]) -> "ConditionRouter":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # class_weight balanced: the unanswerable regime is 2 conditions out of 6, and an
        # accuracy that comes from predicting the majority class is not routing.
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=self.C, class_weight="balanced",
                               random_state=self.seed))
        self.model.fit(np.asarray(X), list(y))
        self.classes_ = list(self.model.classes_)
        return self

    def predict(self, x) -> str | list[str]:
        """Accepts a RouterInput-derived feature vector or a matrix. Never a Task."""
        if isinstance(x, Task):
            raise TypeError(
                "ConditionRouter.predict does not take a Task -- that would put "
                "task.condition/attack_meta on the inference path. Featurise a "
                "RouterInput instead.")
        if self.model is None:
            raise RuntimeError("router is not fitted")
        X = np.asarray(x)
        single = X.ndim == 1
        out = list(self.model.predict(X.reshape(1, -1) if single else X))
        return out[0] if single else out

    def predict_proba(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("router is not fitted")
        X = np.asarray(X)
        return self.model.predict_proba(X.reshape(1, -1) if X.ndim == 1 else X)


# --------------------------------------------------------------------------- reporting
def confusion(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[list[int]]:
    idx = {L: i for i, L in enumerate(labels)}
    M = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        M[idx[t]][idx[p]] += 1
    return M


def per_class(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    out = {}
    for L in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == L and p == L)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != L and p == L)
        support = sum(1 for t in y_true if t == L)
        out[L] = {
            "support": support,
            "recall": round(100.0 * tp / support, 1) if support else None,
            "precision": round(100.0 * tp / (tp + fp), 1) if (tp + fp) else None,
        }
    return out


def regime_cost(y_true: list[str], y_pred: list[str], tasks: list[Task],
                target: str) -> dict:
    """The number B2 lives or dies on, whatever the routing target.

    Routing errors are not symmetric. Sending an unanswerable item to an answerable
    prompt loses an abstention we would have got right; sending an answerable item to
    the decline prompt manufactures a false abstention, which is the collateral that
    made B1 net-negative on the mild conditions. Both are reported, always.
    """
    def regime(label: str, task: Task) -> str:
        if target == "regime":
            return label
        if target == "condition":
            return "unanswerable" if label in UNANSWERABLE else "answerable"
        # family routing cannot express the regime at all: mask_20 and mask_100 share a
        # family. Say so rather than inventing a mapping.
        return ""

    if target == "family":
        return {"note": "family routing does not separate the regimes "
                        "(mask_20 and mask_100 are the same family); "
                        "use --target regime or condition for this number."}
    t = [regime(a, k) for a, k in zip(y_true, tasks)]
    p = [regime(a, k) for a, k in zip(y_pred, tasks)]
    miss = sum(1 for a, b in zip(t, p) if a == "unanswerable" and b == "answerable")
    false = sum(1 for a, b in zip(t, p) if a == "answerable" and b == "unanswerable")
    n_un = t.count("unanswerable")
    n_an = t.count("answerable")
    return {
        "unanswerable_missed": miss,
        "unanswerable_n": n_un,
        "unanswerable_missed_pct": round(100.0 * miss / n_un, 1) if n_un else None,
        "answerable_falsely_declined": false,
        "answerable_n": n_an,
        "answerable_falsely_declined_pct": round(100.0 * false / n_an, 1) if n_an else None,
    }


# --------------------------------------------------------------------------- dataset
def build_dataset(tasks: list[Task], conditions, featurise: Featuriser, target: str,
                  seed: int = 0):
    """(X, y, tasks) over every (task, condition) pair. Missing variants are reported."""
    X, y, kept, missing = [], [], [], Counter()
    for name in conditions:
        attack = get_attack(name)
        for t in tasks:
            try:
                at = attack(t, seed=seed)
            except Exception:
                missing[name] += 1
                continue
            if not Path(at.audio_path).exists():
                missing[name] += 1
                continue
            X.append(featurise(router_input_from_task(at)))
            y.append(label_of(at, target))
            kept.append(at)
    return np.asarray(X), y, kept, dict(missing)


def run(target="regime", conditions=DEFAULT_CONDITIONS, features="audio", limit=None,
        train_frac=0.2, seed=0, out: Path | None = None) -> dict:
    tasks = load_sakura()
    train_tasks, dev_tasks = split_train_dev(tasks, train_frac=train_frac, seed=seed)
    train_tasks, dev_tasks = take(train_tasks, limit, seed), take(dev_tasks, limit, seed)

    cache = (out / "fingerprints.npz") if out else None
    featurise = Featuriser(features, cache_path=cache)
    Xtr, ytr, _, miss_tr = build_dataset(train_tasks, conditions, featurise, target, seed)
    Xte, yte, tte, miss_te = build_dataset(dev_tasks, conditions, featurise, target, seed)
    featurise.save()

    if not len(Xtr) or not len(Xte):
        raise RuntimeError(f"empty dataset (train {len(Xtr)}, dev {len(Xte)}); "
                           f"missing variants: {miss_tr} / {miss_te}")

    router = ConditionRouter(target=target, seed=seed).fit(Xtr, ytr)
    pred = router.predict(Xte)
    labels = sorted(set(yte) | set(pred))
    acc = 100.0 * sum(1 for a, b in zip(yte, pred) if a == b) / len(yte)
    majority = Counter(ytr).most_common(1)[0][0]
    base = 100.0 * sum(1 for a in yte if a == majority) / len(yte)

    report = {
        "target": target,
        "features": features,
        "conditions": list(conditions),
        # WAV-level split: a dev wav was never seen in any condition during fit, so this
        # cannot be memorising the recording and reading its corruption off the identity.
        "n_train_rows": len(Xtr), "n_dev_rows": len(Xte),
        "n_train_wavs": len({t.id.split(':')[0] for t in train_tasks}),
        "n_dev_wavs": len({t.id.split(':')[0] for t in dev_tasks}),
        "missing_variants": {"train": miss_tr, "dev": miss_te},
        "accuracy": round(acc, 1),
        "majority_baseline": round(base, 1),
        "majority_class": majority,
        "labels": labels,
        "per_class": per_class(yte, pred, labels),
        "confusion": {"labels": labels, "matrix": confusion(yte, pred, labels)},
        "regime_cost": regime_cost(yte, pred, tte, target),
    }
    if out:
        out.mkdir(parents=True, exist_ok=True)
        # Feature mode is in the filename: an audio-only and an audio+text run of the
        # same target are different experiments and must not overwrite each other.
        (out / f"router_{target}_{features}.json").write_text(json.dumps(report, indent=2))
    return report


def _print(rep: dict) -> None:
    print(f"\n[router] target={rep['target']} features={rep['features']}")
    print(f"  train {rep['n_train_rows']} rows / {rep['n_train_wavs']} wavs"
          f"   dev {rep['n_dev_rows']} rows / {rep['n_dev_wavs']} wavs")
    if any(rep["missing_variants"].values()):
        print(f"  MISSING VARIANTS: {rep['missing_variants']}")
    print(f"  accuracy {rep['accuracy']}%   majority baseline "
          f"{rep['majority_baseline']}% ({rep['majority_class']})")
    print("  per class:")
    for L, d in rep["per_class"].items():
        print(f"    {L:<16} n={d['support']:<6} recall={d['recall']}  precision={d['precision']}")
    labels = rep["confusion"]["labels"]
    print("  confusion (rows = true):")
    print("    " + " " * 18 + "  ".join(f"{L[:10]:>10}" for L in labels))
    for L, row in zip(labels, rep["confusion"]["matrix"]):
        print(f"    {L:<18}" + "  ".join(f"{v:>10}" for v in row))
    print(f"  regime cost: {json.dumps(rep['regime_cost'])}\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="regime", choices=TARGETS)
    ap.add_argument("--features", default="audio", choices=("audio", "audio_text"))
    ap.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS),
                    choices=sorted(REGISTRY))
    ap.add_argument("--limit", type=int, default=None,
                    help="items per track per split; None = everything")
    ap.add_argument("--train-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    _print(run(target=a.target, conditions=tuple(a.conditions), features=a.features,
               limit=a.limit, train_frac=a.train_frac, seed=a.seed, out=a.out))


if __name__ == "__main__":
    main()
