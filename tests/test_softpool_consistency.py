"""Consistency with the clean answer is scored per ARM against that arm's own clean row."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.expel.run_softpool import consistency_report  # noqa: E402


def _row(tid, arm, cond, pred, outcome="letter"):
    return {"task_id": tid, "arm": arm, "condition": cond, "pred": pred, "outcome": outcome}


def test_consistency_is_per_arm_and_counts_declines():
    rows = [
        _row("t1", "no_pool", "clean", "a"), _row("t1", "no_pool", "noise_0dB", "a"),
        _row("t2", "no_pool", "clean", "b"), _row("t2", "no_pool", "noise_0dB", None, "abstain"),
        _row("t1", "pool", "clean", "c"), _row("t1", "pool", "noise_0dB", "c"),
        _row("t2", "pool", "clean", "b"), _row("t2", "pool", "noise_0dB", "a"),
    ]
    c = consistency_report(rows)
    assert (c["no_pool"]["consistency_pct"], c["no_pool"]["decline_pct"], c["no_pool"]["n"]) == (50.0, 50.0, 2)
    assert c["no_pool"]["transitions_pct"] == {"same_answer": 50.0, "both_decline": 0.0, "answer_to_decline": 50.0,
                                               "decline_to_answer": 0.0, "answer_changed": 0.0}
    # the pool arm's clean answer on t1 is "c", so "c" under noise is consistent for the pool
    assert (c["pool"]["consistency_pct"], c["pool"]["decline_pct"], c["pool"]["n"]) == (50.0, 0.0, 2)
    assert c["pool"]["transitions_pct"]["answer_changed"] == 50.0


def test_declining_on_both_twins_is_the_same_behaviour():
    """A decline on the clean twin AND on the attacked item is consistent; a switch in
    either direction is not. Without this the no-prompt arm, which declines on 10-16 % of
    clean items, is capped below 100 % before the attack does anything."""
    rows = [
        _row("t1", "no_pool", "clean", None, "abstain"), _row("t1", "no_pool", "reverb", None, "abstain"),
        _row("t2", "no_pool", "clean", None, "abstain"), _row("t2", "no_pool", "reverb", "a"),
        _row("t3", "no_pool", "clean", "a"), _row("t3", "no_pool", "reverb", None, "abstain"),
        _row("t4", "no_pool", "clean", "a"), _row("t4", "no_pool", "reverb", "a"),
    ]
    c = consistency_report(rows)["no_pool"]
    assert c["consistency_pct"] == 50.0 and c["decline_pct"] == 50.0
    assert c["transitions_pct"] == {"same_answer": 25.0, "both_decline": 25.0, "answer_to_decline": 25.0,
                                    "decline_to_answer": 25.0, "answer_changed": 0.0}


def test_missing_arm_is_reported_not_crashed():
    c = consistency_report([_row("t1", "no_pool", "clean", "a")])
    assert c["pool"]["n"] == 0 and c["no_pool"]["n"] == 0


def test_consistency_compares_option_text_when_choices_are_present():
    """Under `permute` the letters move; the same TEXT must count as consistent and a
    letter-equal but text-different answer must not."""
    orig = {"a": "dog", "b": "cat"}
    perm = {"a": "cat", "b": "dog"}
    rows = [
        {**_row("t1", "no_pool", "clean", "a"), "choices": orig},
        {**_row("t1", "no_pool", "permute", "b"), "choices": perm},     # dog again: consistent
        {**_row("t2", "no_pool", "clean", "a"), "choices": orig},
        {**_row("t2", "no_pool", "permute", "a"), "choices": perm},     # now cat: not
    ]
    c = consistency_report(rows)
    assert (c["no_pool"]["consistency_pct"], c["no_pool"]["decline_pct"], c["no_pool"]["n"]) == (50.0, 0.0, 2)


def test_consistency_honours_the_arms_argument():
    c = consistency_report([_row("t1", "no_pool", "clean", "a"), _row("t1", "no_pool", "x", "a")],
                           arms=("no_pool",))
    assert list(c) == ["no_pool"] and c["no_pool"]["consistency_pct"] == 100.0


def test_ood_summary_uses_the_shared_rule(tmp_path):
    """The OOD summary is scored from eval_rows with `consistency_report`, so both-decline
    counts as consistent there too, and the transition columns are carried."""
    import csv
    import json
    from src.expel.softpool_ood_summary import summary_rows
    d = tmp_path / "softpool_eval_ood_mmau_test"
    (d / "reverb_1.0s").mkdir(parents=True)
    (d / "softpool_report.json").write_text("{}")
    rows = []
    for arm in ("no_pool", "pool"):
        rows += [{**_row("t1", arm, "clean", None, "abstain"), "correct": False},
                 {**_row("t1", arm, "reverb_1.0s", None, "abstain"), "correct": False},
                 {**_row("t2", arm, "clean", "a"), "correct": True},
                 {**_row("t2", arm, "reverb_1.0s", "b"), "correct": False}]
    (d / "reverb_1.0s" / "eval_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    out = summary_rows(tmp_path)
    assert {r["prompt"] for r in out} == {"none", "soft_L8_full"}
    r = out[0]
    assert (r["model"], r["bench"], r["condition"], r["n"]) == ("qwen2.5-omni", "mmau", "reverb_1.0s", 2)
    assert r["consistency"] == 50.0 and r["both_decline"] == 50.0 and r["answer_changed"] == 50.0
    assert r["clean_acc"] == 50.0 and r["perturbed_acc"] == 0.0
