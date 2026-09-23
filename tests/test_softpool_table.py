import json

from src.expel.softpool_table import format_table, per_category


def _row(track, cond, tid, arm, correct, answerable=True, abstained=False):
    return {"track": track, "condition": cond, "task_id": tid, "arm": arm, "correct": correct,
            "answerable": answerable, "abstained": abstained}


def test_net_is_paired_per_item_and_per_track():
    rows = [
        # Animal: item 1 repaired, item 2 damaged, item 3 unchanged -> net 0
        _row("Animal", "mask_60", "a1", "no_pool", False), _row("Animal", "mask_60", "a1", "pool", True),
        _row("Animal", "mask_60", "a2", "no_pool", True), _row("Animal", "mask_60", "a2", "pool", False),
        _row("Animal", "mask_60", "a3", "no_pool", True), _row("Animal", "mask_60", "a3", "pool", True),
        # Emotion: one repaired -> net +1
        _row("Emotion", "mask_60", "e1", "no_pool", False), _row("Emotion", "mask_60", "e1", "pool", True),
    ]
    t = per_category(rows)
    assert t["Animal"]["mask_60"]["net"] == 0
    assert t["Animal"]["mask_60"]["repaired"] == 1 and t["Animal"]["mask_60"]["damaged"] == 1
    assert t["Animal"]["mask_60"]["none"] == 66.7 and t["Animal"]["mask_60"]["pool"] == 66.7
    assert t["Emotion"]["mask_60"]["net"] == 1 and t["Emotion"]["mask_60"]["pool"] == 100.0


def test_unanswerable_scores_a_decline_as_correct():
    rows = [
        _row("Gender", "mask_100", "g1", "no_pool", False, answerable=False, abstained=False),
        _row("Gender", "mask_100", "g1", "pool", False, answerable=False, abstained=True),
    ]
    t = per_category(rows)
    s = t["Gender"]["mask_100"]
    assert s["unanswerable"] and s["none"] == 0.0 and s["pool"] == 100.0 and s["net"] == 1
    assert s["abstain_pool"] == 100.0


def test_unpaired_rows_are_ignored_and_table_formats():
    rows = [_row("Animal", "noise_0dB", "a1", "no_pool", True)]  # no pool arm
    assert per_category(rows) == {}
    rows += [_row("Animal", "noise_0dB", "a1", "pool", True)]
    out = format_table(per_category(rows))
    assert "== Animal" in out and "noise_0dB" in out
    json.dumps(per_category(rows))  # serialisable


def test_loader_takes_level_from_folder_and_dedupes_clean(tmp_path):
    from src.expel.softpool_table import _load_rows
    for level in ("mask_60", "noise_0dB"):
        d = tmp_path / level
        d.mkdir()
        with open(d / "eval_rows.jsonl", "w") as fh:
            for arm in ("no_pool", "pool"):
                fh.write(json.dumps(_row("Animal", "clean", "a1", arm, True)) + "\n")
                fh.write(json.dumps(_row("Animal", "attacked", "a1", arm, False)) + "\n")
    rows = _load_rows(tmp_path)
    conds = sorted({r["condition"] for r in rows})
    assert conds == ["clean", "mask_60", "noise_0dB"]
    assert sum(r["condition"] == "clean" for r in rows) == 2  # once per arm, not per folder
