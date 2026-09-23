import csv
from pathlib import Path

from src.expel.softpool_bundle import bundle, classify, main


def test_classify_maps_dir_names_to_families():
    assert classify("softpool_eval_text_v3_sakura_p3_test") == ("qwen2.5-omni", "v3", "sakura", "test")
    assert classify("softpool_eval_text_v3_sakura_p7_test") == ("qwen2.5-omni", "v3", "sakura", "test")
    assert classify("softpool_eval_af3_text_v4nat_mmar_test") == ("af3", "v4nat", "mmar", "test")
    assert classify("softpool_eval_qwen3-omni_text_v3_mmau_test") == ("qwen3-omni", "v3", "mmau", "test")
    assert classify("softpool_eval_single_L8_full_sakura_p1_test") == ("qwen2.5-omni", "soft_L8_full", "sakura", "test")
    assert classify("softpool_eval_single_L8_K10_mmau_val") == ("qwen2.5-omni", "soft_L8_K10", "mmau", "val")
    assert classify("softpool_eval_text_v3smoke_sakura_p1_test") is None      # smokes excluded
    assert classify("softpool_eval_text_v3_mmau") is None                      # dev runs excluded


def _write(d: Path, rows):
    d.mkdir(parents=True)
    cols = ["benchmark", "model", "split", "condition", "arm", "task_id", "recording_id", "track", "hop",
            "gold", "pred", "outcome", "correct", "abstained", "answerable", "lure_letter", "raw_full", "choices"]
    with open(d / "per_item.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def test_bundle_merges_parts_names_prompt_and_dedups_clean(tmp_path):
    base = dict(benchmark="sakura", model="qwen2.5-omni", split="test", task_id="t1", recording_id="r1",
                track="Animal", hop="single", gold="a", outcome="letter", answerable="True")
    p1 = tmp_path / "softpool_eval_text_v3_sakura_p1_test"
    p2 = tmp_path / "softpool_eval_text_v3_sakura_p2_test"
    for d, cond in ((p1, "noise_10dB"), (p2, "noise_0dB")):
        _write(d, [dict(base, condition="clean", arm="no_pool", pred="a", correct="True", abstained="False"),
                   dict(base, condition="clean", arm="pool", pred="b", correct="False", abstained="False"),
                   dict(base, condition=cond, arm="no_pool", pred="b", correct="False", abstained="False"),
                   dict(base, condition=cond, arm="pool", pred="a", correct="True", abstained="False")])
    out = bundle([p1, p2], "qwen2.5-omni", "v3", "sakura", "test", tmp_path / "res")
    rows = list(csv.DictReader(open(out)))
    assert "arm" not in rows[0] and set(r["prompt"] for r in rows) == {"none", "v3"}
    assert sum(r["condition"] == "clean" for r in rows) == 2          # once per prompt, not per part
    assert sorted(set(r["condition"] for r in rows)) == ["clean", "noise_0dB", "noise_10dB"]
    assert out.name == "qwen2.5-omni__v3__sakura__test.csv"           # dot in the model name survives
    md = (out.parent / "qwen2.5-omni__v3__sakura__test.summary.md").read_text()
    assert "| noise_0dB | 0.0 | 0.0 | 100.0 | 0.0 | 1 |" in md and "## Animal" in md and "## ALL" in md


def test_main_auto_writes_index(tmp_path):
    base = dict(benchmark="mmau", model="af3", split="test", task_id="t", recording_id="r", track="MMAU-sound",
                hop="single", gold="a", pred="a", outcome="letter", correct="True", abstained="False", answerable="True")
    _write(tmp_path / "softpool_eval_af3_text_v3nat_mmau_test",
           [dict(base, condition="clean", arm="no_pool"), dict(base, condition="clean", arm="pool")])
    main(["--out", str(tmp_path / "res"), "--root", str(tmp_path), "--auto", "--split", "test"])
    assert (tmp_path / "res" / "af3__v3nat__mmau__test.csv").exists()
    assert "af3__v3nat__mmau__test.csv" in (tmp_path / "res" / "index.md").read_text()
