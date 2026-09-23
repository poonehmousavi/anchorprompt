"""Approach 11 (soft prompt pool): the plumbing and the invariant, on CPU with no LALM.

What is pinned here and why:
  * the selector's input is ONE field (audio path) and a Task cannot be passed in --
    the design invariant, in the same shape as RouterInput;
  * the marker token is a single id that repeats cleanly, so N markers = N rows;
  * the hook replaces exactly the marker rows and gradients reach the pool, not the
    embedding -- if this broke, training would print falling losses on nothing;
  * the teacher target is the model's own prediction, never gold;
  * the selection report's clean false-alarm rate is computed the way the README says.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.expel import softpool as sp  # noqa: E402
from src.expel.run_softpool import selection_report  # noqa: E402
from src.expel.train_softpool import build_examples, teacher_targets  # noqa: E402
from src.expel.types import Task, Trajectory  # noqa: E402

TASK = Task(id="Animal/dog1:single", track="Animal", hop="single", audio_path="/x/dog1.wav",
            stem="What animal is this?", choices={"a": "dog", "b": "cat"}, gold="a")


def _tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(sp.__dict__.get("MODEL_ID", "Qwen/Qwen2.5-Omni-7B"),
                                             local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer not in local cache: {e}")


# ---------------------------------------------------------------- invariant

def test_selector_input_has_exactly_one_field():
    assert {f.name for f in dataclasses.fields(sp.SoftPoolInput)} == {"audio_path"}
    with pytest.raises(dataclasses.FrozenInstanceError):
        sp.SoftPoolInput("a").audio_path = "b"


def test_projection_rejects_anything_but_a_task():
    inp = sp.softpool_input_from_task(TASK)
    assert inp == sp.SoftPoolInput(audio_path="/x/dog1.wav")
    with pytest.raises(TypeError):
        sp.softpool_input_from_task({"audio_path": "/x", "condition": "adv_wrong"})


def test_query_refuses_a_task():
    """audio_query must reject a Task BEFORE touching any model: the check is first."""
    with pytest.raises(TypeError, match="SoftPoolInput"):
        sp.audio_query(model=None, processor=None, inp=TASK)


def test_attacked_task_projects_to_the_same_shape():
    attacked = dataclasses.replace(TASK, condition="adv_wrong", attack_meta={"lure_letter": "b"})
    inp = sp.softpool_input_from_task(attacked)
    assert "adv_wrong" not in repr(inp) and "lure" not in repr(inp)


# ---------------------------------------------------------------- marker

def test_marker_tokenises_cleanly():
    tok = _tokenizer()
    ids = tok(sp.MARKER * 7, add_special_tokens=False)["input_ids"]
    assert len(ids) == 7 and len(set(ids)) == 1
    assert sp.marker_id(tok) == ids[0]


def test_conversation_contains_exactly_n_markers_and_none_otherwise():
    tok = _tokenizer()

    class P:  # tokenizer-only stand-in for the processor
        tokenizer = tok

        def apply_chat_template(self, conv, add_generation_prompt, tokenize):
            return tok.apply_chat_template(
                [{"role": m["role"], "content": " ".join(c.get("text", "") for c in m["content"])}
                 for m in conv], add_generation_prompt=add_generation_prompt, tokenize=False)

    assert sp.marker_count(P(), sp.build_direct_conversation(TASK)) == 0
    assert sp.marker_count(P(), sp.build_direct_conversation(TASK, n_marker=24)) == 24


def test_direct_conversation_keeps_question_verbatim():
    conv = sp.build_direct_conversation(TASK, n_marker=3)
    text = conv[1]["content"][1]["text"]
    assert text.endswith("What animal is this?\n(a) dog (b) cat")
    assert conv[1]["content"][0] == {"type": "audio", "audio": "/x/dog1.wav"}
    assert "<Reasoning>" not in conv[0]["content"][0]["text"]


# ---------------------------------------------------------------- injection

def test_hook_replaces_marker_rows_only_and_grads_reach_the_pool():
    embed = nn.Embedding(50, 6)
    embed.weight.requires_grad_(False)
    marker = 7
    pool = sp.SoftPromptPool(conditions=("clean", "x"), slots_per_condition=1, prompt_len=2, dim=6, top_k=1)
    ids = torch.tensor([[1, marker, marker, 3]])
    idx = torch.tensor([1])
    block = pool.block(idx)                       # (2, 6)
    inj = sp.SoftPromptInjector(embed, marker)
    with inj as h:
        h.set_block(block)
        out = embed(ids)
    assert inj.n_replaced == 2
    assert torch.allclose(out[0, 0], embed.weight[1]) and torch.allclose(out[0, 3], embed.weight[3])
    assert torch.allclose(out[0, 1], block[0]) and torch.allclose(out[0, 2], block[1])
    out.sum().backward()
    assert pool.prompts.grad is not None and pool.prompts.grad[1].abs().sum() > 0
    assert pool.prompts.grad[0].abs().sum() == 0          # unselected slot untouched
    assert embed.weight.grad is None
    # hook removed after the context: markers now embed as ordinary ids
    plain = embed(ids)
    assert torch.allclose(plain[0, 1], embed.weight[marker])


def test_hook_count_mismatch_is_an_error():
    embed = nn.Embedding(50, 6)
    inj = sp.SoftPromptInjector(embed, 7)
    with inj as h:
        h.set_block(torch.zeros(3, 6))
        with pytest.raises(RuntimeError, match="marker tokens"):
            embed(torch.tensor([[7, 7]]))


# ---------------------------------------------------------------- pool

def test_select_is_sorted_and_block_has_k_by_l_rows():
    pool = sp.SoftPromptPool(conditions=("clean", "a", "b"), slots_per_condition=2, prompt_len=4, dim=8, top_k=3)
    idx, sims = pool.select(torch.randn(8))
    assert list(idx) == sorted(idx.tolist()) and len(idx) == 3 and sims.shape == (6,)
    assert pool.block(idx).shape == (12, 8)
    assert pool.select(torch.randn(8), top_k=1)[0].shape == (1,)


def test_matching_loss_drives_query_to_assigned_condition():
    torch.manual_seed(0)
    pool = sp.SoftPromptPool(conditions=("clean", "adv_wrong", "mask_60"), slots_per_condition=2,
                             prompt_len=2, dim=16, top_k=1)
    queries = {c: torch.nn.functional.normalize(torch.randn(16), dim=-1) for c in range(3)}
    opt = torch.optim.Adam([pool.keys], lr=0.1)
    for _ in range(60):
        opt.zero_grad()
        loss = sum(pool.matching_loss(q, c) for c, q in queries.items())
        loss.backward()
        opt.step()
    for c, q in queries.items():
        assert pool.predicted_condition(pool.similarities(q)) == pool.conditions[c]


def test_save_load_roundtrip(tmp_path):
    pool = sp.SoftPromptPool(conditions=("clean", "x"), slots_per_condition=3, prompt_len=2, dim=5, top_k=2)
    pool.save(tmp_path / "pool.pt", {"epoch": 1})
    back = sp.SoftPromptPool.load(tmp_path / "pool.pt")
    assert back.hyper() == pool.hyper() and back.loaded_extra == {"epoch": 1}
    assert torch.equal(back.prompts, pool.prompts) and torch.equal(back.slot_condition, pool.slot_condition)


def test_init_from_vocab_shape():
    embed = nn.Embedding(2000, 5)
    init = sp.init_prompts_from_vocab(embed, m=4, prompt_len=3)
    assert init.shape == (4, 3, 5)


# ---------------------------------------------------------------- training targets

class _FakeActor:
    """Answers 'c' on everything, whatever the gold says."""

    def run(self, task, pool=None):
        return Trajectory(task_id=task.id, track=task.track, hop=task.hop, condition=task.condition,
                          answerable=True, family="softpool", context="", raw="(c)", reasoning="",
                          outcome="letter", pred="c", gold=task.gold, correct=False)


def test_teacher_target_is_pred_not_gold(tmp_path):
    t = dataclasses.replace(TASK, choices={"a": "dog", "b": "cat", "c": "cow"}, gold="a")
    targets = teacher_targets(_FakeActor(), [t], tmp_path)
    assert targets == {t.id: "c"}
    row = json.loads((tmp_path / "teacher.jsonl").read_text().splitlines()[0])
    assert row["pred"] == "c" and row["gold_for_analysis_only"] == "a"


def test_unanswerable_target_is_abstain_never_gold():
    """mask_100 / noise_-20dB: the pool is trained to DECLINE. Gold is not consulted."""
    from src.expel.attacks import UNANSWERABLE
    assert {"mask_100", "noise_-20dB"} <= set(UNANSWERABLE)
    t = dataclasses.replace(TASK, condition="mask_100", answerable=False, gold="a")
    assert sp.target_text(t, "b") == sp.ABSTAIN_TARGET
    assert sp.target_text(TASK, "b") == "(b)"
    assert sp.target_ids is not None
    # the abstain phrase must be recognised by the shared parser as an abstention
    from src.parsing import parse_outcome
    assert parse_outcome(sp.ABSTAIN_TARGET, {"a": "dog", "b": "cat"}) == ("abstain", None)


def test_abstain_format_is_shared_by_both_arms_and_names_the_phrase():
    base = sp.build_direct_conversation(TASK, fmt="direct_abstain")[0]["content"][0]["text"]
    pooled = sp.build_direct_conversation(TASK, n_marker=4, fmt="direct_abstain")[0]["content"][0]["text"]
    assert base == pooled and sp.ABSTAIN_TARGET in base and sp.ABSTAIN_TARGET not in sp.SYSTEM_DIRECT
    with pytest.raises(KeyError):
        sp.build_direct_conversation(TASK, fmt="cot")


def test_train_cli_refuses_unanswerable_without_abstain_format():
    from src.expel import train_softpool
    # The default format is now direct_abstain (the default attack set holds the two
    # unanswerable levels), so the guard is exercised with an explicit plain format.
    with pytest.raises(SystemExit, match="direct_abstain"):
        train_softpool.main(["--dry-run", "--attacks", "mask_60", "mask_100", "--format", "direct"])
    train_softpool.main(["--dry-run", "--attacks", "mask_60", "mask_100"])


def test_fewshot_flag_parses_and_dry_runs(capsys):
    from src.expel import train_softpool
    train_softpool.main(["--dry-run", "--fewshot", "mmau:20", "mmar:20"])
    assert "mmau:20" in capsys.readouterr().out


def test_build_examples_counts_unrenderable_attacks():
    ex, skipped = build_examples([TASK], ("clean", "text_inject", "adv_wrong"), seed=0)
    conds = [c for _, c in ex]
    assert 0 in conds and 1 in conds           # clean and text_inject render without files
    assert skipped.get("adv_wrong", 0) == 1    # no adversarial file for /x/dog1.wav
    assert all(t.gold == TASK.gold for t, _ in ex)


# ---------------------------------------------------------------- reporting

def test_selection_report_false_alarm_counts_clean_once():
    rows = {
        "adv_wrong": [
            {"task_id": "t1", "condition": "clean", "arm": "pool", "route": "clean"},
            {"task_id": "t2", "condition": "clean", "arm": "pool", "route": "adv_wrong"},
            {"task_id": "t1", "condition": "attacked", "arm": "pool", "route": "adv_wrong"},
            {"task_id": "t2", "condition": "attacked", "arm": "pool", "route": "clean"},
            {"task_id": "t1", "condition": "clean", "arm": "no_pool", "route": None},
        ],
        "mask_60": [
            {"task_id": "t1", "condition": "clean", "arm": "pool", "route": "clean"},   # duplicate clean
            {"task_id": "t2", "condition": "clean", "arm": "pool", "route": "adv_wrong"},
            {"task_id": "t1", "condition": "attacked", "arm": "pool", "route": "mask_60"},
        ],
    }
    rep = selection_report(rows, ("clean", "adv_wrong", "mask_60"))
    assert rep["confusion"]["clean"] == {"clean": 1, "adv_wrong": 1}
    assert rep["clean_false_alarm_pct"] == 50.0
    assert rep["clean_audio_false_alarm_pct"] == 50.0
    assert rep["recall_pct"]["adv_wrong"] == 50.0 and rep["recall_pct"]["mask_60"] == 100.0


def test_selection_report_text_inject_is_not_an_audio_alarm():
    rows = {"adv_wrong": [
        {"task_id": "t1", "condition": "clean", "arm": "pool", "route": "text_inject"},
        {"task_id": "t2", "condition": "clean", "arm": "pool", "route": "clean"},
        {"task_id": "t3", "condition": "clean", "arm": "pool", "route": "mask_60"},
        {"task_id": "t4", "condition": "clean", "arm": "pool", "route": "clean"}]}
    rep = selection_report(rows, ("clean", "adv_wrong", "mask_60", "text_inject"))
    assert rep["clean_false_alarm_pct"] == 50.0          # any non-clean slot
    assert rep["clean_audio_false_alarm_pct"] == 25.0    # text_inject audio == clean audio


def test_placebo_selection_modes_use_no_audio():
    """random / fixed never call audio_query; fixed picks that condition's slots."""
    pool = sp.SoftPromptPool(conditions=("clean", "adv_wrong"), slots_per_condition=2, prompt_len=2, dim=4, top_k=2)

    class M:  # model stand-in that must never be touched
        def get_input_embeddings(self):
            return nn.Embedding(10, 4)

    class _Tok:
        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": [151650]}

    class P:
        tokenizer = _Tok()
    actor = sp.SoftPoolActor(M(), P(), top_k=2, selection="fixed:adv_wrong")
    _, idx, sims = actor._select(pool, TASK)
    assert idx.tolist() == [2, 3] and pool.predicted_condition(sims) == "adv_wrong"
    actor = sp.SoftPoolActor(M(), P(), top_k=2, selection="random")
    _, idx1, _ = actor._select(pool, TASK)
    _, idx2, _ = actor._select(pool, TASK)
    assert idx1.tolist() == idx2.tolist() and len(idx1) == 2       # seeded by task id
    with pytest.raises(ValueError):
        sp.SoftPoolActor(M(), P(), selection="bogus")


# ---------------------------------------------------------------- CLIs

def test_train_and_eval_dry_run(capsys):
    from src.expel import run_softpool, train_softpool
    train_softpool.main(["--dry-run", "--train-limit", "2", "--epochs", "1", "--out", "/nonexistent"])
    run_softpool.main(["--dry-run", "--pool", "/nonexistent/pool.pt", "--limit", "2"])
    out = capsys.readouterr().out
    assert '"conditions"' in out and "text_inject" in out and "/nonexistent/pool.pt" in out


def test_transfer_benchmarks_refuse_adv_wrong():
    from src.expel import run_softpool
    with pytest.raises(SystemExit, match="no renders"):
        run_softpool.main(["--dry-run", "--pool", "/x.pt", "--benchmark", "mmau",
                           "--conditions", "adv_wrong", "mask_60"])
    run_softpool.main(["--dry-run", "--pool", "/x.pt", "--benchmark", "mmar",
                       "--conditions", "mask_60", "noise_0dB", "text_inject"])


def test_init_pool_rejects_mismatched_layout(tmp_path):
    """--init must fail loudly when the saved slot layout differs: the matching loss is
    keyed by condition index, so a silent mismatch trains slots under the wrong label."""
    import pytest
    from src.expel.softpool import SoftPromptPool
    from src.expel.train_softpool import load_init_pool
    conds = ("clean", "noise_0dB")
    pool = SoftPromptPool(conds, 2, 4, 16, 1, seed=0)
    path = tmp_path / "pool.pt"
    pool.save(path, {})
    same = load_init_pool(path, conds, 2, 4)
    assert tuple(same.conditions) == conds
    with pytest.raises(SystemExit):
        load_init_pool(path, ("clean", "mask_60"), 2, 4)
    with pytest.raises(SystemExit):
        load_init_pool(path, conds, 3, 4)


def test_defaults_cover_every_paper_level():
    """User rule (2026-09-08): training and evaluation default to ALL noise SNRs and mask
    ratios from the faithfulness paper, not the original 4-attack subset."""
    from src.expel.softpool import PAPER_LEVELS, FULL_CONDITIONS
    assert set(PAPER_LEVELS) == {f"noise_{d}dB" for d in (20, 10, 0, -10, -20)} | {
        f"mask_{p}" for p in (20, 40, 60, 80, 100)}
    assert set(PAPER_LEVELS) < set(FULL_CONDITIONS)
    assert {"adv_wrong", "text_inject", "clean"} < set(FULL_CONDITIONS)


def test_train_and_eval_defaults_use_all_levels(capsys, tmp_path):
    from src.expel import train_softpool, run_softpool
    from src.expel.softpool import PAPER_LEVELS
    train_softpool.main(["--dry-run", "--out", str(tmp_path / "t")])
    cfg = capsys.readouterr().out
    assert all(level in cfg for level in PAPER_LEVELS) and '"direct_abstain"' in cfg
    for bench in ("sakura", "mmau"):
        run_softpool.main(["--dry-run", "--pool", "x.pt", "--benchmark", bench,
                           "--out", str(tmp_path / bench)])
        cfg = capsys.readouterr().out
        assert all(level in cfg for level in PAPER_LEVELS) and "text_inject" in cfg
        assert ("adv_wrong" in cfg) == (bench == "sakura")


def test_query_cache_tolerates_a_truncated_file_and_writes_atomically(tmp_path, monkeypatch):
    """Job 10752383: three evals wrote the same query .npy concurrently and one read a
    header-only file (EOFError). A bad cache entry must be recomputed, and the write must
    go through a temp file + os.replace so readers never see a partial file."""
    import numpy as np
    monkeypatch.setattr(sp, "CACHE_ROOT", tmp_path)
    inp = sp.SoftPoolInput(audio_path=str(tmp_path / "x.wav"))
    cp = sp._cache_path(inp.audio_path)
    cp.parent.mkdir(parents=True)
    cp.write_bytes(b"\x93NUMPY")                      # truncated header, as on BeeGFS

    calls = []
    monkeypatch.setattr(sp, "_compute_query", lambda m, p, path: (calls.append(path), torch.ones(4))[1])
    q = sp.audio_query(model=None, processor=None, inp=inp)
    assert calls == [inp.audio_path] and q.shape == (4,)
    assert np.load(cp).shape == (4,)                  # rewritten whole
    assert not list(cp.parent.glob("*.tmp.npy"))      # temp file renamed away
    assert sp.audio_query(model=None, processor=None, inp=inp).shape == (4,)
    assert len(calls) == 1                            # second call served from cache


def test_query_cache_is_namespaced_by_model_family(tmp_path, monkeypatch):
    """A Qwen2.5 query (3584-d) must never be served to a Qwen3-Omni pool (2048-d): the
    cache path carries the model family, with the default family on the legacy flat path."""
    import src.expel.softpool as sp
    monkeypatch.setattr(sp, "CACHE_ROOT", tmp_path)
    a = sp._cache_path("/x/y.wav")
    b = sp._cache_path("/x/y.wav", "qwen3_omni")
    assert a != b and a.parent == tmp_path / "softpool_query" and b.parent == tmp_path / "softpool_query" / "qwen3_omni"


def test_single_prompt_never_queries_the_audio_tower(monkeypatch):
    """n_slots == 1: selection is trivial, so neither the trainer nor the actor may call
    audio_query (which is Qwen-specific and lives in the model's own hidden space)."""
    import torch
    import src.expel.softpool as sp
    from src.expel import train_softpool as ts
    pool = sp.FreePromptPool(1, ("clean", "mask_100"), prompt_len=2, dim=16, top_k=1, seed=0)
    def boom(*a, **k):
        raise AssertionError("audio_query called for a single prompt")
    monkeypatch.setattr(sp, "audio_query", boom)
    monkeypatch.setattr(ts, "audio_query", boom)
    actor = sp.SoftPoolActor.__new__(sp.SoftPoolActor)
    actor.top_k, actor.selection, actor.fixed_condition = None, "audio", None
    q, idx, sims = actor._select(pool, None)
    assert q is None and idx.tolist() == [0] and sims.tolist() == [1.0]
    assert pool.last_idx.tolist() == [0] and int(pool.last_top) == 0   # usage counters read these
    assert float(ts._key_loss(pool, None, idx, 0, 1.0, torch.zeros(1), 0.0)) == 0.0


def test_teacher_forced_inputs_use_model_kind_dispatch(monkeypatch):
    """Job 10770086: the trainer built inputs through the Qwen-only path and AF3's encoder
    died on a 376-vs-1500 frame mismatch. The trainer must go through the same
    `build_inputs` the eval path uses, and must not pass Qwen's `use_audio_in_video` to AF3."""
    import torch
    from src.expel import softpool
    from src import model as model_mod
    calls = {}

    class _Batch(dict):
        def to(self, *_):
            return self

    def fake_build(model, processor, conversation):
        calls["kind"] = model_mod.model_kind(model)
        return _Batch(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3, dtype=torch.long))

    monkeypatch.setattr(model_mod, "build_inputs", fake_build)

    class _M:
        _reprompt_kind = "af3"
        device = "cpu"
        dtype = torch.float32

    x = softpool.teacher_forced_inputs(_M(), None, [{"role": "user", "content": "q"}], [7, 8])
    assert calls["kind"] == "af3"
    assert x["input_ids"].tolist() == [[1, 2, 3, 7, 8]]
    assert x["labels"].tolist() == [[-100, -100, -100, 7, 8]]

    seen = {}

    class _Model(_M):
        def __call__(self, **kw):
            seen.update(kw)
            class _O:
                logits = torch.zeros(1, kw["input_ids"].shape[1], 10)
            return _O()

    softpool.answer_loss(_Model(), dict(x))
    assert "use_audio_in_video" not in seen
    _M._reprompt_kind = "qwen_omni"
    seen.clear(); softpool.answer_loss(_Model(), dict(softpool.teacher_forced_inputs(_M(), None, [], [7])))
    assert seen.get("use_audio_in_video") is False


@pytest.mark.skipif(not Path("data/sakura/data/Animal/metadata.json").exists(),
                    reason="needs the SAKURA data (see README)")
def test_abstain_anyway_lifts_the_native_format_guard(monkeypatch, tmp_path):
    """AF3 native has no decline option, so the trainer refuses the unanswerable levels --
    unless --abstain-anyway says the soft prompt is to TEACH the phrase (user, 2026-09-12)."""
    import pytest
    from src.expel import train_softpool as ts
    base = ["--model", "af3", "--format", "af3_native", "--pool-type", "single", "--prompt-len", "2",
            "--attacks", "mask_100", "--train-limit", "1", "--epochs", "1", "--out", str(tmp_path)]
    with pytest.raises(SystemExit, match="abstain-anyway"):
        ts.main(base)
    # with the flag the guard passes and the run proceeds to the model load, which we stop here
    monkeypatch.setattr("src.model.load_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reached load")))
    with pytest.raises(RuntimeError, match="reached load"):
        ts.main(base + ["--abstain-anyway"])


def test_target_letter_case_follows_the_format():
    """Native layouts render uppercase letters and the model answers "(B) ..."; the
    teacher-forced target must match that case (AF3 smoke 10770191 learned nothing)."""
    from src.expel.softpool import target_text, ABSTAIN_TARGET
    from src.expel.types import Task
    t = Task(id="x", track="Animal", hop="single", audio_path="a.wav", stem="q?",
             choices={"a": "cat", "b": "dog"}, gold="b", condition="clean")
    assert target_text(t, "b") == "(b)"
    assert target_text(t, "b", "direct_abstain") == "(b)"
    assert target_text(t, "b", "af3_native") == "(B)"
    assert target_text(t, "B", "direct_abstain") == "(b)"
    tu = Task(id="x", track="Animal", hop="single", audio_path="a.wav", stem="q?",
              choices={"a": "cat", "b": "dog"}, gold="b", condition="mask_100", answerable=False)
    assert target_text(tu, "b", "af3_native") == ABSTAIN_TARGET


def _fake_model(vocab=10, kind="qwen_omni"):
    """Forward returns logits that favour token (input_id + 1) at every position, so the
    'answer' is predictable per row; records kwargs and the input_ids it saw."""
    import torch

    class _M:
        _reprompt_kind = kind
        device = "cpu"
        dtype = torch.float32
        seen = []

        def __call__(self, **kw):
            _M.seen.append(kw)
            ids = kw["input_ids"]
            lg = torch.full((ids.shape[0], ids.shape[1], vocab), -5.0)
            nxt = (ids + 1).clamp(max=vocab - 1)
            lg.scatter_(2, nxt.unsqueeze(-1), 5.0)

            class _O:
                logits = lg
            return _O()
    return _M


def test_repeat_inputs_repeats_only_batch_one_tensors():
    import torch
    from src.expel.softpool import repeat_inputs
    x = {"input_ids": torch.tensor([[1, 2, 3]]), "feat": torch.zeros(1, 4, 5), "flag": True, "k": torch.zeros(2, 3)}
    y = repeat_inputs(x, 3)
    assert y["input_ids"].shape == (3, 3) and y["feat"].shape == (3, 4, 5)
    assert y["flag"] is True and y["k"].shape == (2, 3)
    assert repeat_inputs(x, 1)["input_ids"].shape == (1, 3)


def test_answer_loss_per_row_matches_pooled_and_slices_targets():
    import torch
    from src.expel.softpool import answer_loss
    M = _fake_model()
    ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 9]])          # targets = last 2 tokens
    labels = torch.tensor([[-100, -100, -100, 4, 5], [-100, -100, -100, 4, 9]])
    loss, agree = answer_loss(M(), {"input_ids": ids, "labels": labels}, per_row=True)
    assert loss.shape == (2,) and agree.tolist() == [True, False]   # row 1 predicts 5 after 4, not 9
    pooled, ag = answer_loss(M(), {"input_ids": ids[:1], "labels": labels[:1]})
    assert torch.isclose(pooled, loss[0]) and ag is True
    assert "use_audio_in_video" in M.seen[-1]
    Maf = _fake_model(kind="af3")
    answer_loss(Maf(), {"input_ids": ids[:1], "labels": labels[:1]})
    assert "use_audio_in_video" not in Maf.seen[-1]


def test_forward_reward_tokens_is_black_box_and_writes_tokens_into_the_text():
    """Discrete arm: candidates go in as token ids at the marker positions; no injector,
    no embedding read, no graph."""
    import torch
    from src.expel import train_softpool as ts
    M = _fake_model(vocab=20)
    marker = 15
    x = {"input_ids": torch.tensor([[1, marker, marker, 7, 8]]),
         "labels": torch.tensor([[-100, -100, -100, -100, 8]])}
    r, ag = ts._forward_reward_tokens(M(), marker, [[2, 3], [4, 6]], x, "match", chunk=1)
    assert r.tolist() == [1.0, 1.0]                                   # 7 -> 8 regardless of prompt
    seen = torch.cat([k["input_ids"] for k in M.seen[-2:]])
    assert seen.tolist() == [[1, 2, 3, 7, 8], [1, 4, 6, 7, 8]]         # markers replaced, per row
    assert not (seen == marker).any()
    with __import__("pytest").raises(RuntimeError):
        ts._forward_reward_tokens(M(), marker, [[2, 3, 4]], x, "match")


def test_discrete_candidates_are_random_tokens_only():
    import random
    from src.expel.train_softpool import discrete_candidates
    c = discrete_candidates([5000, 6000, 7000], 1, 100_000, 8, random.Random(0))
    assert c[0] == 6000 and len(c) == 8 and len(set(c)) == 8 and all(1000 <= t < 100_000 for t in c)


def test_rl_pairs_estimate_averages_pairs(monkeypatch):
    """--rl-pairs P: the ES estimate is the mean over P antithetic pairs and lands in
    pool.prompts.grad; no backward through the model is run."""
    import torch
    from src.expel import train_softpool as ts
    from src.expel.softpool import FreePromptPool
    pool = FreePromptPool(1, ["clean", "x"], 2, 4, 1, init_prompts=torch.zeros(1, 2, 4), seed=0)
    calls = {}

    def fake_prepare(model, processor, pool, task, target, fmt):
        idx = torch.zeros(1, dtype=torch.long)
        pool.last_idx, pool.last_top = idx, idx[0]
        return None, idx, torch.ones(1), {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[-100, 2]])}

    def fake_batch(model, injector, blocks, x, reward, chunk):
        calls["n"] = blocks.shape[0]
        # reward = +sum of the block: r_plus - r_minus = 2*sum(eps) -> estimate points along eps
        return blocks.sum(dim=(1, 2)), torch.ones(blocks.shape[0])

    monkeypatch.setattr(ts, "_prepare", fake_prepare)
    monkeypatch.setattr(ts, "_forward_reward_batch", fake_batch)
    monkeypatch.setattr(ts, "_key_loss", lambda *a, **k: torch.tensor(0.0))
    out = ts._step(None, None, pool, None, None, 0, "b", 1.0, True, objective="rl", rl_sigma=0.1,
                   rl_reward="logp", rl_pairs=4, rl_chunk=8)
    assert calls["n"] == 8                                             # 2 x P rows in one call
    g = pool.prompts.grad
    assert g is not None and g.shape == (1, 2, 4)
    assert g.abs().sum() > 0                                           # an estimate was written
    assert out[0] is None
