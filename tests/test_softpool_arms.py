"""Approach 11, the two trainer arms added 2026-09-09: `--objective rl` (no backward
through the model) and `--pool-type free` (untagged slots, pull + diversity key loss).
CPU only, no LALM.

Pinned:
  * the ES estimate is a LOSS gradient: on a quadratic reward it points at the optimum;
  * the diversity term pushes only OVER-used keys away from their query, in proportion,
    and is silent while usage is uniform;
  * a free pool never consumes the attack label (matching_loss raises) and its post-hoc
    tags round-trip through save/load and drive the route report;
  * the free-pool default keeps the parameter count of the tagged pool it is compared to.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.expel import softpool as sp  # noqa: E402
from src.expel.train_softpool import es_gradient, _usage_entropy, main as train_main  # noqa: E402

CONDS = ("clean", "noise_0dB", "mask_60")


def test_es_gradient_points_at_the_optimum_of_a_quadratic_reward():
    torch.manual_seed(0)
    target = torch.randn(4, 6)
    block = torch.zeros(4, 6)
    sigma = 0.1
    g = torch.zeros_like(block)
    for _ in range(400):
        eps = torch.randn_like(block) * sigma
        r_plus = -float(((block + eps) - target).pow(2).sum())
        r_minus = -float(((block - eps) - target).pow(2).sum())
        g += es_gradient(r_plus, r_minus, eps, sigma)
    g /= 400
    true_grad = 2 * (block - target)             # d(-reward)/d(block)
    cos = torch.nn.functional.cosine_similarity(g.flatten(), true_grad.flatten(), dim=0)
    assert cos > 0.9, float(cos)


def test_diversity_penalty_hits_only_overused_slots_and_is_silent_when_uniform():
    torch.manual_seed(0)
    pool = sp.FreePromptPool(6, CONDS, prompt_len=2, dim=8)
    q = torch.randn(8)
    idx, _ = pool.select(q)
    uniform = torch.ones(6)
    l_uniform = pool.key_loss(q, idx, uniform, lam_div=5.0)
    l_pull = pool.key_loss(q, idx, uniform, lam_div=0.0)
    assert torch.isclose(l_uniform, l_pull), "no slot is over its share, so no penalty"
    over = torch.zeros(6)
    over[idx[0]] = 30                                # one slot takes everything
    pool.keys.grad = None
    pool.key_loss(q, idx, over, lam_div=5.0).backward()
    g_over = pool.keys.grad.clone()
    pool.keys.grad = None
    pool.key_loss(q, idx, uniform, lam_div=5.0).backward()
    g_pull = pool.keys.grad.clone()
    # the over-used key moves differently from pure pull; the other selected keys do not
    assert not torch.allclose(g_over[idx[0]], g_pull[idx[0]])
    for i in idx[1:]:
        assert torch.allclose(g_over[i], g_pull[i])
    # sign: the penalty gradient on the over-used key is along +key-direction-toward-query
    # reduction, i.e. it opposes the pull term
    pen = g_over[idx[0]] - g_pull[idx[0]]
    assert torch.dot(pen, g_pull[idx[0]]) < 0


def test_free_pool_never_consumes_the_label_and_tags_are_posthoc(tmp_path):
    pool = sp.FreePromptPool(5, CONDS, prompt_len=2, dim=8)
    with pytest.raises(RuntimeError):
        pool.matching_loss(torch.randn(8), 1)
    hist = torch.tensor([[0, 9, 0], [3, 0, 0], [0, 0, 7], [1, 1, 5], [0, 0, 0]], dtype=torch.float)
    pool.set_posthoc_tags(hist)
    assert pool.slot_condition.tolist() == [1, 0, 2, 2, 0]
    pool.save(tmp_path / "p.pt")
    back = sp.SoftPromptPool.load(tmp_path / "p.pt")      # base-class load dispatches
    assert isinstance(back, sp.FreePromptPool) and back.hyper()["pool_type"] == "free"
    assert back.slot_condition.tolist() == [1, 0, 2, 2, 0]
    sims = torch.full((5,), -1.0); sims[2] = 1.0
    assert back.predicted_condition(sims) == "mask_60"


def test_tagged_pool_hyper_and_old_checkpoint_load_as_tagged(tmp_path):
    pool = sp.SoftPromptPool(CONDS, 2, prompt_len=2, dim=8)
    assert pool.hyper()["pool_type"] == "tagged" and pool.n_slots == 6
    ck = {"hyper": {k: v for k, v in pool.hyper().items() if k not in ("pool_type", "n_slots")},
          "state": pool.state_dict(), "extra": {}}
    torch.save(ck, tmp_path / "old.pt")
    back = sp.SoftPromptPool.load(tmp_path / "old.pt")
    assert type(back) is sp.SoftPromptPool


def test_usage_entropy_bounds():
    assert _usage_entropy(torch.tensor([5., 5., 5., 5.])) == 1.0
    assert _usage_entropy(torch.tensor([9., 0., 0., 0.])) == 0.0
    assert _usage_entropy(torch.zeros(4)) == 0.0


def test_cli_dry_run_accepts_both_arms(capsys):
    train_main(["--dry-run", "--objective", "rl", "--rl-reward", "match", "--train-limit", "1"])
    train_main(["--dry-run", "--pool-type", "free", "--n-slots", "44", "--train-limit", "1"])
    out = capsys.readouterr().out
    assert '"objective": "rl"' in out and '"pool_type": "free"' in out


def test_select_records_top_slot_for_usage_counters():
    pool = sp.SoftPromptPool(CONDS, 2, prompt_len=2, dim=8)
    q = torch.randn(8)
    idx, sims = pool.select(q)
    assert int(pool.last_top) == int(sims.argmax())
    assert int(pool.last_top) in idx.tolist()


def test_condition_similarity_is_ordinal_and_family_aware():
    conds = ("clean", "noise_20dB", "noise_0dB", "noise_-10dB", "noise_-20dB", "mask_60", "mask_100")
    S = sp.condition_similarity(conds)
    i = {c: k for k, c in enumerate(conds)}
    assert torch.allclose(S, S.T) and torch.all(S.diag() == 1)
    assert S[i["noise_0dB"], i["noise_-10dB"]] > S[i["noise_0dB"], i["noise_-20dB"]]
    assert S[i["noise_0dB"], i["clean"]] < 0.05
    assert S[i["clean"], i["noise_20dB"]] > 0.4
    # same level, different family: closer than far levels, farther than the same family
    assert S[i["noise_0dB"], i["mask_60"]] < S[i["noise_0dB"], i["noise_-10dB"]]
    assert S[i["mask_100"], i["noise_-20dB"]] > S[i["mask_100"], i["clean"]]
    # one level apart but across the answer/decline boundary: far
    assert S[i["noise_-10dB"], i["noise_-20dB"]] < 0.05


def test_contrastive_loss_pulls_toward_similar_prototype_and_tags_no_slot():
    torch.manual_seed(0)
    conds = ("clean", "noise_0dB", "noise_-10dB", "noise_-20dB")
    pool = sp.FreePromptPool(8, conds, prompt_len=2, dim=16)
    S = sp.condition_similarity(conds)
    protos = torch.full((4, 8), 1 / 8)
    protos[3] = 0; protos[3, 0] = 1.0            # noise_-20dB always selects slot 0
    protos[1] = 0; protos[1, 5] = 1.0            # noise_0dB always selects slot 5
    q = torch.randn(16)
    z0 = pool.selection_dist(q).detach()
    opt = torch.optim.SGD([pool.keys], lr=0.5)
    for _ in range(60):
        opt.zero_grad()
        pool.contrastive_key_loss(q, cond_idx=2, protos=protos, sim=S).backward()   # noise_-10dB
        opt.step()
    z1 = pool.selection_dist(q).detach()
    assert z1[5] > z0[5], "should move toward the similar condition's slot"
    assert z1[0] < z0[0], "and away from the dissimilar (unanswerable) one"
    assert pool.slot_condition.tolist() == [0] * 8, "no slot tagged by the loss"


def test_cli_contrastive_requires_free_pool(capsys):
    with pytest.raises(SystemExit):
        train_main(["--dry-run", "--key-loss", "contrastive", "--train-limit", "1"])


def test_rl_projection_maps_z_to_prompts_and_sync_rebuilds_them():
    from src.expel.train_softpool import RLProjection
    pool = sp.SoftPromptPool(CONDS, 2, prompt_len=2, dim=8)
    proj = RLProjection(pool, dim=4, seed=0)
    before = pool.prompts.detach().clone()
    proj.sync()
    assert torch.allclose(pool.prompts, before), "z = 0 leaves the prompts at P0"
    with torch.no_grad():
        proj.z[1] = torch.tensor([1.0, 0, 0, 0])
    proj.sync()
    assert not torch.allclose(pool.prompts[1], before[1]) and torch.allclose(pool.prompts[0], before[0])
    eps_z, eps = proj.perturb(torch.tensor([0, 2]), 0.1)
    assert eps_z.shape == (2, 4) and eps.shape == (4, 8)
    # the image of eps_z under A is what perturb returns
    assert torch.allclose(eps, (eps_z @ proj.A.T).reshape(-1, 8))


def test_cli_rl_dim_dry_run(capsys):
    train_main(["--dry-run", "--objective", "rl", "--rl-dim", "512", "--train-limit", "1"])
    assert '"rl_dim": 512' in capsys.readouterr().out


def test_single_pool_type_has_one_slot_no_keys_loss_and_full_block():
    from src.expel.train_softpool import _key_loss
    pool = sp.FreePromptPool(1, CONDS, prompt_len=24, dim=8, top_k=1)
    q = torch.randn(8)
    idx, _ = pool.select(q)
    assert idx.tolist() == [0] and pool.block(idx).shape == (24, 8)
    assert float(_key_loss(pool, q, idx, 1, 1.0, torch.zeros(1), 1.0)) == 0.0


def test_cli_single_forces_one_slot(capsys):
    train_main(["--dry-run", "--pool-type", "single", "--prompt-len", "24", "--train-limit", "1"])
    out = capsys.readouterr().out
    assert '"pool_type": "single"' in out and '"prompt_len": 24' in out


def test_text_prompt_goes_where_the_soft_block_goes_and_question_is_untouched():
    from src.expel.types import Task
    t = Task(id="Animal/x:single", track="Animal", hop="single", audio_path="/x.wav",
             stem="What animal is this?", choices={"a": "dog", "b": "cat"}, gold="a")
    conv = sp.build_direct_conversation(t, fmt="direct_abstain", text_prompt="Judge only from what is audible.")
    user = conv[1]["content"]
    assert user[0]["type"] == "audio"
    assert user[1]["text"].startswith("Judge only from what is audible.\n\n")
    assert user[1]["text"].endswith(f"{t.stem}\n{t.choices_block}")
    with pytest.raises(ValueError):
        sp.build_direct_conversation(t, n_marker=8, text_prompt="x")
    with pytest.raises(ValueError):
        sp.TextPromptHandle("   ")


def test_eval_cli_requires_exactly_one_of_pool_or_text_prompt(tmp_path):
    from src.expel.run_softpool import main as eval_main
    with pytest.raises(SystemExit):
        eval_main(["--dry-run", "--limit", "1"])
    f = tmp_path / "p.txt"; f.write_text("hello")
    eval_main(["--dry-run", "--limit", "1", "--text-prompt", str(f)])


def test_rl_block_flag_is_exclusive_with_rl_dim():
    with pytest.raises(SystemExit):
        train_main(["--dry-run", "--objective", "rl", "--rl-block", "--rl-dim", "512", "--train-limit", "1"])
    train_main(["--dry-run", "--objective", "rl", "--rl-block", "--train-limit", "1"])
