"""Train the soft prompt pool (approach 11). Frozen LALM; only prompts and keys learn.

Objective, per example (a train task under one condition):
    loss = CE(answer letter | audio, soft prompts, question)  +  lambda * matching
where the answer TARGET is the model's own greedy answer on the CLEAN twin of the same
task under the same direct format and NO soft prompts (the teacher pass). Gold is never
read for training. `matching` is a cross-entropy over conditions from the key
similarities -- the one place the attack label is consumed, training-side only.

Usage (smoke, then full):
    python -m src.expel.train_softpool --train-limit 2 --epochs 1 --out output/softpool_smoke
    python -m src.expel.train_softpool --train-limit 200 --epochs 3 --out output/softpool

Two experiment axes, each one flag so a run differs from the baseline in ONE thing:
  --objective grad | rl
      grad: cross-entropy on the target, backward through the frozen model (default).
      rl:   NO backward through the model. The prompt block is a Gaussian policy around
            the pool's prompts (sigma = --rl-sigma); each example draws one antithetic
            pair (+eps, -eps), scores both by a reward that needs only a forward pass,
            and updates the selected prompts with the ES / REINFORCE estimate
            (r+ - r-) / (2 sigma) * eps. Reward: --rl-reward logp (mean log-prob of the
            target = consistency with the clean answer, or with the abstain phrase) or
            match (1 if the greedy answer equals the target). Keys train exactly as in
            the grad arm (matching or key loss, which never touch the model).
  --pool-type tagged | free
      tagged: 4 slots per condition, supervised matching loss on the keys (default).
      free:   --n-slots untagged slots, L2P pull loss + a diversity penalty
              (--lambda-div) on over-used slots. The attack label is never consumed.
              Tags for the eval's route report are assigned post-hoc from usage.

Outputs under --out: teacher.jsonl, train_log.jsonl, pool.pt (best held-out agreement),
pool_last.pt, train_report.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from src.expel.attacks import get_attack
from src.expel.data import load_sakura, split_train_dev, take, wav_id
from src.expel.softpool import (repeat_inputs, CONDITIONS, FULL_CONDITIONS, FORMATS, FreePromptPool, SoftPromptPool, SoftPromptInjector,
                                condition_similarity,
                                SoftPoolActor, answer_loss, build_direct_conversation,
                                init_prompts_from_vocab, marker_id, audio_query,
                                softpool_input_from_task, target_ids, target_text,
                                teacher_forced_inputs)
from src.expel.types import Task


def _usage_entropy(usage: torch.Tensor) -> float:
    """Normalised entropy of the per-epoch slot usage: 1.0 = every slot used equally,
    0.0 = one slot only. The diversity statistic for both pool types."""
    u = usage.float()
    if u.sum() <= 0 or u.numel() < 2:
        return 0.0
    p = u / u.sum()
    p = p[p > 0]
    return round(float(-(p * p.log()).sum() / math.log(u.numel())), 4)


def load_init_pool(path: Path, conditions, slots_per_condition: int, prompt_len: int) -> SoftPromptPool:
    """A saved pool to fine-tune from. Its slot layout must equal this run's, otherwise the
    condition-keyed matching loss would train slots under the wrong label."""
    pool = SoftPromptPool.load(path)
    want = {"conditions": tuple(conditions), "slots_per_condition": slots_per_condition,
            "prompt_len": prompt_len}
    have = {"conditions": tuple(pool.conditions), "slots_per_condition": pool.slots_per_condition,
            "prompt_len": pool.prompt_len}
    if want != have:
        raise SystemExit(f"--init {path} does not match this run: saved {have}, asked {want}")
    return pool


def teacher_targets(actor: SoftPoolActor, tasks: list[Task], out: Path) -> dict[str, str]:
    """Greedy clean answers, direct format, no soft prompts. Writes teacher.jsonl.
    The gold letter is written next to the prediction for the READER's analysis only;
    training reads `pred` and nothing else (see test_teacher_target_is_pred_not_gold)."""
    targets: dict[str, str] = {}
    n_right = 0
    with open(out / "teacher.jsonl", "w") as f:
        for n, t in enumerate(tasks):
            assert t.condition == "clean"
            traj = actor.run(t, None)
            row = {"task_id": t.id, "pred": traj.pred, "outcome": traj.outcome,
                   "gold_for_analysis_only": t.gold, "raw": traj.raw}
            f.write(json.dumps(row) + "\n")
            if traj.pred is not None:
                targets[t.id] = traj.pred
                n_right += traj.pred == t.gold
            if (n + 1) % 20 == 0:
                print(f"  teacher {n + 1}/{len(tasks)}", flush=True)
    print(f"teacher: {len(targets)}/{len(tasks)} parseable, "
          f"{100.0 * n_right / max(1, len(targets)):.1f}% agree with gold (analysis only)",
          flush=True)
    return targets


def build_examples(tasks: list[Task], conditions, seed: int) -> tuple[list[tuple[Task, int]], dict]:
    """(task_under_condition, condition_index) for every task x condition. Attacks that
    cannot be rendered for an item are skipped and counted, never silently dropped."""
    ex, skipped = [], {}
    for t in tasks:
        for ci, c in enumerate(conditions):
            if c == "clean":
                ex.append((t, ci))
                continue
            try:
                ex.append((get_attack(c)(t, seed=seed), ci))
            except Exception as e:                       # noqa: BLE001
                skipped[c] = skipped.get(c, 0) + 1
                if skipped[c] <= 2:
                    print(f"  skip {c} on {t.id}: {type(e).__name__}: {e}", flush=True)
    return ex, skipped


def _backward_keys(l_key: torch.Tensor, accum: int) -> bool:
    """Backward the key loss under rl. A single-slot pool has no key objective: `_key_loss`
    returns a constant zero with no graph, and calling backward on it raises. Returns
    whether a backward ran."""
    if not l_key.requires_grad:
        return False
    (l_key / accum).backward()
    return True


def es_gradient(r_plus: float, r_minus: float, eps: torch.Tensor, sigma: float) -> torch.Tensor:
    """Antithetic ES / REINFORCE estimate of d(-reward)/d(block) for a Gaussian policy
    N(block, sigma^2 I): the LOSS gradient, so an optimiser that minimises moves the
    block toward higher reward. Pure tensor arithmetic; no model involved."""
    return -((float(r_plus) - float(r_minus)) / (2.0 * float(sigma))) * eps


class RLProjection:
    """Black-box-prompt-tuning subspace for the rl objective: each slot's prompt is
    P0 + A z_s with a SHARED random A (L*D, d), columns scaled 1/sqrt(d) so a z with
    per-coordinate std s gives a prompt perturbation with per-coordinate std s. The ES
    estimate then lives in d coordinates per slot instead of L*D, and pool.prompts is
    rebuilt from (P0, z) after every optimiser step. z is the trainable tensor."""

    def __init__(self, pool, dim: int, seed: int):
        g = torch.Generator().manual_seed(seed + 101)
        flat = pool.prompt_len * pool.dim
        self.A = (torch.randn(flat, dim, generator=g) / dim ** 0.5).to(pool.prompts.device)
        self.P0 = pool.prompts.detach().clone()
        self.z = torch.nn.Parameter(torch.zeros(pool.n_slots, dim, device=pool.prompts.device))
        self.pool = pool

    def perturb(self, idx: torch.Tensor, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
        """eps_z (k, d) ~ N(0, sigma^2) and its image in prompt space (k*L, D)."""
        eps_z = torch.randn(len(idx), self.A.shape[1], device=self.A.device) * sigma
        eps = (eps_z @ self.A.T).reshape(-1, self.pool.dim)
        return eps_z, eps

    @torch.no_grad()
    def sync(self) -> None:
        self.pool.prompts.data.copy_(
            self.P0 + (self.z @ self.A.T).reshape(self.pool.n_slots, self.pool.prompt_len, self.pool.dim))


def _forward_reward(model, injector, block: torch.Tensor, x: dict, reward: str):
    """One forward pass with `block` injected; returns (reward, greedy_agree). No graph."""
    with torch.no_grad(), injector as inj:
        inj.set_block(block)
        loss, agree = answer_loss(model, dict(x))
        if inj.n_replaced != block.shape[0]:
            raise RuntimeError(f"replaced {inj.n_replaced} rows, expected {block.shape[0]}")
        inj.n_replaced = 0
    r = float(agree) if reward == "match" else -float(loss)
    return r, agree


def _forward_reward_batch(model, injector, blocks: torch.Tensor, x: dict, reward: str,
                          chunk: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Score B prompt blocks (B, rows, D) on ONE example in batched forwards of at most
    `chunk` rows: the input is repeated along the batch axis and the injector fills the
    marker rows of each batch element with its own block (row-major, which is the order
    `ids == marker` enumerates). Returns (reward (B,), agree (B,)) on CPU. No graph."""
    rs, ags = [], []
    with torch.no_grad(), injector as inj:
        for c0 in range(0, blocks.shape[0], chunk):
            rows = blocks[c0:c0 + chunk]
            xb = repeat_inputs(x, rows.shape[0])
            inj.set_block(rows.reshape(-1, rows.shape[-1]))
            loss, agree = answer_loss(model, xb, per_row=True)
            if inj.n_replaced != rows.shape[0] * rows.shape[1]:
                raise RuntimeError(f"replaced {inj.n_replaced} rows, expected {rows.shape[0] * rows.shape[1]}")
            inj.n_replaced = 0
            rs.append(agree.float().cpu() if reward == "match" else -loss.float().cpu())
            ags.append(agree.float().cpu())
    return torch.cat(rs), torch.cat(ags)


def _prepare(model, processor, pool, task: Task, target: str, fmt: str):
    if pool.n_slots == 1:                      # single prompt: nothing to select, no query
        q = None
        idx = torch.zeros(1, dtype=torch.long, device=pool.keys.device)
        sims = torch.ones(1, device=pool.keys.device)
        pool.last_idx, pool.last_top = idx, idx[0]   # what select() records for the usage counters
    else:
        q = audio_query(model, processor, softpool_input_from_task(task)).to(pool.keys.device)
        idx, sims = pool.select(q)
    conv = build_direct_conversation(task, n_marker=len(idx) * pool.prompt_len, fmt=fmt)
    x = teacher_forced_inputs(model, processor, conv,
                              target_ids(processor, target_text(task, target, fmt)))
    return q, idx, sims, x


class ContrastiveState:
    """EMA selection prototypes per condition for the contrastive key loss."""

    def __init__(self, n_cond: int, n_slots: int, conditions, momentum: float = 0.99):
        self.protos = torch.full((n_cond, n_slots), 1.0 / n_slots)
        self.sim = condition_similarity(conditions)
        self.m = momentum

    def update(self, z: torch.Tensor, cond_idx: int) -> None:
        z = z.detach().float().cpu()
        self.protos[cond_idx] = self.m * self.protos[cond_idx] + (1 - self.m) * z


def _key_loss(pool, q, idx, cond_idx: int, lam: float, usage, lam_div: float,
              contrast: ContrastiveState | None = None, lam_con: float = 1.0):
    """Key objective by pool type. Tagged: the supervised matching loss (a consumer of the
    attack label). Free: pull + diversity (label-free), plus, when `contrast` is set, the
    label-aware contrastive term (slot-free but it reads the condition label)."""
    if pool.n_slots == 1:                                # single shared prompt: nothing to select
        return torch.zeros((), device=pool.keys.device)
    if isinstance(pool, FreePromptPool):
        loss = pool.key_loss(q, idx, usage, lam_div)
        if contrast is not None:
            loss = loss + lam_con * pool.contrastive_key_loss(q, cond_idx, contrast.protos, contrast.sim)
            contrast.update(pool.selection_dist(q), cond_idx)
        return loss
    return lam * pool.matching_loss(q, cond_idx)


def _step(model, processor, pool, injector, task: Task, cond_idx: int, target: str,
          lam: float, train: bool, fmt: str = "direct", objective: str = "grad",
          rl_sigma: float = 0.01, rl_reward: str = "logp", accum: int = 1,
          usage: torch.Tensor | None = None, lam_div: float = 0.0,
          contrast: ContrastiveState | None = None, lam_con: float = 1.0,
          proj: RLProjection | None = None, rl_block: bool = False, block_ctr: list | None = None,
          rl_pairs: int = 1, rl_chunk: int = 16):
    """One example: select with the CURRENT keys, inject, score. `target` is the
    teacher's letter; on an unanswerable task the target becomes the abstain phrase.

    grad: returns the total loss (caller backpropagates).
    rl:   writes the ES estimate straight into pool.prompts.grad for the selected rows
          (scaled by 1/accum) and backpropagates only the key loss; returns None.
    Returns (loss_or_None, answer_term, key_term, agree, match_ok)."""
    q, idx, sims, x = _prepare(model, processor, pool, task, target, fmt)
    if usage is None:
        usage = torch.zeros(pool.n_slots)
    l_key = _key_loss(pool, q, idx, cond_idx, lam, usage, lam_div,
                      contrast if train else None, lam_con)
    match_ok = pool.predicted_condition(sims) == pool.conditions[cond_idx]
    if objective == "grad" or not train:
        block = pool.block(idx)
        if not train:
            block = block.detach()
        with injector as inj:
            inj.set_block(block)
            l_ans, agree = answer_loss(model, x)
            if inj.n_replaced != block.shape[0]:
                raise RuntimeError(f"replaced {inj.n_replaced} rows, expected {block.shape[0]}")
            inj.n_replaced = 0
        if objective == "rl" and not train:            # held-out: report the reward scale
            return None, (float(agree) if rl_reward == "match" else -float(l_ans)), float(l_key), agree, match_ok
        return l_ans + l_key, float(l_ans), float(l_key), agree, match_ok
    # -- rl, training: antithetic pair(s), no graph through the model
    block = pool.block(idx).detach()
    if rl_pairs > 1:
        # P antithetic pairs on the SAME example, scored in batched forwards: the estimate
        # averages P independent directions, so its variance falls by P at ~P/3 the cost of
        # P separate steps (forward-only, no backward). Job 10770089 (P = 1) was flat.
        if proj is None:
            eps_z = None
            eps = torch.randn(rl_pairs, *block.shape, device=block.device) * rl_sigma
        else:
            eps_z = torch.randn(rl_pairs, len(idx), proj.A.shape[1], device=proj.A.device) * rl_sigma
            eps = (eps_z @ proj.A.T).reshape(rl_pairs, -1, pool.dim)
        r, ag = _forward_reward_batch(model, injector, torch.cat([block + eps, block - eps], 0),
                                      x, rl_reward, rl_chunk)
        r_plus, r_minus = r[:rl_pairs].to(block.device), r[rl_pairs:].to(block.device)
        coef = -((r_plus - r_minus) / (2.0 * rl_sigma))            # (P,) loss-gradient sign
        if proj is None:
            g = (coef[:, None, None] * eps).mean(0).reshape(len(idx), pool.prompt_len, pool.dim)
            if pool.prompts.grad is None:
                pool.prompts.grad = torch.zeros_like(pool.prompts)
            pool.prompts.grad[idx] += g.to(pool.prompts.dtype) / accum
        else:
            g = (coef[:, None, None] * eps_z).mean(0)                # (k, d)
            if proj.z.grad is None:
                proj.z.grad = torch.zeros_like(proj.z)
            proj.z.grad[idx] += g / accum
        _backward_keys(l_key, accum)
        return None, float(r.mean()), float(l_key), float(ag.mean()), match_ok
    if proj is None:
        eps_z, eps = None, torch.randn_like(block) * rl_sigma
        if rl_block:
            # block-coordinate ES: perturb ONE of the k*L prompt vectors per sample, cycling,
            # so each estimate lives in D dims (not k*L*D) while the full space stays reachable
            j = block_ctr[0] % block.shape[0]
            block_ctr[0] += 1
            mask = torch.zeros(block.shape[0], 1, device=block.device)
            mask[j] = 1.0
            eps = eps * mask
    else:
        eps_z, eps = proj.perturb(idx, rl_sigma)
    r_plus, ag_plus = _forward_reward(model, injector, block + eps, x, rl_reward)
    r_minus, ag_minus = _forward_reward(model, injector, block - eps, x, rl_reward)
    if proj is None:
        g = es_gradient(r_plus, r_minus, eps, rl_sigma).reshape(len(idx), pool.prompt_len, pool.dim)
        if pool.prompts.grad is None:
            pool.prompts.grad = torch.zeros_like(pool.prompts)
        pool.prompts.grad[idx] += g.to(pool.prompts.dtype) / accum
    else:
        g = es_gradient(r_plus, r_minus, eps_z, rl_sigma)          # (k, d)
        if proj.z.grad is None:
            proj.z.grad = torch.zeros_like(proj.z)
        proj.z.grad[idx] += g / accum
    _backward_keys(l_key, accum)                       # keys only; the model is not in this graph
    return None, 0.5 * (r_plus + r_minus), float(l_key), 0.5 * (float(ag_plus) + float(ag_minus)), match_ok


def discrete_candidates(ids: list[int], j: int, n_vocab: int, n_cands: int, rng: random.Random) -> list[int]:
    """Candidate replacements for slot j: the current token first (so the search is
    monotone), then uniform ordinary vocabulary tokens. BLACK-BOX by construction: no
    embedding neighbours, no gradient, nothing read from the model's weights."""
    cands = [ids[j]]
    while len(cands) < n_cands:
        t = rng.randint(1000, n_vocab - 1)
        if t not in cands:
            cands.append(t)
    return cands


def _forward_reward_tokens(model, marker: int, token_rows: list[list[int]], x: dict, reward: str,
                           chunk: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Score B token prompts on ONE example: each row's marker positions in input_ids are
    REPLACED BY REAL TOKEN IDS (the prompt is text the model reads, not an injected vector),
    then a plain forward. The injector is not involved. Returns (reward (B,), agree (B,))."""
    rs, ags = [], []
    with torch.no_grad():
        for c0 in range(0, len(token_rows), chunk):
            rows = token_rows[c0:c0 + chunk]
            xb = repeat_inputs(x, len(rows))
            ids = xb["input_ids"].clone()
            mask = ids == marker
            for r, toks in enumerate(rows):
                if int(mask[r].sum()) != len(toks):
                    raise RuntimeError(f"{int(mask[r].sum())} marker positions, {len(toks)} tokens")
                ids[r, mask[r]] = torch.tensor(toks, dtype=ids.dtype, device=ids.device)
            xb["input_ids"] = ids
            loss, agree = answer_loss(model, xb, per_row=True)
            rs.append(agree.float().cpu() if reward == "match" else -loss.float().cpu())
            ags.append(agree.float().cpu())
    return torch.cat(rs), torch.cat(ags)


def discrete_search(model, processor, pool, injector, fit_ex, held_ex, targets, a, cfg, log,
                    evaluate_heldout, t0) -> tuple[float, int]:
    """Forward-only coordinate search over REAL tokens for a single-slot pool: each epoch
    visits every slot once, scores --disc-cands replacement tokens on --disc-batch fresh
    training examples in batched forwards, and keeps the best.

    Black-box contract: the search reads only the model's OUTPUTS (target log-prob, or the
    greedy match) for prompts given as token ids; it never backpropagates and never reads
    a weight. The one place the embedding table is touched is `set_pool`, which stores the
    chosen tokens' embeddings in pool.pt so the ordinary eval pipeline (which injects at
    the embedding layer) evaluates the identical prompt; tokens.json is the real artifact.
    Returns (best, best_epoch)."""
    if pool.n_slots != 1:
        raise SystemExit("--objective discrete is defined for --pool-type single")
    embed = model.get_input_embeddings()
    n_vocab = min(embed.num_embeddings, 100_000)
    marker = marker_id(processor)
    g = torch.Generator().manual_seed(a.seed)
    ids = torch.randint(1000, n_vocab, (a.prompt_len,), generator=g).tolist()   # same draw as init_prompts_from_vocab(m=1)
    tok = getattr(processor, "tokenizer", processor)
    rng = random.Random(a.seed)

    def set_pool(id_list):                      # export only: embeddings of the chosen tokens
        with torch.no_grad():
            W = embed.weight.detach()
            pool.prompts.data.copy_(W[torch.tensor(id_list, device=W.device)].to(pool.prompts.dtype).unsqueeze(0))

    def dump_tokens(epoch, held):
        (a.out / "tokens.json").write_text(json.dumps(
            {"epoch": epoch, "ids": ids, "tokens": [tok.decode([i]) for i in ids],
             "text": tok.decode(ids), "heldout": held}, indent=2))

    model.eval()
    best, best_epoch = -1.0, -1
    for epoch in range(a.epochs):
        order = list(range(a.prompt_len))
        rng.shuffle(order)
        for j in order:
            cands = discrete_candidates(ids, j, n_vocab, a.disc_cands, rng)
            rows = [ids[:j] + [c] + ids[j + 1:] for c in cands]
            R = torch.zeros(len(cands))
            batch = rng.sample(fit_ex, min(a.disc_batch, len(fit_ex)))
            for t, ci in batch:
                _, _, _, x = _prepare(model, processor, pool, t, targets[t.id], a.format)
                r, _ = _forward_reward_tokens(model, marker, rows, x, a.rl_reward, a.rl_chunk)
                R += r
            R /= len(batch)
            k = int(R.argmax())
            changed = cands[k] != ids[j]
            if changed:
                ids[j] = cands[k]
            rec = {"epoch": epoch, "slot": j, "reward_cur": round(float(R[0]), 4),
                   "reward_best": round(float(R[k]), 4), "changed": changed,
                   "token": tok.decode([ids[j]]), "elapsed_s": round(time.time() - t0, 1),
                   "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                   if torch.cuda.is_available() else None}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
        set_pool(ids)
        held = evaluate_heldout() if held_ex else {"agree": None, "match": None, "n": 0}
        rec = {"epoch": epoch, "heldout": held, "text": tok.decode(ids)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
        pool.save(a.out / "pool_last.pt", {"epoch": epoch, "heldout": held, "config": cfg, "ids": ids})
        score = held["agree"] if held["agree"] is not None else 0.0
        if score > best:
            best, best_epoch = score, epoch
            pool.save(a.out / "pool.pt", {"epoch": epoch, "heldout": held, "config": cfg, "ids": ids})
            dump_tokens(epoch, held)
    return best, best_epoch


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attacks", nargs="+", default=[c for c in FULL_CONDITIONS if c != "clean"],
                    help="default: every paper noise/mask level plus adv_wrong and text_inject")
    ap.add_argument("--format", default="direct_abstain", choices=sorted(FORMATS),
                    help="direct_abstain makes declining representable; required when the "
                         "attack list holds unanswerable conditions (mask_100, noise_-20dB)")
    ap.add_argument("--train-limit", type=int, default=8, help="train tasks per track")
    ap.add_argument("--train-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--slots-per-condition", type=int, default=4)
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-match", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=8, help="examples per optimizer step")
    ap.add_argument("--heldout-frac", type=float, default=0.1, help="train wavs kept for model selection")
    # FEW-SHOT ADAPTATION to a transfer benchmark, label-free: K items per track from the
    # benchmark's TRAIN side (same split as run_softpool evaluates the DEV side of) join
    # the training mixture. Their gold is never read -- the target is still the model's
    # own clean answer -- so this uses only unlabeled target-domain audio.
    ap.add_argument("--fewshot", nargs="*", default=[], metavar="BENCH:K",
                    help="e.g. mmau:20 mmar:20 -- K train items per track from that benchmark")
    ap.add_argument("--init", type=Path, default=None, metavar="POOL.PT",
                    help="fine-tune from a saved pool instead of a fresh vocab init; its "
                         "conditions/slots/prompt_len must match this run's")
    ap.add_argument("--objective", default="grad", choices=("grad", "rl", "discrete"),
                    help="grad: backward through the frozen model; rl: antithetic ES on the "
                         "selected prompts from a forward-only reward (no backward); discrete: "
                         "the block is --prompt-len REAL vocabulary tokens chosen by a forward-only "
                         "coordinate search on the same reward (single pool only)")
    ap.add_argument("--rl-pairs", type=int, default=1,
                    help="rl: antithetic pairs per example, scored in batched forwards (1 = the "
                         "original one pair per example)")
    ap.add_argument("--rl-chunk", type=int, default=16, help="rows per batched forward (rl-pairs, discrete)")
    ap.add_argument("--disc-cands", type=int, default=32,
                    help="discrete: candidate tokens per slot update (current token + neighbours + random)")
    ap.add_argument("--disc-batch", type=int, default=64,
                    help="discrete: training examples each candidate set is scored on")

    ap.add_argument("--rl-sigma", type=float, default=0.002,
                    help="policy std per coordinate. Prompt rows have norm ~1.2 and std ~0.02; "
                         "0.002 perturbs a row by ~0.12 (10%%). 0.01 (50%%) wrecked the pool in "
                         "the first smoke (held-out agreement 18%% after 17 steps)")
    ap.add_argument("--rl-lr", type=float, default=1e-4,
                    help="Adam lr for the PROMPTS under rl (keys keep --lr). Adam normalises the "
                         "noisy ES estimate into full-size steps, so the prompt lr bounds the "
                         "random walk: 1e-4 over ~4000 steps drifts ~0.006 per coordinate")
    ap.add_argument("--rl-block", action="store_true",
                    help="rl in the FULL prompt space, one prompt vector (3584-d) perturbed per "
                         "sample, cycling through the block; incompatible with --rl-dim")
    ap.add_argument("--rl-dim", type=int, default=0,
                    help="rl: perturb and learn in a shared random d-dim subspace per slot "
                         "(black-box prompt tuning) instead of the full 8x3584; 0 = full")
    ap.add_argument("--rl-reward", default="logp", choices=("logp", "match"),
                    help="logp: mean log-prob of the target (dense); match: greedy == target (0/1)")
    ap.add_argument("--pool-type", default="tagged", choices=("tagged", "free", "single"),
                    help="tagged: 4 slots per condition + matching loss; free: --n-slots untagged "
                         "slots, pull + diversity loss, attack label never consumed; single: ONE "
                         "shared prompt of --prompt-len vectors, no keys, no selection (the "
                         "ablation that asks whether choosing from the audio adds anything)")
    ap.add_argument("--n-slots", type=int, default=0,
                    help="free pool size (0 = len(conditions) x slots-per-condition, same params)")
    ap.add_argument("--lambda-div", type=float, default=1.0, help="free pool diversity weight")
    ap.add_argument("--key-loss", default="pull", choices=("pull", "contrastive"),
                    help="free pool only. pull: L2P pull + diversity, label-free. contrastive: "
                         "adds a term that pulls the selection toward conditions of similar "
                         "corruption level (noise_0dB near noise_-10dB, far from clean and "
                         "noise_-20dB) and away from dissimilar ones; reads the condition "
                         "label but tags no slot")
    ap.add_argument("--lambda-con", type=float, default=1.0, help="contrastive weight")
    ap.add_argument("--model", default=None, help="qwen2.5-omni (default) | qwen3-omni | af3")
    ap.add_argument("--abstain-anyway", action="store_true",
                    help="teacher-force the abstain phrase on unanswerable examples even in a format "
                         "that never offers it (AF3 native): the soft prompt has to TEACH the decline, "
                         "and the parser scores CANNOT DETERMINE as abstain in every format")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("output/softpool"))
    ap.add_argument("--dry-run", action="store_true", help="print the resolved config and exit")
    a = ap.parse_args(argv)

    conditions = ("clean", *a.attacks)
    from src.expel.attacks import UNANSWERABLE
    from src.expel.softpool import ABSTAIN_FORMATS
    teaches_abstain = a.format in ABSTAIN_FORMATS or a.abstain_anyway
    if any(c in UNANSWERABLE for c in a.attacks) and not teaches_abstain:
        raise SystemExit(f"unanswerable attacks need --format in {ABSTAIN_FORMATS} (or --abstain-anyway): "
                         "the target is to DECLINE, which the plain direct format cannot express")
    if a.key_loss == "contrastive" and a.pool_type != "free":
        raise SystemExit("--key-loss contrastive is defined for --pool-type free only")
    if a.rl_block and a.rl_dim > 0:
        raise SystemExit("--rl-block and --rl-dim are alternatives; give one")
    if a.prompt_len < 1:
        raise SystemExit("--prompt-len must be >= 1: a 0-length block injects nothing and "
                         "trains nothing while exiting 0")
    if a.pool_type == "single":
        a.n_slots, a.top_k = 1, 1            # before the config snapshot, so the report is truthful
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    cfg["conditions"] = list(conditions)
    print(json.dumps(cfg, indent=2), flush=True)
    if a.dry_run:
        return

    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    random.seed(a.seed)

    train_all, _ = split_train_dev(load_sakura(), a.train_frac, a.seed)
    train_tasks = take(train_all, a.train_limit, a.seed)
    fewshot_n = {}
    for spec in a.fewshot:
        bench, k = spec.split(":")
        from src.expel.benchmarks import load_benchmark
        b_train, _ = split_train_dev(load_benchmark(bench), a.train_frac, a.seed)
        extra = take(b_train, int(k), a.seed)
        fewshot_n[bench] = len(extra)
        train_tasks = train_tasks + extra
    if fewshot_n:
        print(f"few-shot: {fewshot_n} train items added (label-free)", flush=True)
        cfg["fewshot_n"] = fewshot_n
    # Held-out slice BY WAV for model selection, never touched by the optimizer.
    wavs = sorted({wav_id(t) for t in train_tasks})
    rng = random.Random(a.seed + 11)
    rng.shuffle(wavs)
    n_held = max(1, round(len(wavs) * a.heldout_frac)) if len(wavs) > 1 else 0
    held = set(wavs[:n_held])
    fit_tasks = [t for t in train_tasks if wav_id(t) not in held]
    held_tasks = [t for t in train_tasks if wav_id(t) in held]
    print(f"train tasks {len(train_tasks)}: fit {len(fit_tasks)}, held-out {len(held_tasks)}", flush=True)

    from src.model import env_report, load_model
    print(json.dumps(env_report(), indent=2), flush=True)
    model, processor = load_model(a.model)
    actor = SoftPoolActor(model, processor, top_k=a.top_k, fmt=a.format,
                          max_new_tokens=12 if teaches_abstain else 8)

    # 1. teacher pass -- clean items, no soft prompts, model.eval()
    model.eval()
    targets = teacher_targets(actor, train_tasks, a.out)

    # 2. examples
    fit_ex, skipped = build_examples(fit_tasks, conditions, a.seed)
    held_ex, _ = build_examples(held_tasks, conditions, a.seed)
    fit_ex = [(t, ci) for t, ci in fit_ex if t.id in targets]
    held_ex = [(t, ci) for t, ci in held_ex if t.id in targets]
    print(f"examples: fit {len(fit_ex)}, held-out {len(held_ex)}, skipped {skipped}", flush=True)
    if not fit_ex:
        raise SystemExit("no training examples (teacher unparseable or attacks unrenderable)")

    # 3. pool
    m = a.n_slots or len(conditions) * a.slots_per_condition
    if a.init is not None:
        if a.pool_type != "tagged":
            raise SystemExit("--init is only defined for tagged pools (layout check is by condition)")
        pool = load_init_pool(a.init, conditions, a.slots_per_condition, a.prompt_len)
        pool.top_k = a.top_k
        pool = pool.to(model.device)
        print(f"init: fine-tuning from {a.init}", flush=True)
    else:
        init = init_prompts_from_vocab(model.get_input_embeddings(), m, a.prompt_len, a.seed)
        if a.pool_type in ("free", "single"):
            pool = FreePromptPool(m, conditions, a.prompt_len, init.shape[-1], a.top_k,
                                  init_prompts=init, seed=a.seed).to(model.device)
        else:
            pool = SoftPromptPool(conditions, a.slots_per_condition, a.prompt_len, init.shape[-1],
                                  a.top_k, init_prompts=init, seed=a.seed).to(model.device)
    injector = SoftPromptInjector(model.get_input_embeddings(), marker_id(processor))
    proj = RLProjection(pool, a.rl_dim, a.seed) if (a.objective == "rl" and a.rl_dim > 0) else None
    block_ctr = [0]
    prompt_param = proj.z if proj is not None else pool.prompts
    groups = [{"params": [pool.keys], "lr": a.lr},
              {"params": [prompt_param], "lr": a.rl_lr if a.objective == "rl" else a.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    if proj is not None:
        pool.prompts.requires_grad_(False)
        print(f"rl: subspace dim {a.rl_dim} per slot ({pool.n_slots * a.rl_dim} ES coordinates)", flush=True)
    steps_total = max(1, math.ceil(len(fit_ex) / a.accum) * a.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_total)

    if a.objective == "grad":
        # Gradients must reach the INPUT embeddings through a frozen model: non-reentrant
        # checkpointing, and train() because HF only checkpoints when self.training is True.
        # Qwen2 has zero dropout, so train() changes nothing else.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    n_trainable = sum(p.numel() for p in pool.parameters()) if proj is None else (pool.keys.numel() + proj.z.numel())
    n_frozen_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_frozen_grad == 0, "base model must stay frozen"
    print(f"pool params {n_trainable}, slots {m}, block {a.top_k * a.prompt_len} tokens, "
          f"objective {a.objective}, pool {a.pool_type}", flush=True)
    contrast = (ContrastiveState(len(conditions), m, conditions)
                if a.key_loss == "contrastive" else None)
    step_kw = dict(objective=a.objective, rl_sigma=a.rl_sigma, rl_reward=a.rl_reward,
                   accum=a.accum, lam_div=a.lambda_div, contrast=contrast, lam_con=a.lambda_con,
                   proj=proj, rl_block=a.rl_block, block_ctr=block_ctr,
                   rl_pairs=a.rl_pairs, rl_chunk=a.rl_chunk)
    usage = torch.zeros(m)                       # per-epoch top-k selection counts
    hist = torch.zeros(m, len(conditions))       # top-slot x true condition, per epoch

    def evaluate_heldout() -> dict:
        model.eval()
        agree = match = 0
        with torch.no_grad():
            for t, ci in held_ex:
                _, _, _, ag, mo = _step(model, processor, pool, injector, t, ci, targets[t.id],
                                        a.lambda_match, train=False, fmt=a.format, **step_kw)
                agree += ag
                match += mo
        n = max(1, len(held_ex))
        return {"agree": round(100.0 * agree / n, 2), "match": round(100.0 * match / n, 2), "n": len(held_ex)}

    log = open(a.out / "train_log.jsonl", "w")
    best, best_epoch, step, t0 = -1.0, -1, 0, time.time()
    if a.objective == "discrete":
        best, best_epoch = discrete_search(model, processor, pool, injector, fit_ex, held_ex, targets,
                                           a, cfg, log, evaluate_heldout, t0)
        log.close()
        n_trainable = 0
        report = {"config": cfg, "objective": a.objective, "pool_type": a.pool_type, "key_loss": a.key_loss,
                  "n_slots": m, "n_train_tasks": len(train_tasks), "fewshot": fewshot_n,
                  "n_fit_examples": len(fit_ex), "n_heldout_examples": len(held_ex), "skipped": skipped,
                  "best_epoch": best_epoch, "best_heldout_agree": best, "pool_params": 0,
                  "elapsed_s": round(time.time() - t0, 1),
                  "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                  if torch.cuda.is_available() else None}
        (a.out / "train_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return
    for epoch in range(a.epochs):
        order = list(range(len(fit_ex)))
        random.Random(a.seed + epoch).shuffle(order)
        model.train() if a.objective == "grad" else model.eval()
        run = {"loss": 0.0, "ans": 0.0, "match": 0.0, "agree": 0, "match_ok": 0, "n": 0}
        usage.zero_(); hist.zero_()
        opt.zero_grad(set_to_none=True)
        for j, i in enumerate(order):
            t, ci = fit_ex[i]
            loss, l_ans, l_match, ag, mo = _step(model, processor, pool, injector, t, ci,
                                                 targets[t.id], a.lambda_match, train=True,
                                                 fmt=a.format, usage=usage, **step_kw)
            sel = pool.last_idx
            usage[sel.cpu()] += 1
            hist[int(pool.last_top.cpu()), ci] += 1
            if loss is not None:
                (loss / a.accum).backward()
                run["loss"] += float(loss)
            else:
                run["loss"] += -l_ans                 # rl: -reward, so the column still falls
            run["ans"] += l_ans; run["match"] += l_match
            run["agree"] += ag; run["match_ok"] += mo; run["n"] += 1
            if (j + 1) % a.accum == 0 or j + 1 == len(order):
                torch.nn.utils.clip_grad_norm_([pool.keys, prompt_param], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if proj is not None:
                    proj.sync()
            if run["n"] % 50 == 0 or j + 1 == len(order):
                n = run["n"]
                ent = _usage_entropy(usage)
                rec = {"epoch": epoch, "step": step, "seen": n, "loss": round(run["loss"] / n, 4),
                       "answer_loss" if a.objective == "grad" else "reward": round(run["ans"] / n, 4),
                       "key_loss": round(run["match"] / n, 4), "usage_entropy": ent,
                       "agree_pct": round(100.0 * run["agree"] / n, 2),
                       "match_pct": round(100.0 * run["match_ok"] / n, 2),
                       "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1),
                       "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                       if torch.cuda.is_available() else None}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
        if isinstance(pool, FreePromptPool):
            pool.set_posthoc_tags(hist)              # analysis-only tags from this epoch's usage
        held = evaluate_heldout() if held_ex else {"agree": None, "match": None, "n": 0}
        held["usage_entropy"] = _usage_entropy(usage)
        held["slots_used"] = int((usage > 0).sum())
        rec = {"epoch": epoch, "heldout": held}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
        pool.save(a.out / "pool_last.pt", {"epoch": epoch, "heldout": held, "config": cfg})
        score = held["agree"] if held["agree"] is not None else run["agree"] / max(1, run["n"])
        if score > best:
            best, best_epoch = score, epoch
            pool.save(a.out / "pool.pt", {"epoch": epoch, "heldout": held, "config": cfg})
    log.close()
    report = {"config": cfg, "objective": a.objective, "pool_type": a.pool_type, "key_loss": a.key_loss,
              "n_slots": m, "n_train_tasks": len(train_tasks), "fewshot": fewshot_n,
              "n_fit_examples": len(fit_ex),
              "n_heldout_examples": len(held_ex), "skipped": skipped, "best_epoch": best_epoch,
              "best_heldout_agree": best, "pool_params": n_trainable,
              "elapsed_s": round(time.time() - t0, 1),
              "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
              if torch.cuda.is_available() else None}
    (a.out / "train_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
