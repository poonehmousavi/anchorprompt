"""Approach 11 -- a pool of SOFT prompts for the frozen LALM, selected by the audio.

L2P / CODA-Prompt shape: a pool of continuous prompt vectors with one learned KEY per
slot. At inference the model's own audio encoder produces a query, the top-k keys by
cosine are chosen, and their prompts are injected into the thinker's input embeddings.
The LALM is never updated -- only `prompts` and `keys` train.

What this is and is not:
  * weights frozen  -- yes, every base parameter has requires_grad=False.
  * label-free      -- yes, the training target is the model's OWN answer on the clean
                       twin, never the gold letter (see train_softpool.py).
  * gradient-free   -- NO. Backward runs through the frozen model into the prompt
                       parameters. Say so wherever the method is described.

The design invariant survives unchanged: selection consumes a `SoftPoolInput` that holds
the audio path and nothing else, mirroring `RouterInput` in src/checker.py. The attack
label is read in exactly one place, the matching loss in train_softpool.py, at training
time. `tests/test_softpool.py` pins the shape.

Injection mechanics: the prompt block is rendered into the user turn as N copies of a
MARKER token that our conversations never otherwise contain. A forward hook on the
thinker's token-embedding layer replaces those rows with the selected prompt vectors.
The thinker scatters audio features into the audio placeholders AFTER embedding
(transformers' modeling_qwen2_5_omni.py, forward step 2), so the two never touch.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.expel.types import Task, Trajectory
from src.parsing import parse_outcome

# A single token in the Qwen2.5-Omni tokenizer (id 151650), repeats cleanly, and
# nothing in SAKURA/MMAU/MMAR text contains it. Verified by test_marker_tokenises_cleanly.
MARKER = "<|quad_start|>"

# The training mixture and the slot assignment. Order is the condition index.
# The original 5-condition pool (kept for the pool-class default and the first results).
CONDITIONS = ("clean", "adv_wrong", "mask_60", "noise_0dB", "text_inject")
# Every noise SNR and mask ratio from the faithfulness paper (arXiv:2509.22363 Sec. 3).
PAPER_LEVELS = ("noise_20dB", "noise_10dB", "noise_0dB", "noise_-10dB", "noise_-20dB",
                "mask_20", "mask_40", "mask_60", "mask_80", "mask_100")
# Default training/eval set: all paper levels plus the two injection attacks. Includes the
# two unanswerable levels, so it needs the `direct_abstain` format.
FULL_CONDITIONS = ("clean", *PAPER_LEVELS, "adv_wrong", "text_inject")

HIDDEN = 3584   # Qwen2.5-Omni-7B thinker hidden size; the audio tower projects into it

# Direct-answer format for this arm. Teacher forcing needs a differentiable target, so
# the chain-of-thought block is dropped -- for BOTH arms, so the comparison stays within
# one format. The CoT `none` numbers in report.md are quoted next to these for the
# format cost.
SYSTEM_DIRECT = ("You are an expert at reasoning about sounds, speech, and the things that "
                 "produce them. Answer multiple-choice questions about audio.\n\n"
                 "Respond with only the letter of your choice in parentheses, for "
                 "example (b).")

# Same format with declining made REPRESENTABLE. Applied to BOTH arms, like the abstain
# variants in lalm.py, so the floor moves equally. `ABSTAIN_TARGET` is the teacher-forced
# target on unanswerable examples (mask_100, noise_-20dB): the faithful answer is to
# decline, and `parse_outcome` recognises the phrase.
ABSTAIN_TARGET = "CANNOT DETERMINE"
SYSTEM_DIRECT_ABSTAIN = SYSTEM_DIRECT + (
    f"\nIf the recording does not contain enough information to answer, respond with "
    f"exactly: {ABSTAIN_TARGET}")
# Audio Flamingo 3's OWN benchmark layout (model card): no system prompt, the question
# followed by "Choose the correct option among the options below: (A) .. (B) ..", answered
# as "(A) Sad to happy". `af3_think` appends the AF-Think trigger and needs the `think` LoRA
# adapter (`--model af3-think`). Used to check the clean base in the format the model was
# evaluated with, not for training.
# Qwen2.5-Omni / Qwen3-Omni publish MMAU/MMAR numbers but no prompt (Qwen3-Omni README lists
# default prompts for ASR/S2TT/lyrics only; MMAU's repo ships a scorer, not a template).
# `qwen_native` is the lmms-eval MMAU layout -- question, one option per line, "Answer with
# the option's letter from the given choices directly." -- no system turn (the chat template
# supplies Qwen's default one). It is the clean-base check against the papers, like af3_native.
_AF3_NATIVE = "{stem} Choose the correct option among the options below: {opts_inline}"
_ABSTAIN_CLAUSE = f" If the recording does not contain enough information to answer, respond with exactly: {ABSTAIN_TARGET}"
NATIVE_FORMATS = {"af3_native": _AF3_NATIVE,
                  # AF3's own layout + our decline clause (user decision 2026-09-11): the
                  # reported AF3 format. Same sentence both arms, so abstention is scorable.
                  "af3_native_abstain": _AF3_NATIVE + _ABSTAIN_CLAUSE,
                  "af3_think": _AF3_NATIVE + " Please think and reason about the input audio before you respond.",
                  "qwen_native": "{stem}\n{opts_lines}\nAnswer with the option's letter from the given choices directly."}
# Formats in which CANNOT DETERMINE is an offered answer (needed to train/score declining).
ABSTAIN_FORMATS = ("direct_abstain", "af3_native_abstain")
FORMATS = {"direct": SYSTEM_DIRECT, "direct_abstain": SYSTEM_DIRECT_ABSTAIN,
           **{k: None for k in NATIVE_FORMATS}}


def native_question(task, fmt: str) -> str:
    items = [(L.upper(), t) for L, t in sorted(task.choices.items())]
    return NATIVE_FORMATS[fmt].format(stem=task.stem,
                                      opts_inline=" ".join(f"({L}) {t}" for L, t in items),
                                      opts_lines="\n".join(f"{L}. {t}" for L, t in items))

CACHE_ROOT = Path(os.environ.get("REPROMPT_CACHE", "data/_cache"))


# --------------------------------------------------------------------------- legality

@dataclass(frozen=True)
class SoftPoolInput:
    """Everything the selector may see. One field, by construction."""
    audio_path: str


def softpool_input_from_task(task: Task) -> SoftPoolInput:
    """The ONLY projection from a Task to the selector's input. Explicit field pick, never
    `**task.__dict__`, so a new Task field cannot leak in by accident."""
    if not isinstance(task, Task):
        raise TypeError(f"softpool_input_from_task takes a Task, got {type(task).__name__}")
    return SoftPoolInput(audio_path=task.audio_path)


def _require_input(inp) -> SoftPoolInput:
    if not isinstance(inp, SoftPoolInput):
        raise TypeError("the soft-pool selector accepts a SoftPoolInput only "
                        f"(got {type(inp).__name__}); project with softpool_input_from_task")
    return inp


# --------------------------------------------------------------------------- pool

class SoftPromptPool(nn.Module):
    """`prompts (M, L, D)`, `keys (M, D)`, `slot_condition (M,)`."""

    pool_type = "tagged"

    def __init__(self, conditions=CONDITIONS, slots_per_condition: int = 4,
                 prompt_len: int = 8, dim: int = HIDDEN, top_k: int = 3, tau: float = 0.1,
                 init_prompts: torch.Tensor | None = None, seed: int = 0,
                 n_slots: int | None = None):
        super().__init__()
        self.conditions = tuple(conditions)
        self.slots_per_condition = int(slots_per_condition)
        self.prompt_len, self.dim, self.top_k, self.tau = int(prompt_len), int(dim), int(top_k), float(tau)
        if n_slots is None:
            m = len(self.conditions) * self.slots_per_condition
            tags = [c for c in range(len(self.conditions)) for _ in range(self.slots_per_condition)]
        else:                                   # free pool: no tags at creation (post-hoc only)
            m = int(n_slots)
            tags = [0] * m
        g = torch.Generator().manual_seed(seed)
        if init_prompts is None:
            init_prompts = torch.randn(m, self.prompt_len, self.dim, generator=g) * 0.02
        assert tuple(init_prompts.shape) == (m, self.prompt_len, self.dim), init_prompts.shape
        self.prompts = nn.Parameter(init_prompts.clone().float())
        self.keys = nn.Parameter(F.normalize(torch.randn(m, self.dim, generator=g), dim=-1))
        self.register_buffer("slot_condition", torch.tensor(tags))

    # -- selection ---------------------------------------------------------------
    @property
    def n_slots(self) -> int:
        return self.keys.shape[0]

    def similarities(self, query: torch.Tensor) -> torch.Tensor:
        q = F.normalize(query.to(self.keys.dtype).reshape(-1), dim=-1)
        return F.normalize(self.keys, dim=-1) @ q           # (M,)

    def select(self, query: torch.Tensor, top_k: int | None = None):
        """Top-k slots by cosine, returned in ascending slot order so the block is
        order-invariant. Returns (idx LongTensor(k), sims Tensor(M))."""
        k = int(top_k or self.top_k)
        sims = self.similarities(query)
        top = torch.topk(sims, k=min(k, self.n_slots)).indices
        idx = top.sort().values
        self.last_idx, self.last_top = idx.detach(), top[0].detach()   # for usage counters
        return idx, sims

    def block(self, idx: torch.Tensor) -> torch.Tensor:
        """(k*L, D) prompt block for the selected slots."""
        return self.prompts[idx].reshape(-1, self.dim)

    def condition_logits(self, sims: torch.Tensor) -> torch.Tensor:
        """Per-condition score = max over that condition's slots, temperature-scaled."""
        c = len(self.conditions)
        out = torch.full((c,), -1e4, dtype=sims.dtype, device=sims.device)
        return out.scatter_reduce(0, self.slot_condition, sims / self.tau, reduce="amax",
                                  include_self=True)

    def matching_loss(self, query: torch.Tensor, cond_idx: int) -> torch.Tensor:
        """Supervised key matching: cross-entropy over conditions given the query.
        The ONLY consumer of the attack label, and training-side only."""
        logits = self.condition_logits(self.similarities(query))
        target = torch.tensor(int(cond_idx), device=logits.device)
        return F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))

    def predicted_condition(self, sims: torch.Tensor) -> str:
        return self.conditions[int(self.condition_logits(sims).argmax())]

    # -- persistence -------------------------------------------------------------
    def hyper(self) -> dict:
        return {"pool_type": self.pool_type, "conditions": list(self.conditions),
                "slots_per_condition": self.slots_per_condition, "n_slots": self.n_slots,
                "prompt_len": self.prompt_len, "dim": self.dim, "top_k": self.top_k,
                "tau": self.tau, "marker": MARKER}

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"hyper": self.hyper(), "state": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str | Path, map_location="cpu") -> "SoftPromptPool":
        """Dispatches on the saved `pool_type`, so a free pool loads as FreePromptPool
        whichever class `load` was called on (older checkpoints have no field: tagged)."""
        ck = torch.load(path, map_location=map_location)
        h = ck["hyper"]
        if h.get("pool_type", "tagged") == "free":
            pool = FreePromptPool(n_slots=h["n_slots"], conditions=h["conditions"],
                                  prompt_len=h["prompt_len"], dim=h["dim"], top_k=h["top_k"],
                                  tau=h["tau"])
        else:
            pool = SoftPromptPool(conditions=h["conditions"], slots_per_condition=h["slots_per_condition"],
                                  prompt_len=h["prompt_len"], dim=h["dim"], top_k=h["top_k"], tau=h["tau"])
        pool.load_state_dict(ck["state"])
        pool.loaded_extra = ck.get("extra", {})
        return pool


class FreePromptPool(SoftPromptPool):
    """UNTAGGED pool (plan arm T4 + T1): M slots with no condition assignment and no
    matching loss, so the attack label is not consumed by the pool at all. Keys learn
    from the L2P pull loss on the selected slots plus a DIVERSITY penalty that pushes
    over-used keys away from the queries that keep selecting them (usage counted per
    epoch, reset each epoch).

    `slot_condition` is filled POST-HOC from a usage histogram (slot x condition over
    the last training epoch) so that the evaluation's route/recall/false-alarm report
    still has something to say; those tags are analysis-only and never used to train.
    """
    pool_type = "free"

    def __init__(self, n_slots: int, conditions=FULL_CONDITIONS, prompt_len: int = 8,
                 dim: int = HIDDEN, top_k: int = 3, tau: float = 0.1,
                 init_prompts: torch.Tensor | None = None, seed: int = 0):
        super().__init__(conditions=conditions, slots_per_condition=0, prompt_len=prompt_len,
                         dim=dim, top_k=top_k, tau=tau, init_prompts=init_prompts, seed=seed,
                         n_slots=int(n_slots))
        self.register_buffer("usage_hist", torch.zeros(int(n_slots), len(self.conditions)))

    def matching_loss(self, query, cond_idx):
        raise RuntimeError("a free pool has no condition tags; use key_loss (label-free)")

    def key_loss(self, query: torch.Tensor, idx: torch.Tensor, usage: torch.Tensor,
                 lam_div: float) -> torch.Tensor:
        """pull = mean(1 - cos) over the selected slots (L2P);
        diversity = mean(max(0, f_i - 1) * cos_i) where f_i = usage share x M (1 = uniform),
        so only slots selected MORE than their fair share are pushed away, in proportion.
        `usage` is the per-epoch selection count (detached)."""
        cos = self.similarities(query)[idx]
        share = usage.to(cos.device).float()
        share = share / share.sum().clamp(min=1.0) * self.n_slots
        over = torch.clamp(share[idx] - 1.0, min=0.0).detach()
        pull = (1.0 - cos).mean()
        div = (over * cos).mean()
        return pull + float(lam_div) * div

    # -- contrastive key loss (label-aware, slot-free) ------------------------------
    def selection_dist(self, query: torch.Tensor) -> torch.Tensor:
        """Soft selection over slots, softmax(cos / tau). Differentiable in the keys."""
        return F.softmax(self.similarities(query) / self.tau, dim=0)

    def contrastive_key_loss(self, query: torch.Tensor, cond_idx: int, protos: torch.Tensor,
                             sim: torch.Tensor) -> torch.Tensor:
        """Pull this query's selection toward the selections of SIMILAR conditions and
        away from dissimilar ones, without tagging any slot.

        z = softmax(cos / tau) over slots; protos (C, M) are EMA selection distributions
        per condition (detached); <z, proto_c> is the chance of co-selecting a slot with
        condition c. sim (C, C) in [0, 1] is the condition similarity (condition_similarity):
        loss = -log( sum_c sim[true, c] <z, proto_c>  /  sum_c <z, proto_c> ).
        Consumes the condition label (like matching_loss) but assigns no tag to any slot."""
        z = self.selection_dist(query)
        co = protos.to(z.device) @ z                          # (C,)
        w = sim[int(cond_idx)].to(z.device)
        return -torch.log((w * co).sum() / co.sum().clamp(min=1e-8) + 1e-8)

    def set_posthoc_tags(self, hist: torch.Tensor) -> None:
        """hist (M, C): how often each slot was the TOP slot under each condition. Tag =
        argmax; slots never selected keep tag 0 and are reported as such."""
        assert tuple(hist.shape) == tuple(self.usage_hist.shape), hist.shape
        self.usage_hist.copy_(hist.to(self.usage_hist.device))
        self.slot_condition.copy_(hist.argmax(dim=1).to(self.slot_condition.device))


def condition_level(cond: str) -> tuple[str, float]:
    """(family, level in [0, 1]) for the ordinal structure over conditions. Level 0 = the
    audio is untouched (clean, text_inject); 1 = nothing left to hear (mask_100,
    noise_-20dB). Noise SNR 20/10/0/-10/-20 dB -> 0.2/0.4/0.6/0.8/1.0; mask p% -> p/100."""
    if cond in ("clean", "text_inject"):
        return "none", 0.0
    if cond.startswith("noise_"):
        snr = float(cond[len("noise_"):-2])
        return "noise", round((30.0 - snr) / 50.0, 3)
    if cond.startswith("mask_"):
        return "mask", int(cond[len("mask_"):]) / 100.0
    if cond.startswith("adv_"):
        return "inject", 0.5
    raise KeyError(f"no level defined for condition {cond!r}")


def condition_similarity(conditions, width: float = 0.25, family_gap: float = 0.3,
                         target_gap: float = 0.3) -> torch.Tensor:
    """(C, C) in [0, 1]: exp(-(d / width)^2) with d = |level_a - level_b|, plus `family_gap`
    when both are corrupted in different ways (noise vs mask), plus `target_gap` across
    the answer/decline boundary (level 1.0 = unanswerable, whose target is to decline).
    Diagonal 1. So noise_0dB is near noise_-10dB (0.53), far from clean (0.003), and
    noise_-10dB is far from noise_-20dB (0.02) although only one level apart, because
    the two are trained toward different behaviours."""
    lv = [condition_level(c) for c in conditions]
    n = len(lv)
    out = torch.eye(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            (fa, la), (fb, lb) = lv[i], lv[j]
            d = abs(la - lb)
            if fa != fb and la > 0 and lb > 0:
                d += family_gap
            if (la >= 1.0) != (lb >= 1.0):
                d += target_gap
            out[i, j] = float(torch.exp(torch.tensor(-(d / width) ** 2)))
    return out


def init_prompts_from_vocab(embed: nn.Embedding, m: int, prompt_len: int, seed: int = 0) -> torch.Tensor:
    """Initialise prompts from the embeddings of random ordinary tokens (P-tuning trick):
    keeps the block on the manifold the thinker expects, so early steps do not derail it."""
    g = torch.Generator().manual_seed(seed)
    n_vocab = min(embed.num_embeddings, 100_000)   # avoid the special-token tail
    ids = torch.randint(1000, n_vocab, (m * prompt_len,), generator=g)
    with torch.no_grad():
        vec = embed.weight[ids.to(embed.weight.device)].detach().float().cpu()
    return vec.reshape(m, prompt_len, -1)


# --------------------------------------------------------------------------- injection

def marker_id(processor) -> int:
    tok = getattr(processor, "tokenizer", processor)
    ids = tok(MARKER, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise RuntimeError(f"MARKER {MARKER!r} must be a single token, tokenised to {ids}")
    return int(ids[0])


class SoftPromptInjector:
    """Context manager: while active, rows of the token-embedding output at MARKER
    positions are replaced by `block`. Gradient flows into `block`."""

    def __init__(self, embed: nn.Module, marker: int):
        self.embed, self.marker = embed, int(marker)
        self.block: torch.Tensor | None = None
        self.n_replaced = 0
        self._handle = None

    def set_block(self, block: torch.Tensor | None) -> None:
        self.block = block

    def _hook(self, module, args, output):
        ids = args[0]
        mask = ids == self.marker
        n = int(mask.sum())
        if n == 0:
            return output
        if self.block is None:
            raise RuntimeError("MARKER tokens present but no soft-prompt block was set")
        if n != self.block.shape[0]:
            raise RuntimeError(f"{n} marker tokens but block has {self.block.shape[0]} rows")
        full = torch.zeros_like(output)
        full[mask] = self.block.to(output.dtype)
        self.n_replaced += n
        return torch.where(mask.unsqueeze(-1), full, output)

    def __enter__(self):
        self._handle = self.embed.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.block = None
        return False


# --------------------------------------------------------------------------- query

def _cache_path(audio_path: str, kind: str = "qwen_omni") -> Path:
    """Query cache, keyed by wav AND model family: the query lives in the model's own
    hidden space (3584-d for Qwen2.5-Omni, 2048-d for Qwen3-Omni), so a file written by
    one model is wrong for another (job 10767270 read a Qwen2.5 query into a Qwen3 pool).
    The default family keeps the legacy flat layout so existing caches stay valid."""
    h = hashlib.sha1(str(Path(audio_path).resolve()).encode()).hexdigest()
    root = CACHE_ROOT / "softpool_query"
    return (root if kind == "qwen_omni" else root / kind) / f"{h}.npy"


@torch.no_grad()
def _compute_query(model, processor, audio_path: str) -> torch.Tensor:
    """The model call behind `audio_query`, separated so the cache guard is testable."""
    from qwen_omni_utils import process_mm_info
    conv = [{"role": "user", "content": [{"type": "audio", "audio": audio_path}]}]
    text = processor.apply_chat_template(conv, add_generation_prompt=False, tokenize=False)
    audios, _, _ = process_mm_info(conv, use_audio_in_video=False)
    x = processor(text=text, audio=audios, return_tensors="pt", padding=True,
                  use_audio_in_video=False)
    feats = x["input_features"].to(model.device).to(model.dtype)
    fam = x["feature_attention_mask"].to(model.device)
    hs = model.get_audio_features(feats, fam).last_hidden_state   # (T, D)
    return F.normalize(hs.float().mean(dim=0), dim=-1).cpu()


@torch.no_grad()
def audio_query(model, processor, inp: SoftPoolInput, cache: bool = True) -> torch.Tensor:
    """Mean of the thinker's audio-tower output for this recording, L2-normalised, in
    the thinker's hidden space. Never next to the source wav: cached under CACHE_ROOT."""
    _require_input(inp)
    from src.model import model_kind
    cp = _cache_path(inp.audio_path, model_kind(model) if model is not None else "qwen_omni")
    if cache and cp.exists():
        try:
            return torch.from_numpy(np.load(cp)).float()
        except (EOFError, ValueError, OSError):
            # A concurrent job may have written a partial file (job 10752383 read a
            # header-only .npy); fall through and recompute, then overwrite atomically.
            pass
    q = _compute_query(model, processor, inp.audio_path)
    if cache:
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_name(f"{cp.stem}.{os.getpid()}.tmp.npy")
        np.save(tmp, q.numpy())          # np.save appends .npy only if missing: keep the suffix
        os.replace(tmp, cp)              # atomic: readers see the old file or the whole new one
    return q


# --------------------------------------------------------------------------- prompt

def build_direct_conversation(task: Task, n_marker: int = 0, fmt: str = "direct",
                              text_prompt: str | None = None) -> list[dict]:
    """Audio first, then (optional marker block OR hand-written instruction), then the
    untouched question."""
    question = native_question(task, fmt) if fmt in NATIVE_FORMATS else f"{task.stem}\n{task.choices_block}"
    text = question
    if n_marker and text_prompt:
        raise ValueError("either a soft block or a text prompt, not both")
    if n_marker:
        text = f"{MARKER * n_marker}\n\n{text}"
    elif text_prompt:
        text = f"{text_prompt}\n\n{text}"
    assert text.endswith(question), "question/choices altered"
    user = {"role": "user", "content": [{"type": "audio", "audio": task.audio_path},
                                        {"type": "text", "text": text}]}
    if fmt in NATIVE_FORMATS:
        return [user]                     # the model's template supplies its default system turn
    return [{"role": "system", "content": [{"type": "text", "text": FORMATS[fmt]}]}, user]


def marker_count(processor, conversation: list[dict]) -> int:
    """How many MARKER ids the rendered conversation contains (test + assertion helper)."""
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    tok = getattr(processor, "tokenizer", processor)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return sum(1 for i in ids if i == marker_id(processor))


# --------------------------------------------------------------------------- actor

class SoftPoolHandle:
    """What `evaluate` passes as the `pool` argument for the soft-prompt arm."""

    def __init__(self, pool: SoftPromptPool):
        self.pool = pool


class TextPromptHandle:
    """Hand-written baseline: a fixed TEXT instruction placed exactly where the soft block
    goes (between the audio and the question), no learning, no selection. Passed as the
    `pool` argument so the eval's two arms stay 'no_pool' vs 'pool'."""

    def __init__(self, text: str):
        if not text.strip():
            raise ValueError("text prompt is empty")
        self.text = text.strip()


# How the actor picks slots. `audio` is the method. The other two are PLACEBOS that
# reuse the trained prompts but discard the audio: they exist to show the gain comes
# from selecting the right slots, not from any soft block being present.
SELECTION_MODES = ("audio", "random", "fixed")


def parse_selection(mode: str) -> tuple[str, str | None]:
    if mode.startswith("fixed:"):
        return "fixed", mode.split(":", 1)[1]
    if mode not in ("audio", "random"):
        raise ValueError(f"selection must be audio | random | fixed:<condition>, got {mode!r}")
    return mode, None


class SoftPoolActor:
    """`run(task, pool=None)`: direct-answer format; `pool` is a SoftPoolHandle for the
    soft-prompt arm and None for the control. `top_k` is settable at eval time."""

    def __init__(self, model, processor, top_k: int | None = None, max_new_tokens: int = 8,
                 selection: str = "audio", fmt: str = "direct"):
        self.model, self.processor = model, processor
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        if fmt not in FORMATS:
            raise ValueError(f"fmt must be one of {sorted(FORMATS)}, got {fmt!r}")
        self.fmt = fmt
        self.marker = marker_id(processor)
        self.injector = SoftPromptInjector(model.get_input_embeddings(), self.marker)
        self.last_selection: dict | None = None
        self.selection, self.fixed_condition = parse_selection(selection)

    def _select(self, pool: SoftPromptPool, task: Task):
        k = int(self.top_k or pool.top_k)
        if pool.n_slots == 1:
            # A single shared prompt has nothing to select: no audio query, no encoder
            # call, no model-specific query space. Same block on every item.
            idx = torch.zeros(1, dtype=torch.long, device=pool.keys.device)
            pool.last_idx, pool.last_top = idx, idx[0]
            return None, idx, torch.ones(1, device=pool.keys.device)
        if self.selection == "audio":
            q = audio_query(self.model, self.processor, softpool_input_from_task(task))
            idx, sims = pool.select(q.to(pool.keys.device), k)
            return q, idx, sims
        # Placebos: no query. `sims` is a one-hot over the chosen slots so
        # predicted_condition reports what was injected.
        if self.selection == "random":
            g = torch.Generator().manual_seed(int(hashlib.sha1(task.id.encode()).hexdigest()[:8], 16))
            idx = torch.randperm(pool.n_slots, generator=g)[:k].sort().values.to(pool.keys.device)
        else:
            if self.fixed_condition not in pool.conditions:
                raise ValueError(f"fixed:{self.fixed_condition} not in pool {pool.conditions}")
            c = pool.conditions.index(self.fixed_condition)
            slots = torch.nonzero(pool.slot_condition == c).flatten()
            idx = slots[:k].to(pool.keys.device)
        sims = torch.full((pool.n_slots,), -1.0, device=pool.keys.device)
        sims[idx] = 1.0
        return None, idx, sims

    def run(self, task: Task, pool: SoftPoolHandle | None = None) -> Trajectory:
        from src.model import generate
        retrieved, route, context = [], None, ""
        if pool is None:
            conv = build_direct_conversation(task, fmt=self.fmt)
            raw, raw_full = generate(self.model, self.processor, conv, max_new_tokens=self.max_new_tokens, return_full=True)
        elif isinstance(pool, TextPromptHandle):
            conv = build_direct_conversation(task, fmt=self.fmt, text_prompt=pool.text)
            raw, raw_full = generate(self.model, self.processor, conv, max_new_tokens=self.max_new_tokens, return_full=True)
            context = "text_prompt"
            route = "text_prompt"
        else:
            p = pool.pool
            _, idx, sims = self._select(p, task)
            block = p.block(idx).detach()
            conv = build_direct_conversation(task, n_marker=block.shape[0], fmt=self.fmt)
            with self.injector as inj:
                inj.set_block(block)
                raw, raw_full = generate(self.model, self.processor, conv, max_new_tokens=self.max_new_tokens, return_full=True)
                if inj.n_replaced != block.shape[0]:
                    raise RuntimeError(f"replaced {inj.n_replaced} rows, expected {block.shape[0]}")
                inj.n_replaced = 0
            retrieved = [int(i) for i in idx]
            route = p.predicted_condition(sims)          # top-slot condition, for analysis
            context = f"softpool:{retrieved}"
            self.last_selection = {"slots": retrieved, "route": route,
                                   "sims": [round(float(s), 4) for s in sims]}
        return _score_direct(task, raw, context, retrieved, route, raw_full=raw_full)


def after_think(raw: str) -> str:
    """AF-Think emits a reasoning trace before the answer. Score only what follows the last
    `</think>`; without a closing tag, the last non-empty line (the answer is last)."""
    if "</think>" in raw:
        return raw.rsplit("</think>", 1)[1].strip()
    if "<think>" in raw:
        lines = [l for l in raw.splitlines() if l.strip()]
        return lines[-1] if lines else raw
    return raw


def score_text(raw: str, choices: dict[str, str], answerable: bool, gold: str) -> tuple[str, str | None, bool]:
    """(outcome, pred, correct) for one decoded output. The single place that turns text
    into a score; `softpool_reparse` re-applies it to saved rows."""
    options = {L: t.lower() for L, t in choices.items()}
    outcome, pred = parse_outcome(after_think(raw), options)
    correct = (outcome == "abstain") if not answerable else (pred == gold)
    return outcome, pred, correct


def _score_direct(task: Task, raw: str, context: str, retrieved: list, route, raw_full: str = "") -> Trajectory:
    outcome, pred, correct = score_text(raw, task.choices, task.answerable, task.gold)
    return Trajectory(task_id=task.id, track=task.track, hop=task.hop, condition=task.condition,
                      answerable=task.answerable, family="softpool", context=context, raw=raw,
                      reasoning="", outcome=outcome, pred=pred, gold=task.gold,
                      correct=correct, retrieved=retrieved, route=route, raw_full=raw_full)


# --------------------------------------------------------------------------- training forward

def target_text(task: Task, teacher_pred: str, fmt: str = "direct") -> str:
    """What the pool is trained to make the model say on this example: the teacher's
    letter on answerable items, the abstain phrase when the evidence is gone. Reads
    `task.answerable` (set by the attack), never `task.gold`.

    The letter takes the CASE the format renders: the native layouts print "(A) .. (B)"
    and the model answers "(B) ...", and "(B" and "(b" are different merged tokens, so a
    lowercase target there is wrong at its first token on every example (AF3 smoke
    10770191: loss ~10, agreement 0 %)."""
    if not task.answerable:
        return ABSTAIN_TARGET
    letter = teacher_pred.upper() if fmt in NATIVE_FORMATS else teacher_pred.lower()
    return f"({letter})"


def target_ids(processor, text: str) -> list[int]:
    """`text` is a full target string: "(b)" or ABSTAIN_TARGET."""
    tok = getattr(processor, "tokenizer", processor)
    if len(text) == 1:                      # legacy: a bare letter
        text = f"({text})"
    return tok(text, add_special_tokens=False)["input_ids"]


def teacher_forced_inputs(model, processor, conversation: list[dict], target: list[int]) -> dict:
    """Prompt (with audio) + target ids; labels are -100 everywhere except the target.

    Inputs come from `src.model.build_inputs`, the SAME model-kind dispatch the eval path
    uses. The trainer used to call the Qwen-only path (qwen_omni_utils + `padding=True`)
    directly; on AF3 that pads the mel features to the clip length instead of the fixed
    30 s window its Whisper encoder expects, and the first forward died with
    "size of tensor a (376) must match ... (1500)" (job 10770086)."""
    from src.model import build_inputs
    x = build_inputs(model, processor, conversation)
    x = x.to(model.device).to(model.dtype)
    ids = x["input_ids"]
    tgt = torch.tensor([target], dtype=ids.dtype, device=ids.device)
    x["input_ids"] = torch.cat([ids, tgt], dim=1)
    x["attention_mask"] = torch.cat([x["attention_mask"], torch.ones_like(tgt)], dim=1)
    labels = torch.full_like(x["input_ids"], -100)
    labels[:, ids.shape[1]:] = tgt
    x["labels"] = labels
    return x


def repeat_inputs(x: dict, n: int) -> dict:
    """The same single-example inputs repeated `n` times along the batch axis, so `n`
    different prompt blocks can be scored in ONE forward (batched ES pairs, discrete
    candidates). Only tensors with a leading batch dim of 1 are repeated."""
    out = {}
    for k, v in x.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1 and n > 1:
            out[k] = v.repeat(n, *([1] * (v.dim() - 1)))
        else:
            out[k] = v
    return out


class _HeadSlice:
    """Forward-pre-hook on lm_head that keeps only the positions whose logits predict the
    trailing target tokens: (B, L, D) -> (B, n_target, D). The full-vocabulary logits are
    the memory hog (B x L x 152k floats), not the transformer; with the slice a batch of
    32 candidate blocks costs 32 x n_target x V instead of 32 x 800 x V (~15 GB)."""

    def __init__(self, head, n_target: int):
        self.head, self.n, self._h = head, n_target, None

    def __enter__(self):
        if self.head is not None:
            self._h = self.head.register_forward_pre_hook(
                lambda m, args: (args[0][:, -(self.n + 1):-1],) + tuple(args[1:]))
        return self

    def __exit__(self, *exc):
        if self._h is not None:
            self._h.remove()
        return False


def answer_loss(model, inputs: dict, per_row: bool = False):
    """Mean CE over the target positions and whether the greedy argmax matches at all of
    them (the 'agreement' statistic). Computed here, not via the model's own loss path,
    so the reduction is explicit. The targets are the trailing `n_target` tokens of every
    row (see `teacher_forced_inputs`); rows may be a repeated input with different soft
    blocks (`repeat_inputs`). `per_row=True` returns (loss (B,), agree (B,) bool) instead
    of the pooled (scalar, bool)."""
    labels = inputs.pop("labels")
    n_t = int((labels[0] != -100).sum())
    kw = {} if getattr(model, "_reprompt_kind", "qwen_omni") == "af3" else {"use_audio_in_video": False}
    with _HeadSlice(getattr(model, "lm_head", None), n_t):
        out = model(**inputs, use_cache=False, **kw)
    lg = out.logits
    if lg.shape[1] != n_t:                    # no lm_head hook (fake model in tests): slice here
        lg = lg[:, -(n_t + 1):-1]
    lg = lg.float()
    tgt = labels[:, -n_t:]
    ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape)
    loss_rows = ce.mean(1)
    agree_rows = (lg.argmax(-1) == tgt).all(1)
    if per_row:
        return loss_rows, agree_rows
    return loss_rows.mean(), bool(agree_rows.all())
