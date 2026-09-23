"""The knowledge pool: scoped insights with ExpeL importance counters.

Four operations, from ExpeL (Zhao et al., 2024): ADD, UPVOTE, DOWNVOTE, EDIT.

Ranking and retirement are a bandit over win rates, not a raw counter. The first full
runs showed why: a raw net count inflates ~5x faster than the evidence (every resisted
pair upvotes all k retrieved insights), so the leaders reached 133-139 and a downvote
moved them by 1. Retirement worked at the bottom of the pool and was inert at the top --
the exact failure the counter existed to prevent.

  * retrieval ranks by the UPPER bound (optimism), so a freshly added insight is tried
    before it has a record instead of being buried by incumbents it never got to beat;
  * retirement uses the LOWER bound against the ambient resist rate (pessimism), so an
    insight is dropped only when it is confidently no better than carrying nothing.

Two guards, both because the pool would otherwise learn the wrong thing and still look
healthy:
  * `_asserts_an_answer` — an insight that names the answer memorises the training half
    instead of teaching a strategy. It scores well on train and transfers nothing.
  * near-duplicate detection — the same lesson re-proposed is an UPVOTE, not a second
    entry, otherwise retrieval fills with paraphrases of whatever the reflector likes
    saying and the top-k stops being a summary of what was learned. Exact-string dedup
    is not enough: the first GPU smoke run (job 10500992) produced
        k0002 "Always verify the consistency of the voice characteristics across
               multiple attempts with the same recording."
        k0003 k0002 + ", regardless of the question text."
    as two entries out of four. Both a high Jaccard and a strict-superset restatement
    are therefore treated as the same insight.
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

from src.expel.types import Insight, PoolOp

MAX_INSIGHT_CHARS = 320

# Retirement. An insight earns its place only by beating the AMBIENT resist rate -- how
# often the model shrugs off the attack with no help from that insight. A fixed
# threshold cannot discriminate here: because all top-k insights share credit for every
# outcome, nearly every win rate lands near 0.5 (measured, job 10501061: k0012 0.52,
# k0022 0.52, against an ambient 244/432 = 0.565). Judged against ambient, those are
# below water and retire; k0029 (0.82) and k0020 (0.70) clear it comfortably.
Z = 1.96
MIN_TRIALS_TO_RETIRE = 12
PRIOR_WINS, PRIOR_LOSSES = 1, 2

# Attribution. In the first runs every scope held <= top_k insights, so every item in a
# track received the identical block and their damage counts came out identical
# (Gender 59/300, 55/300) and perfectly collinear -- no insight could be credited or
# blamed, which is the whole point of keeping per-insight counters. Retrieval now (a)
# never returns a whole scope and (b) varies the subset per item, seeded so it is
# reproducible. RETRIEVAL_SLACK is how many extra candidates the sample draws from:
# larger means better attribution, weaker average block.
RETRIEVAL_SLACK = 2

# Similarity retrieval. An insight matches an item when the item's (audio, instruction)
# key is close to the mean key of the failures the insight was learned from. The floor
# keeps an unrelated rule out of the prompt entirely: with scope retrieval a lesson
# learned on a barking dog was injected into every question in the track, and injecting
# nothing is a real option that scope retrieval never had.
SIM_FLOOR = 0.25

# Advice the model cannot act on at inference. All three observed in the first full runs
# and all three passed the answer-memorisation guard, because being useless is not the
# same as naming an answer.
_DEVELOPER_ADVICE = re.compile(
    r"\b(train(?:ing|ed)?\s+(?:the\s+|on\s+)?(?:model|dataset)|fine-?tun|training data|"
    r"re-?train|the model (?:is|be|should be) trained)\b", re.I)
_MULTI_ATTEMPT = re.compile(
    r"\b(?:multiple|different|several|across|previous|each)\s+(?:attempts?|runs?|trials?)"
    r"|attempts? with the same\b", re.I)
# The injected channel itself. Naming it is fine ONLY alongside scepticism -- "verify the
# metadata's prediction against the audio INDEPENDENTLY" is a strategy; "cross-reference
# the speaker's gender WITH the metadata provided" tells the model to trust the attack.
_INJECTED_CHANNEL = re.compile(r"\b(metadata|caption|annotator|classifier|hint|tag(?:ged)?)\b", re.I)
_SCEPTICAL = re.compile(
    # Suffixes matter: the word actually emitted is "independently", which \bindependent\b
    # does not match, so the best insight of the first run was rejected as its opposite.
    r"\b(independent\w*|regardless|ignor\w*|disregard\w*|unverified|misleading|not rely|"
    r"rather than|over any|prioriti[sz]\w*|do not trust|without rely\w*|even if|"
    r"may be false|skeptic\w*|sceptic\w*|claim\w*)\b", re.I)


# The reflector echoes its own prompt grammar and gets cut off mid-sentence. Both
# observed in the first full runs: "<a new general strategy, one sentence>" entered the
# pool verbatim, as did the fragments "Always verify the context" and "Always verify the
# accuracy of the". All three clear the 15-char length floor.
_TEMPLATE_ECHO = re.compile(r"[<>]|\ba (?:new )?general strategy\b|one sentence\b", re.I)
MIN_INSIGHT_WORDS = 6


def _malformed(text: str) -> str:
    """Reason this is not a well-formed sentence, or '' if it is."""
    if _TEMPLATE_ECHO.search(text):
        return "echoes the prompt template"
    if len(text.split()) < MIN_INSIGHT_WORDS:
        return f"under {MIN_INSIGHT_WORDS} words"
    if not text.rstrip().endswith((".", "!", "?")):
        return "truncated mid-sentence"
    return ""


# Direction, not just vocabulary. The v2 run (job 10504094) admitted, at 8-1 on dev:
#     "Prioritize the question text over the audio content when the audio content is
#      misleading."
# That is the attack adopted as policy. It passed every guard because it contains
# "misleading" -- a scepticism marker -- while instructing the model to prefer the
# injected channel. Naming the right words in the wrong order is the failure mode.
_PREFERENCE = re.compile(r"([^.;]{0,90}?)\b(?:over|rather than|instead of|in preference to|"
                         r"takes? precedence over)\b([^.;]{0,90})", re.I)
_TEXT_SIDE = re.compile(r"\b(question|text|metadata|caption|hint|annotator|classifier|"
                        r"prompt|claim|label|tag)\w*", re.I)
_AUDIO_SIDE = re.compile(r"\b(audio|recording|sound|clip|acoustic|hear|listen|speech|"
                         r"voice|utterance)\w*", re.I)
# "when the audio is misleading, use the question text" -- the same reversal, conditional.
_COND_REVERSE = re.compile(
    r"\b(?:when|if|whenever|unless)\b[^.;]{0,70}?"
    r"\b(?:audio|recording|sound|clip|speech)\w*[^.;]{0,50}?"
    r"\b(?:misleading|unclear|ambiguous|unreliable|contradict\w*|uncertain|inconclusive)\b"
    r"[^.;]{0,70}?\b(?:use|rely|prefer\w*|prioriti[sz]\w*|follow|trust|trust\w*|defer)\b"
    r"[^.;]{0,40}?\b(?:question|text|metadata|caption|hint|prompt|option)\w*", re.I)


def _reverses_evidence_order(text: str) -> str:
    """Reason this puts the question text above the audio, or '' if it does not."""
    for m in _PREFERENCE.finditer(text):
        first, second = m.group(1), m.group(2)
        if (_TEXT_SIDE.search(first) and _AUDIO_SIDE.search(second)
                and not _AUDIO_SIDE.search(first)):
            return "prefers the question text over the audio"
    if _COND_REVERSE.search(text):
        return "falls back to the question text when the audio is hard"
    return ""


def _inactionable(text: str) -> str:
    """Reason this insight cannot be acted on at inference, or '' if it can."""
    if reason := _reverses_evidence_order(text):
        return reason
    if _DEVELOPER_ADVICE.search(text):
        return "advice to the developer, not usable at inference"
    if _MULTI_ATTEMPT.search(text):
        return "refers to multiple attempts; the model gets one pass"
    if _INJECTED_CHANNEL.search(text) and not _SCEPTICAL.search(text):
        return "treats the injected channel as evidence"
    return ""


def wilson_bounds(wins: int, losses: int, z: float = Z) -> tuple:
    """(lower, upper) confidence bounds on the win rate. n=0 -> (0.0, 1.0)."""
    n = wins + losses
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom, (centre + margin) / denom

# "the answer is (b)", "always pick dog", "choose option c" — memorisation, not strategy.
_ANSWER_ASSERTION = re.compile(
    r"\b(the (?:correct )?answer is|always (?:choose|pick|select|answer)|"
    r"(?:choose|pick|select|answer) (?:option )?\(?[a-d]\)?\b|"
    r"the gold (?:answer|label)|correct option is)", re.I)


# Two normalised insights count as the same lesson at either of these. Jaccard catches
# reworded near-copies; containment catches "the same sentence plus a trailing clause",
# which a 7B reflector emits readily and which Jaccard alone scores only ~0.7.
DUP_JACCARD = 0.70
DUP_CONTAINMENT = 0.85
MIN_DUP_TOKENS = 6          # below this, generic fragments would over-merge


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def _tokens(text: str) -> set:
    return set(_norm(text).split())


def same_lesson(a: str, b: str) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    if inter and inter / len(ta | tb) >= DUP_JACCARD:
        return True
    if min(len(ta), len(tb)) >= MIN_DUP_TOKENS and inter / min(len(ta), len(tb)) >= DUP_CONTAINMENT:
        return True
    return False


def _asserts_an_answer(text: str) -> bool:
    return bool(_ANSWER_ASSERTION.search(text))


def scope_of(track: str, family: str) -> str:
    return f"{track}|{family}"


class KnowledgePool:
    """Insights keyed by scope. Retrieval is what the actor sees; ops are how it grows."""

    # HOW MUCH AN INSIGHT MAY SAY ABOUT THE PERTURBATION. Three policies, because the
    # right answer depends on how the insight is SELECTED, not on its wording:
    #
    #   "strict"          no perturbation vocabulary at all. The rule agent_optimize uses
    #                     for the CATEGORY arm, where selection is keyed on the question
    #                     category -- which cannot tell noise from silence, so naming one
    #                     is unearned.
    #   "presupposition"  hedged mentions allowed, assertions and discard-directives not.
    #   "off"             naming the perturbation is allowed outright. DEFAULT.
    #
    # "off" is the default because in this pool retrieval is keyed on the AUDIO
    # FINGERPRINT (src/expel/retrieval.py), so an insight about noise is retrieved BECAUSE
    # THE AUDIO SOUNDS NOISY. The perturbation identity is earned from the signal, exactly
    # as in the routed arm -- and the router measures how well that works: 99.9% on regime
    # and 85.5% on condition from audio alone, on 1600 held-out wavs. retrieval.py's own
    # docstring makes the same argument: "unanswerable inputs cluster on their own and
    # retrieve the insight that tells the model to decline -- without anyone telling the
    # model that this input was masked."
    #
    # WHAT THIS DOES NOT LICENSE, and what still has to hold:
    #   * selection must never read the true label. That is the ORACLE arm and it is an
    #     upper bound, never a deployable result (src/checker.py:assert_selection_legal).
    #   * clean collateral is still reported in every cell. A prompt naming noise is
    #     applied to clean audio too whenever the fingerprint is close, and what that
    #     costs is measured rather than assumed.
    # Set leak_policy="strict" to reproduce the conservative arm for comparison.
    def __init__(self, insights: list[Insight] | None = None, leak_policy: str = "off"):
        if leak_policy not in ("strict", "presupposition", "off"):
            raise ValueError(f"unknown leak_policy {leak_policy!r}")
        self.leak_policy = leak_policy
        self.insights: dict[str, Insight] = {i.id: i for i in (insights or [])}
        self.log: list[PoolOp] = []
        self._next = len(self.insights) + 1
        # Ambient: how often the attack is shrugged off overall. The bar an insight must
        # clear to justify occupying a retrieval slot.
        self.ambient_wins = 0
        self.ambient_losses = 0

    @property
    def ambient_rate(self) -> float:
        n = self.ambient_wins + self.ambient_losses
        return self.ambient_wins / n if n else 0.5

    def observe_ambient(self, wins: int, losses: int) -> None:
        self.ambient_wins += wins
        self.ambient_losses += losses

    @staticmethod
    def rank_score(ins: Insight) -> float:
        """Optimistic upper bound under a weak pessimistic prior.

        Unsmoothed, every untried insight scores exactly 1.0 and the whole top-k is
        untried — measured on the first runs' op streams, 8 of 13 survivors had no record
        at all, so proven strategies were starved by a queue of unknowns. The (1, 2)
        prior puts a fresh insight around 0.75: ahead of a mediocre incumbent (~0.63),
        behind a strong one (~0.83). It still gets tried; it no longer displaces
        everything that has earned its place.
        """
        return wilson_bounds(ins.n_up + PRIOR_WINS, ins.n_down + PRIOR_LOSSES)[1]

    def retire(self) -> list[PoolOp]:
        """Drop insights confidently no better than carrying nothing. Call per batch."""
        out: list[PoolOp] = []
        for ins in list(self.insights.values()):
            trials = ins.n_up + ins.n_down
            if trials < MIN_TRIALS_TO_RETIRE:
                continue
            lcb = wilson_bounds(ins.n_up, ins.n_down)[0]
            if lcb < self.ambient_rate:
                del self.insights[ins.id]
                op = PoolOp(op="RETIRE", scope=ins.scope, insight_id=ins.id, applied=True,
                            text=ins.text,
                            reason=f"lcb {lcb:.3f} < ambient {self.ambient_rate:.3f} "
                                   f"over {trials} trials")
                self.log.append(op)
                out.append(op)
        return out

    # ---------------------------------------------------------------- read side
    def retrieve_similar(self, key, k: int = 5, floor: float = SIM_FLOOR) -> list[Insight]:
        """Insights whose centroid is closest to this item's key, above the floor.

        Retrieval is keyed on the audio and the instruction only -- the two things the
        actor is given -- so nothing about which intervention was applied enters the
        decision. Unanswerable audio routes to the declining insights because silence
        and broadband noise SOUND unlike every clean recording, not because they are
        labelled.
        """
        from src.expel.retrieval import cosine

        scored = [(cosine(key, i.centroid), i) for i in self.insights.values() if i.centroid]
        scored = [(sim, i) for sim, i in scored if sim >= floor]
        scored.sort(key=lambda si: (-si[0] * self.rank_score(si[1]), si[1].id))
        # Never inject the entire pool. Without this cap the "similarity" arm returned
        # all of it on every item: measured on the phase 2 dev grid, all 4 insights
        # appeared in 590 of 600 rows and their damage counts were identical (60/43),
        # so nothing could be attributed and the arm was a fixed preamble wearing a
        # retrieval label. The cap lives in retrieve() for the scope path; it was simply
        # never applied here.
        if len(scored) > 1:
            k = min(k, len(scored) - 1)
        return [i for _, i in scored[:k]]

    def render_similar(self, key, k: int = 5, floor: float = SIM_FLOOR) -> str:
        hits = self.retrieve_similar(key, k, floor)
        if not hits:
            return ""
        lines = "\n".join(f"{n}. {i.text}" for n, i in enumerate(hits, 1))
        return ("Things you have learned from previous attempts at this kind of "
                f"question:\n{lines}")

    def retrieve(self, track: str, family: str, k: int = 5,
                 salt: str | None = None) -> list[Insight]:
        """Track-specific insights first, then cross-track ones for the same family.

        `salt` (the task id) varies which subset an item sees, so an insight is present
        for some items and absent for others and its win rate means something. Without
        it every item in a scope gets the same block and the counters are collinear.
        """
        want = {scope_of(track, family), scope_of("*", family)}
        hits = [i for i in self.insights.values() if i.scope in want]
        hits.sort(key=lambda i: (i.scope.startswith("*"), -self.rank_score(i), i.id))
        if len(hits) <= 1:
            return hits                       # a scope of one cannot be varied
        k_eff = min(k, len(hits) - 1)         # never inject the whole scope
        if salt is None:
            return hits[:k_eff]
        window = hits[: min(len(hits), k_eff + RETRIEVAL_SLACK)]
        chosen = random.Random(f"{salt}|{track}|{family}").sample(window, k_eff)
        return sorted(chosen, key=lambda i: (i.scope.startswith("*"), -self.rank_score(i), i.id))

    def render(self, track: str, family: str, k: int = 5, salt: str | None = None) -> str:
        """The knowledge block injected into the actor's prompt. '' when the pool is empty."""
        hits = self.retrieve(track, family, k, salt)
        if not hits:
            return ""
        lines = "\n".join(f"{n}. {i.text}" for n, i in enumerate(hits, 1))
        return ("Things you have learned from previous attempts at this kind of "
                f"question:\n{lines}")

    def retrieved_ids(self, track: str, family: str, k: int = 5,
                      salt: str | None = None) -> list[str]:
        return [i.id for i in self.retrieve(track, family, k, salt)]

    # ---------------------------------------------------------------- write side
    def apply(self, op: PoolOp) -> PoolOp:
        """Apply one operation, recording why it was refused if it was."""
        handler = {"ADD": self._add, "UPVOTE": self._vote, "DOWNVOTE": self._vote,
                   "EDIT": self._edit}.get(op.op.upper())
        if handler is None:
            op.rejected = f"unknown op {op.op!r}"
        else:
            handler(op)
        self.log.append(op)
        return op

    def apply_all(self, ops: list[PoolOp]) -> list[PoolOp]:
        return [self.apply(o) for o in ops]

    def _find_duplicate(self, scope: str, text: str) -> Insight | None:
        """Highest-count existing insight in `scope` that says the same thing."""
        hits = [i for i in self.insights.values()
                if i.scope == scope and same_lesson(i.text, text)]
        return max(hits, key=lambda i: (i.count, i.id)) if hits else None

    def _validate(self, op: PoolOp, text: str) -> bool:
        text = text.strip()
        from src.agent_optimize import discards_audio as _discards_audio
        from src.agent_optimize import presupposes_perturbation as _presupposes
        if len(text) < 15:
            op.rejected = "too short to be a strategy"
        elif len(text) > MAX_INSIGHT_CHARS:
            op.rejected = f"over {MAX_INSIGHT_CHARS} chars"
        elif _asserts_an_answer(text):
            op.rejected = "asserts a specific answer (memorisation, not strategy)"
        elif reason := _malformed(text):
            op.rejected = reason
        elif reason := _inactionable(text):
            op.rejected = reason
        elif reason := _discards_audio(text):
            # Kept under EVERY policy, including "off", because the objection is not about
            # leaking. "Ignore the injected second voice" tells the model to throw away
            # part of what it heard, and on a clean item there is nothing extraneous to
            # throw away -- so it can only do harm there, and clean items are the majority
            # of any deployment. Measured precedent: "disregard any secondary or background
            # elements such as laughter, engines, or speech patterns" scored 79.2 on dev
            # and passed every word-level guard before this check existed.
            op.rejected = f"discards audio: {reason}"
        elif self.leak_policy != "off" and (reason := _presupposes(text)):
            op.rejected = f"presupposes the perturbation: {reason}"
        elif self.leak_policy == "strict":
            from src.agent_optimize import mentions_perturbation
            if bad := mentions_perturbation(text):
                op.rejected = f"names the perturbation: {bad}"
        return not op.rejected

    def _add(self, op: PoolOp) -> None:
        text = (op.text or "").strip()
        if not self._validate(op, text):
            return
        dup = self._find_duplicate(op.scope, text)
        if dup is not None:                    # same lesson again -> endorsement
            dup.count += 1
            dup.n_up += 1
            dup.absorb(op.key)
            op.op, op.insight_id, op.applied = "UPVOTE", dup.id, True
            return
        iid = f"k{self._next:04d}"
        self._next += 1
        ins = Insight(id=iid, scope=op.scope, text=text)
        ins.absorb(op.key)
        self.insights[iid] = ins
        op.insight_id, op.applied = iid, True

    def _vote(self, op: PoolOp) -> None:
        ins = self.insights.get(op.insight_id or "")
        if ins is None:
            op.rejected = f"no insight {op.insight_id!r}"
            return
        if op.op.upper() == "UPVOTE":
            ins.count += 1
            ins.n_up += 1
            ins.absorb(op.key)          # the cluster grows toward what it works on
        else:
            ins.count -= 1
            ins.n_down += 1
        op.applied = True

    def _edit(self, op: PoolOp) -> None:
        ins = self.insights.get(op.insight_id or "")
        if ins is None:
            op.rejected = f"no insight {op.insight_id!r}"
            return
        if not self._validate(op, op.text or ""):
            return
        ins.text = (op.text or "").strip()
        ins.count += 1
        ins.n_edit += 1
        op.applied = True

    # ---------------------------------------------------------------- persistence
    def to_json(self) -> dict:
        return {"insights": [i.to_json() for i in self.insights.values()],
                "ambient": {"wins": self.ambient_wins, "losses": self.ambient_losses},
                "log": [o.to_json() for o in self.log]}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgePool":
        d = json.loads(Path(path).read_text())
        pool = cls([Insight(**i) for i in d.get("insights", [])])
        pool.log = [PoolOp(**o) for o in d.get("log", [])]
        amb = d.get("ambient", {})
        pool.ambient_wins, pool.ambient_losses = amb.get("wins", 0), amb.get("losses", 0)
        nums = [int(i[1:]) for i in pool.insights if i[1:].isdigit()]
        pool._next = max(nums, default=0) + 1
        return pool

    def summary(self) -> dict:
        by_scope: dict[str, int] = {}
        for i in self.insights.values():
            by_scope[i.scope] = by_scope.get(i.scope, 0) + 1
        applied = sum(1 for o in self.log if o.applied)
        return {"n_insights": len(self.insights), "by_scope": by_scope,
                "ops_applied": applied, "ops_rejected": len(self.log) - applied,
                "ambient_rate": round(self.ambient_rate, 3),
                "retired": sum(1 for o in self.log if o.op == "RETIRE")}
