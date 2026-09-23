"""RoP's text perturbations, rendered deterministically in code.

RoP (arXiv:2506.03627) perturbs its questions with GPT-4o, prompted to rewrite each one
"using the specified perturbation strategy" while preserving the answer. We do it in
code instead, for the reason `src/expel/render.py` seeds audio variants by file path:
the transform is then identical across runs and phases, and there is no generator
fidelity to defend. An LLM asked to corrupt text while preserving an answer is itself an
unverified step, and it is the step every downstream number rests on.

Four transforms, matching the paper's taxonomy (UIC is the fifth and already exists as
`attacks.INJECTION_TEMPLATES`):

    EC   error character     interior characters of a word shuffled ("times" -> "tmies")
    SC   similar character   visually confusable substitution      ("will"  -> "wiļļ")
    WOO  words out of order  adjacent words transposed
    HW   homophone           phonetically equivalent replacement   ("be"    -> "bee")

ANSWER PRESERVATION IS ENFORCED, NOT ASSERTED. Every transform preserves the word count
and never touches a token that appears in the choice set -- corrupting the word "dog" in
a question whose options include "dog" would not be a perturbation of the question, it
would be a different question. `check_recoverable` states both and is called on every
item; `attacks.py` refuses to run a stem-rewriting attack that has no such check.
"""
from __future__ import annotations

import random
import re

# --------------------------------------------------------------------------- tables
# Cyrillic and Latin-supplement lookalikes. Restricted to characters that render as the
# Latin letter in an ordinary font, so the perturbation is invisible to a reader and
# hostile only to the tokenizer -- which is the point of SC.
HOMOGLYPHS = {
    "a": "а", "c": "с", "e": "е", "i": "і", "j": "ј", "o": "о", "p": "р",
    "s": "ѕ", "x": "х", "y": "у", "d": "ԁ", "h": "һ", "l": "ӏ", "n": "ո",
}

# Only TRUE homophones: the replacement must sound identical, so a speech-grounded model
# loses nothing it could have heard and the corruption is purely orthographic. "the" ->
# "thee" was in an earlier draft and is wrong (/ðə/ vs /ðiː/); so was "been" -> "bean",
# which is a homophone only in some accents. Both are excluded -- a near-homophone makes
# the perturbation harder in a way the taxonomy does not claim.
HOMOPHONES = {
    "be": "bee", "by": "buy", "for": "four", "hear": "here",
    "here": "hear", "in": "inn", "its": "it's", "know": "no", "made": "maid",
    "male": "mail", "no": "know", "one": "won", "or": "oar", "our": "hour",
    "piece": "peace", "right": "write", "sea": "see", "see": "sea", "some": "sum",
    "son": "sun", "their": "there", "there": "their", "threw": "through",
    "to": "too", "two": "too", "wait": "weight", "way": "weigh", "weak": "week",
    "week": "weak", "which": "witch", "whole": "hole", "would": "wood", "write": "right",
    "you're": "your", "your": "you're",
}

TYPES = ("ec", "sc", "woo", "hw")
_TOKEN = re.compile(r"[A-Za-z']+")
DEFAULT_RATE = 0.3

# `rate` is a fraction of ELIGIBLE positions, and eligibility differs enormously between
# transforms: WOO can move nearly any adjacent pair, while HW needs an actual homophone.
# Measured over 240 SAKURA items, one shared rate of 0.3 corrupts 5.1% of words under HW
# and 43.6% under WOO -- an eightfold difference. Comparing repairability across types at
# a shared `rate` would therefore compare how much damage was done, not how repairable it
# is, and would do so while looking like a controlled experiment.
#
# These rates equalise the REALISED corruption at ~15% of words:
#     ec 0.25 -> 15.4%     sc 0.18 -> 15.2%     woo 0.08 -> 14.7%
#
# HW is absent because it cannot reach 15%: perturbing every eligible word tops out at
# 12.3% and still leaves 8% of items with no homophone at all. It is not rate-matchable
# on stems this short. It stays in ROP_REPRODUCTION_CONDITIONS anyway, because RoP does
# not rate-match either -- a faithful reproduction runs all five at their natural rates,
# with the realised corruption reported alongside.
CALIBRATED_RATES = {"ec": 0.25, "sc": 0.18, "woo": 0.08}
TARGET_EDIT_FRACTION = 0.15


def _rng(task_id: str, kind: str, seed: int) -> random.Random:
    """Seeded by (item, transform), so a variant is identical across runs and phases."""
    return random.Random(f"{kind}|{task_id}|{seed}")


def protected_words(choices: dict) -> set[str]:
    """Every word appearing in any answer option. These are never perturbed.

    Corrupting a word that is also an option changes what the question is asking about,
    not how it is written, and no amount of error correction can be scored fairly on it.
    """
    out: set[str] = set()
    for text in choices.values():
        out.update(w.lower() for w in _TOKEN.findall(str(text)))
    return out


def _eligible(word: str, protected: set[str], min_len: int) -> bool:
    return len(word) >= min_len and word.lower() not in protected


class NoEligibleWords(ValueError):
    """Nothing in this stem can carry this perturbation.

    Raised rather than returning the stem unchanged. An unperturbed item sitting in the
    attacked arm is the worst kind of failure available here: it does not crash, it
    scores as a clean item under an attack label, and it dilutes the measured attack
    effect toward zero -- which reads as "the perturbation is harmless".

    It happens legitimately: `hw` needs a homophone in the stem, and short SAKURA
    questions often contain none. Callers must count these and exclude them from the
    condition, not silently keep them.
    """


def _pick(indices: list[int], rate: float, rng: random.Random, kind: str) -> set[int]:
    """At least one index whenever anything is eligible."""
    if not indices:
        raise NoEligibleWords(
            f"no word in this stem is eligible for {kind!r} (every candidate is too "
            "short, or appears in the answer options and may not be touched)")
    n = max(1, round(len(indices) * rate))
    return set(rng.sample(indices, min(n, len(indices))))


# --------------------------------------------------------------------------- transforms
def error_character(stem: str, choices: dict, rng: random.Random,
                    rate: float = DEFAULT_RATE) -> str:
    """Shuffle each selected word's interior characters. First and last are kept, so the
    word stays readable and its character multiset is unchanged."""
    protected = protected_words(choices)
    words = stem.split()
    # _core, not the raw word: a word carrying punctuation ("response,") does not match
    # the protected set as written, and an earlier draft compared the raw string and so
    # corrupted an option word. check_recoverable caught it, which is what it is for.
    idx = [i for i, w in enumerate(words) if _eligible(_core(w), protected, 4)]
    for i in _pick(idx, rate, rng, "ec"):
        words[i] = _map_core(words[i], lambda c: _shuffle_interior(c, rng))
    return " ".join(words)


def similar_character(stem: str, choices: dict, rng: random.Random,
                      rate: float = DEFAULT_RATE) -> str:
    """Replace one or more characters with visual lookalikes."""
    protected = protected_words(choices)
    words = stem.split()
    idx = [i for i, w in enumerate(words)
           if _eligible(_core(w), protected, 3) and any(c in HOMOGLYPHS for c in _core(w).lower())]
    for i in _pick(idx, rate, rng, "sc"):
        words[i] = _map_core(words[i], lambda c: _swap_glyph(c, rng))
    return " ".join(words)


def words_out_of_order(stem: str, choices: dict, rng: random.Random,
                       rate: float = DEFAULT_RATE) -> str:
    """Transpose adjacent word pairs. The word multiset is unchanged.

    Pairs are chosen from non-overlapping positions so a word is moved at most once and
    the count is exactly preserved.
    """
    protected = protected_words(choices)
    words = stem.split()
    idx = [i for i in range(len(words) - 1)
           if _core(words[i]).lower() not in protected
           and _core(words[i + 1]).lower() not in protected]
    chosen, used = [], set()
    for i in sorted(_pick(idx, rate, rng, "woo")):
        if i in used or i + 1 in used:
            continue
        chosen.append(i)
        used.update({i, i + 1})
    for i in chosen:
        words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def homophone(stem: str, choices: dict, rng: random.Random,
              rate: float = DEFAULT_RATE) -> str:
    """Replace words with phonetically identical ones."""
    protected = protected_words(choices)
    words = stem.split()
    idx = [i for i, w in enumerate(words)
           if _core(w).lower() in HOMOPHONES and _core(w).lower() not in protected]
    for i in _pick(idx, rate, rng, "hw"):
        words[i] = _map_core(words[i], lambda c: _match_case(c, HOMOPHONES[c.lower()]))
    return " ".join(words)


TRANSFORMS = {"ec": error_character, "sc": similar_character,
              "woo": words_out_of_order, "hw": homophone}


def perturb(stem: str, choices: dict, kind: str, task_id: str, seed: int = 0,
            rate: float = DEFAULT_RATE) -> str:
    if kind not in TRANSFORMS:
        raise ValueError(f"unknown perturbation {kind!r}; have {TYPES}")
    return TRANSFORMS[kind](stem, choices, _rng(task_id, kind, seed), rate)


# --------------------------------------------------------------------------- helpers
def _core(word: str) -> str:
    """The alphabetic core, without surrounding punctuation ('sound?' -> 'sound')."""
    m = _TOKEN.search(word)
    return m.group(0) if m else ""


def _map_core(word: str, fn) -> str:
    """Apply `fn` to the alphabetic core, leaving punctuation where it was."""
    m = _TOKEN.search(word)
    if not m:
        return word
    return word[:m.start()] + fn(m.group(0)) + word[m.end():]


def _shuffle_interior(core: str, rng: random.Random) -> str:
    mid = list(core[1:-1])
    for _ in range(8):                       # a shuffle that changes nothing is not one
        rng.shuffle(mid)
        if "".join(mid) != core[1:-1]:
            break
    return core[0] + "".join(mid) + core[-1]


def _swap_glyph(core: str, rng: random.Random) -> str:
    pos = [i for i, c in enumerate(core) if c.lower() in HOMOGLYPHS]
    if not pos:
        return core
    out = list(core)
    for i in rng.sample(pos, max(1, len(pos) // 2)):
        out[i] = HOMOGLYPHS[core[i].lower()]
    return "".join(out)


def _match_case(src: str, repl: str) -> str:
    return repl.capitalize() if src[:1].isupper() else repl


# --------------------------------------------------------------------------- the guard
class NotRecoverable(ValueError):
    """The perturbation touched something that decides the answer."""


def check_recoverable(src_stem: str, out_stem: str, choices: dict, kind: str) -> None:
    """Answer preservation for a transform that REWRITES the stem.

    `attacks._assert_answer_preserving` requires the original stem to survive as a
    contiguous substring, which is right for injection (content is only added) and
    impossible here by construction. This is the replacement, and it is not weaker in the
    way that matters: the two things that could move the answer are a protected word
    being corrupted and words being added or dropped, and both are checked.
    """
    if len(src_stem.split()) != len(out_stem.split()):
        raise NotRecoverable(
            f"{kind} changed the word count ({len(src_stem.split())} -> "
            f"{len(out_stem.split())}); every transform here must preserve it")

    protected = protected_words(choices)
    src_hits = [w for w in (_core(x).lower() for x in src_stem.split()) if w in protected]
    out_words = {_core(x).lower() for x in out_stem.split()}
    for w in src_hits:
        if w not in out_words:
            raise NotRecoverable(
                f"{kind} corrupted {w!r}, which appears in the answer options. That "
                "changes what the question asks, not how it is written.")

    if kind == "woo":
        if sorted(src_stem.split()) != sorted(out_stem.split()):
            raise NotRecoverable("woo must permute words, not alter them")
    if kind == "ec":
        for a, b in zip(src_stem.split(), out_stem.split()):
            if sorted(_core(a).lower()) != sorted(_core(b).lower()):
                raise NotRecoverable(
                    f"ec altered the letters of {a!r} -> {b!r}; it may only reorder them")


def edit_count(src_stem: str, out_stem: str) -> int:
    """How many word positions actually changed.

    Transform-agnostic on purpose. Eligibility differs sharply between transforms -- WOO
    can move almost any adjacent pair while HW needs a homophone -- so the same `rate`
    buys very different amounts of corruption, and a comparison across perturbation types
    needs the realised number rather than the requested one.

    It is also the last no-op check. `_pick` guarantees a position was SELECTED; it cannot
    guarantee the edit took, because shuffling the interior of "aaaa" changes nothing.
    """
    return sum(1 for a, b in zip(src_stem.split(), out_stem.split()) if a != b)
