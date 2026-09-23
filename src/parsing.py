"""Answer parsing for SAKURA multiple-choice responses.

`extract_letter`, `parse_options`, `split_instruction` and `_norm` are ported from
an earlier internal evaluation library. That
version is the most hardened of the four implementations across the prior repos, and
its ordering guards are documented against specific observed false positives — the
"A sharp gaze" article bug, and 1-char answers matching any option containing that
letter. Do not reorder the rules.

New here: three-way outcome classification. The prior code returned a letter or None,
which collapses "the model correctly refused because the audio is silent" into the
same bucket as "the model produced garbage". Phase 0 needs those separated: under
100% masking and -20 dB SNR, abstention is the *correct* behaviour, and scoring it as
a wrong answer would invert the finding at exactly the conditions that matter.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- ported: text / options
OPTION_RE = re.compile(r"\(([a-d])\)\s*(.+?)(?=\s*\([a-d]\)|$)", re.IGNORECASE)
_OPT_SPLIT = re.compile(r"\([aA]\)")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def parse_options(instruction: str) -> dict[str, str]:
    """'... (a) Feathers (b) Scales ...' -> {'a': 'feathers', 'b': 'scales', ...}."""
    return {m.group(1).lower(): _norm(m.group(2)) for m in OPTION_RE.finditer(instruction)}


def parse_options_raw(instruction: str) -> dict[str, str]:
    """Same, but keeping original casing/punctuation — needed to re-render permuted choices."""
    return {m.group(1).lower(): m.group(2).strip() for m in OPTION_RE.finditer(instruction)}


def split_instruction(instr: str) -> tuple[str, str]:
    """'... question? (a) X (b) Y' -> ('... question?', '(a) X (b) Y')."""
    m = _OPT_SPLIT.search(instr)
    return (instr[:m.start()].strip(), instr[m.start():].strip()) if m else (instr.strip(), "")


# --------------------------------------------------------------------------- ported: letter extraction
_CONCLU = re.compile(r"<Conclu[^>]*>(.*?)(?:</Conclu|$)", re.S | re.I)
_REASON = re.compile(r"<Reason[^>]*>(.*?)(?:</Reason|$)", re.S | re.I)
_PAREN = re.compile(r"\(([a-dA-D])\)")
_PUNCT = re.compile(r"(?:^|[^a-zA-Z])([a-dA-D])[\.\):]")
_STANDALONE = re.compile(r"(?:^|[^a-zA-Z])([a-dA-D])(?:[^a-zA-Z]|$)")
_LEADANS = re.compile(r"^\s*([a-dA-D])(?:[\.\)\:]|\s*$)")


def extract_letter(raw: str, options: dict[str, str]) -> str | None:
    """Map an Omni answer to an option letter. Ported verbatim; ordering is load-bearing."""
    m = _CONCLU.search(raw)
    content = (m.group(1) if m else raw).strip()
    content = re.sub(r"<[^>]+>", " ", content)
    cn = _norm(content)

    m = _PAREN.search(content)
    if m:
        return m.group(1).lower()
    m = _LEADANS.match(content)
    if m and m.group(1).lower() in options:
        return m.group(1).lower()
    for L, t in options.items():
        if t and cn == t:
            return L
    m = _PUNCT.search(content)
    if m:
        return m.group(1).lower()
    for L, t in options.items():
        if t and len(cn) >= 3 and (t in cn or cn in t):
            return L
    ms = list(_STANDALONE.finditer(content))
    if ms:
        return ms[-1].group(1).lower()
    if cn:
        cset, best, bestL = set(cn.split()), 0, None
        for L, t in options.items():
            ov = len(cset & set(t.split()))
            if ov > best:
                best, bestL = ov, L
        if best:
            return bestL
    return None


# --------------------------------------------------------------------------- new: reasoning / abstention
def split_trace(raw: str) -> tuple[str, str]:
    """Split a CoT response into (reasoning, conclusion).

    Handles <Reasoning>..</Reasoning><Conclusion>..</Conclusion>, a bare <Conclusion>,
    and unstructured output (everything counts as conclusion so parsing still runs).
    """
    reasoning = ""
    m = _REASON.search(raw)
    if m:
        reasoning = m.group(1).strip()
    m = _CONCLU.search(raw)
    if m:
        return reasoning, m.group(1).strip()
    if reasoning:
        # Reasoning tag present but no conclusion tag: conclusion is whatever follows it.
        tail = raw[raw.lower().rfind("</reason") :] if "</reason" in raw.lower() else ""
        return reasoning, re.sub(r"^</[^>]+>", "", tail).strip() or raw.strip()
    return "", raw.strip()


# Phrases indicating the model declined to choose. Matched against the CONCLUSION only —
# never the reasoning, because a valid CoT routinely says "I cannot clearly hear X, but
# based on Y ..." and still concludes. Matching the whole trace would misclassify those.
_ABSTAIN_PAT = re.compile(
    r"\b("
    r"cannot (?:be )?(?:answer|determin|identif|tell|hear|discern|conclud)"
    r"|can'?t (?:answer|determine|identify|tell|hear)"
    r"|unable to (?:answer|determine|identify|tell|hear|discern)"
    r"|not (?:possible|enough information|sufficient)"
    r"|no (?:audible|discernible|audio|sound|signal|speech)"
    r"|there is no (?:sound|audio|speech|signal)"
    r"|silent|silence"
    r"|none of the (?:\w+\s+){0,2}(?:options|choices|answers|above)"
    r"|insufficient (?:information|audio|evidence)"
    r"|i (?:do not|don'?t) know"
    r"|impossible to (?:determine|answer|tell)"
    # "(n/a)" and "N/A" are refusals, but extract_letter's _PUNCT rule matches the
    # "/a)" inside them and returns option (a). Observed in the Gate A smoke run
    # (job 10461220) on an Emotion item: <Conclusion>(n/a)</Conclusion> -> "a".
    r"|n\s*/\s*a\b|not applicable"
    # Observed phrasings that the "no <noun>" rule above does not reach.
    r"|(?:does|do|did) not contain (?:any )?(?:speech|sound|audio|signal)"
    r"|too short to (?:determine|tell|analyz)"
    r"|no clear (?:emotional |audible )?(?:cue|indication|signal)"
    r")",
    re.I,
)


def detect_abstention(conclusion: str) -> bool:
    """True if the conclusion declines to pick an option."""
    return bool(_ABSTAIN_PAT.search(conclusion))


def parse_outcome(raw: str, options: dict[str, str]) -> tuple[str, str | None]:
    """Classify a raw response as ('letter', 'a') | ('abstain', None) | ('unparseable', None).

    Abstention is checked BEFORE letter extraction and on the conclusion only. Order
    matters: `extract_letter`'s token-overlap fallback will happily map "None of the
    options fit" onto some option, so a refusal checked second is silently scored as a
    wrong answer.
    """
    _, conclusion = split_trace(raw)
    probe = conclusion or raw
    if detect_abstention(probe):
        # An explicit "(c)" alongside a hedge means it did answer; the hedge is commentary.
        if not _PAREN.search(probe):
            return "abstain", None
    letter = extract_letter(raw, options)
    if letter is None:
        return "unparseable", None
    return "letter", letter
