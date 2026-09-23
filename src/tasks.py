"""A task taxonomy that spans SAKURA, MMAU and MMAR.

The checker keyed corrective prompts on SAKURA's four tracks, which cannot transfer:
MMAU and MMAR have no Animal/Emotion/Gender/Language, so infer_track would map every
item onto some SAKURA track and apply an unrelated prompt — producing a meaningless
number rather than an error. This module replaces that key with a taxonomy all three
benchmarks share.

    speech    the answer is in what a voice IS or SAYS — speaker attributes, language,
              emotion, transcribed content
    sound     the answer is in a non-speech event or environment
    music     the answer is about musical content — instrument, genre, key, tempo
    temporal  the answer is about ORDER, COUNT or TIME rather than identity. Cuts across
              the other three, and takes priority: "how many times does the dog bark"
              is a counting question whose difficulty is counting, not animal ID.

GOLD vs INFERRED. Every benchmark labels tasks in its own metadata, and those labels are
TRAIN-SIDE ONLY — used to stratify dev pools and to score how well inference works. At
inference the checker sees audio + instruction only, so it must call infer_task() on the
instruction text. Keeping the two apart is the same invariant as src/checker.py: the
gold label must never reach the inference path.

Coverage note: SAKURA supplies speech and sound only. It has no music and no
temporal/counting questions, so prompts for those tasks cannot be learned from SAKURA
and require dev items from MMAU/MMAR.
"""
from __future__ import annotations

import re

TASKS = ("speech", "sound", "music", "temporal")

# Temporal is tested FIRST: it cuts across the other three and describes the reasoning
# the prompt has to support, which matters more than the audio's modality.
# TEMPORAL must be NARROW. An earlier version matched first/last/before/after/start/
# end/order/next, which are ordinary English: it swallowed 235 of 333 MMAU speech items
# and drove overall inference to 48%. Only phrases whose presence really does make the
# question about order, count or duration belong here.
_TEMPORAL = re.compile(
    r"(how many|how much longer|number of (times|occurrences)|how often|"
    r"\bcount(ing|ed)?\b|occurrences? of|"
    r"in (what|which) order|what order|which (one )?(comes|came|happens|happened) "
    r"(first|last|before|after)|"
    r"(before|after) the|precede[sd]?|follow(s|ed) the|"
    r"how long|duration of|total (time|length))", re.I)

_MUSIC = re.compile(
    r"\b(music|musical|instrument|guitar|piano|violin|drum|melody|chord|"
    r"key|tempo|beat|rhythm|genre|song|singer|sing(s|ing)?|band|orchestra|"
    r"pitch of the note|harmon(y|ic)|major|minor|bpm)\b", re.I)

_SPEECH = re.compile(
    r"\b(speak(er|s|ing)?|voice|said|says|saying|utter(s|ance|ed)?|word(s)?|"
    r"sentence|language|accent|dialect|emotion|feel(s|ing)?|mood|tone of the speaker|"
    r"gender|man|woman|male|female|child|conversation|dialogue|transcri|"
    r"pronounc|talk(s|ing)?)\b", re.I)

_SOUND = re.compile(
    r"\b(sound|noise|animal|environment|scene|event|sourc(e|es)|"
    r"heard|hear(s|ing)?|acoustic|audio clip|recording contains|"
    r"bark|meow|siren|engine|door|water|rain|wind|bird|dog|cat)\b", re.I)


def infer_task(instruction: str, choices: dict[str, str] | None = None) -> str:
    """Predict the task type from the INSTRUCTION alone (plus choices as a tiebreak).

    Inference-safe: reads only what the checker is allowed to see. Never raises — an
    unmatched item falls back to `sound`, the broadest category, rather than to a
    benchmark-specific default that would bias one corpus.
    """
    text = instruction or ""
    if _TEMPORAL.search(text):
        return "temporal"
    scores = {
        "music": len(_MUSIC.findall(text)),
        "speech": len(_SPEECH.findall(text)),
        "sound": len(_SOUND.findall(text)),
    }
    if choices:
        blob = " ".join(str(t) for t in choices.values())
        scores["music"] += len(_MUSIC.findall(blob))
        scores["speech"] += len(_SPEECH.findall(blob))
        scores["sound"] += len(_SOUND.findall(blob))
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] else "sound"


# --------------------------------------------------------------- gold labels (TRAIN ONLY)
_SAKURA_TASK = {"Animal": "sound", "Emotion": "speech",
                "Gender": "speech", "Language": "speech"}


def gold_task_sakura(track: str) -> str:
    return _SAKURA_TASK.get(track, "sound")


def gold_task_mmau(group: str) -> str:
    """MMAU labels every item music / sound / speech."""
    g = (group or "").lower()
    return g if g in ("music", "sound", "speech") else "sound"


def gold_task_mmar(modality: str, sub_category: str) -> str:
    """MMAR gold, keyed on SUB-CATEGORY first.

    Sub-category describes the TASK; modality describes the audio. Keying on modality
    put every `mix-sound-speech` item under speech even when the question was about the
    sound, which made the label disagree with the question for ~240 items. The task is
    what a corrective prompt has to serve, so it wins.
    """
    sub = (sub_category or "").lower()
    if "temporal" in sub or "counting" in sub:
        return "temporal"
    if "music" in sub:
        return "music"
    if any(k in sub for k in ("speaker", "emotion", "culture", "content analysis")):
        return "speech"
    if any(k in sub for k in ("environmental", "anomaly", "acoustic quality")):
        return "sound"
    m = (modality or "").lower()
    if m == "speech":
        return "speech"
    if m == "music":
        return "music"
    if m == "sound":
        return "sound"
    return "sound"
