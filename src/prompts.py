"""Prompt formats for Qwen2.5-Omni on SAKURA, plus the corrective-prompt injection hook.

Three formats exist across the prior repos and they disagree with each other. Gate A
sweeps them on clean data before anything else is built, because a wrong base format
makes every later intervention delta a measurement against a broken baseline.

  strict   listen-to-reason/src/eval_lib.py SYSTEM_STRICT  — official Omni identity +
           <Conclusion>[Letter]</Conclusion>. Cleanest extraction, but eval_lib's own
           comment records it inflating REFUSALS on hard tracks (Emotion): the model
           says "none of the options fit" rather than guessing.
  lenient  eval_lib.py SYSTEM_LENIENT — "respond with ONLY the letter".
  faith    faith_new/core/qwen_omni_utils.py:200 — "Question: .. Choices: .." layout.

Ordering: eval_lib's build_conversation docstring specifies "system identity +
(text-then-audio) user turn", but that was never tested against audio-then-text here,
so it is swept rather than inherited.

`context` is the injection point for a corrective prompt at evaluation time. It is
prepended to the user text and is the ONLY channel through which a corrective prompt
reaches the model.
"""
from __future__ import annotations

BASE_IDENTITY = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable "
                 "of perceiving auditory and visual inputs.")

SYSTEM_STRICT = (f"{BASE_IDENTITY}\n\n"
                 "CRITICAL: Respond ONLY with the conclusion tag. No conversational text.\n"
                 "<Conclusion>\n[Letter]\n</Conclusion>")

SYSTEM_LENIENT = ("You are an expert at reasoning about sounds, speech, and the things that "
                  "produce them. Answer multiple-choice questions about audio. Respond with ONLY "
                  "the letter (a, b, c, or d) of the correct option.")

# Reasoning variants. The experiment needs a CoT trace: the prompt inducer has no strategy
# to distill from a bare letter, and the CoT-consistency metric needs the trace itself.
_COT_TAIL = ("\n\nThink step by step about what you hear, then give your final choice.\n"
             "Respond in exactly this format:\n"
             "<Reasoning>\nyour step-by-step reasoning\n</Reasoning>\n"
             "<Conclusion>\n(letter)\n</Conclusion>")

SYSTEM_STRICT_COT = f"{BASE_IDENTITY}{_COT_TAIL}"
SYSTEM_LENIENT_COT = ("You are an expert at reasoning about sounds, speech, and the things that "
                      "produce them. Answer multiple-choice questions about audio."
                      f"{_COT_TAIL}")


def _system(fmt: str, reasoning: bool) -> str:
    if fmt == "strict":
        return SYSTEM_STRICT_COT if reasoning else SYSTEM_STRICT
    if fmt == "lenient":
        return SYSTEM_LENIENT_COT if reasoning else SYSTEM_LENIENT
    if fmt == "faith":
        return SYSTEM_STRICT_COT if reasoning else BASE_IDENTITY
    if fmt == "faith_repl":
        return SYSTEM_FAITH_REPL_COT if reasoning else SYSTEM_STRICT
    raise ValueError(f"unknown format {fmt!r}")


def _user_text(fmt: str, stem: str, choices: str, reasoning: bool) -> str:
    if fmt == "strict":
        return f"{stem} Select one option from the provided choices.\n{choices}."
    if fmt == "lenient":
        tail = "" if reasoning else "\n\nAnswer with only the letter (a, b, c, or d)."
        return f"{stem}\n{choices}{tail}"
    if fmt == "faith":
        return f"Question: {stem}\n Select one option from the provided choices. Choices:\n{choices}"
    if fmt == "faith_repl":
        tail = ("\nPlease think and reason about the input audio before you respond "
                "using the XML template.") if reasoning else ""
        return f"{stem} Select one option from the provided choices.\n{choices}.{tail}"
    raise ValueError(f"unknown format {fmt!r}")


# Exact replication of the pipeline that produced the reference numbers.
#
# CAREFUL: faith_new contains THREE disagreeing Qwen-Omni prompt implementations.
#   core/qwen_omni_utils.py:195-219                      -> kept here as `faith`
#   pooneh_version/interface/qwen_2.5_pmni_wrapper)old.py -> superseded, no system role
#   pooneh_version/interface/qwen_2.5_omni_wrapper.py:88-130  <- THIS ONE
#
# Which one produced the reference tables, established from the repo rather than
# assumed (they differ in user text, system tail, and choice-letter case, and picking
# wrong measures our baseline against a prompt that never produced those numbers):
#   * core/ is driven by main.py, which writes to config.RESULTS_DIR="./results"
#     -> faith_new/results/qwen_omni/ contains ZERO files. It never ran to completion.
#   * the readme's own analysis scripts (pooneh_version/plot_adv.py:235,
#     audio_mask_noise_vis.py:233) default to --base_dir pooneh_version/result/baseline,
#     and result/baseline/qwen2.5/<track>/adv/{correct,wrong}/ exists with exactly the
#     layout the readme's adversarial tables report. That tree is written by the
#     interface/ wrapper (readme Step 3: --output result/.../baseline_{name}_REAS.jsonl).
#   * ")old" (2026-02-27 13:09) has no system role and a "Template:" phrasing; it was
#     replaced 6 h later by this file (19:32, "# --- UPDATED PROMPTING ---"). The readme
#     is 2026-03-03 16:11, after both.
# The result .jsonl files themselves were lost to a scratch purge (empty dir skeletons),
# so the comparison is aggregate-to-aggregate and this provenance is the evidence.
# Verified byte-for-byte against that file:
#   system  == SYSTEM_STRICT (no-CoT) / SYSTEM_FAITH_REPL_COT (CoT)
#   user    == the `strict` layout
#   choices == UPPERCASE, so faith_repl must be run with --upper-choices
#   ordering == TEXT-then-AUDIO, max_new_tokens == 128 (no-CoT) / 1024 (CoT)
# Gate A showed text_audio is the WORSE ordering — that is the point: to check our
# baseline against their numbers we must reproduce their setup, not improve on it.
SYSTEM_FAITH_REPL_COT = (
    f"{BASE_IDENTITY}\n\n"
    "CRITICAL: You must provide your analysis in a structured format using XML tags. "
    "Do not engage in conversational filler. Use the following structure:\n"
    "<Reasoning>\n[Describe the acoustic features and your logic]\n</Reasoning>\n"
    "<Conclusion>\n[Single Letter Only]\n</Conclusion>")

FORMATS = ("strict", "lenient", "faith", "faith_repl")
ORDERINGS = ("text_audio", "audio_text")


def build_conversation(stem: str, choices: str, audio_path: str, *,
                       fmt: str = "strict", ordering: str = "text_audio",
                       reasoning: bool = True, context: str | None = None) -> list[dict]:
    """Qwen2.5-Omni conversation. `context` is the corrective-prompt injection point.

    The Omni identity system prompt is mandatory — without it the model hallucinates
    "Human: ..." dialogue chains (documented in alas/scripts/get_qwen_omni_hiddenstates.py).
    """
    if ordering not in ORDERINGS:
        raise ValueError(f"unknown ordering {ordering!r}")
    text = _user_text(fmt, stem, choices, reasoning)
    if context:
        text = f"{context.rstrip()}\n\n{text}"
    audio_part = {"type": "audio", "audio": audio_path}
    text_part = {"type": "text", "text": text}
    content = [text_part, audio_part] if ordering == "text_audio" else [audio_part, text_part]
    return [
        {"role": "system", "content": [{"type": "text", "text": _system(fmt, reasoning)}]},
        {"role": "user", "content": content},
    ]


# --------------------------------------------------------------------------- arms
# The generic arm must be a genuinely reasonable prompt, not a straw man: ExpeL's
# hand-crafted insights recovered a third of the gap over its baseline unaided
# (28.0 -> 32.0 vs 39.0 learned). A weak generic arm manufactures a positive result.
GENERIC_PROMPT = (
    "Be careful. Listen to the whole recording attentively before answering. "
    "Base your answer only on what you actually hear, not on what the question or the "
    "answer options suggest you should hear. If the recording does not contain enough "
    "information to answer, say so instead of guessing."
)
