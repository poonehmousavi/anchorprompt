"""PHASE 1 — the Checker as a failure-driven prompt optimizer (ProTeGi / PromptAgent style).

The proposal's method figure specifies the Checker as a "failure-driven prompt
optimizer (ProTeGi / PromptAgent-style), black-box, no internals". This is that agent.

ProTeGi (Pryzant et al. 2023, proposal ref [13]) loop, per question category:

  1. EVALUATE a candidate corrective prompt on a dev minibatch with the frozen LALM.
  2. TEXTUAL GRADIENT — an LLM reads the prompt together with the cases it still gets
     wrong and writes a criticism of WHY the prompt failed.
  3. GRADIENT STEP — the LLM edits the prompt in the direction that criticism implies,
     producing several candidates.
  4. PARAPHRASE — Monte-Carlo rewordings of the survivors, for exploration.
  5. BEAM — keep the top-B by dev score. Repeat.

The objective is the proposal's own metric, not accuracy: maximise genuine failures
repaired while penalising damage. A prompt that makes the model hedge can "recover"
items by luck and break others; scoring net repair prices that in.

  score = failures_recovered - damage_on_clean

TWO INVARIANTS, both enforced mechanically here:

  * The optimizer runs on the DEV split and may see intervention identity. Its OUTPUT
    may not. Every candidate is screened by _mentions_perturbation() and rejected if it
    names noise, masking, silence, injected speech, reordered options and so on — a
    prompt that says "ignore the injected second voice" cannot be selected at inference
    from audio + instruction alone, so it is not a legal solution however well it scores.
  * Candidates are scored on clean items too, so damage to the clean case is visible
    inside the optimization loop rather than discovered afterwards.

Usage:
  python -m src.agent_optimize --items data/phase0/items.jsonl \\
      --scored output/characterize/dev.jsonl --track Animal \\
      --rounds 3 --beam 2 --expansions 2 --minibatch 24 --out configs/phase1_prompts.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.data_transfer import GLOBAL_TRACK
from src.interventions import (FAITHFULNESS_VARIANTS, ILLEGAL_VARIANTS,
                               adversarial_variants,
                               audio_variants, render_choices)
from src.parsing import parse_options, parse_outcome
from src.prompts import build_conversation

# A corrective prompt naming the perturbation is illegal: at inference the checker sees
# only audio + instruction, so it cannot know which of these applies.
_PERTURB_WORDS = re.compile(
    # NOTE the trailing (?:e?s)? on the noun group. Without it PLURALS escaped: a real
    # run produced "ignore any surrounding NOISES", which presumes the noise
    # intervention, and the guard passed it because \\bnoise\\b cannot match "noises".
    # Measured: 4 of 7 hand-written leak phrases slipped through before this fix.
    r"\b(noise|noisy|snr|mask(ed|ing)?|silence|silent|corrupt(ed|ion)?|degrad(ed|ation)|"
    r"inject(ed|ion)?|adversarial|perturb(ed|ation)?|distort(ed|ion)?|"
    r"permut(ed|ation)?|shuffl(ed|e)|reorder(ed)?|scrambl(ed|e)|"
    r"second voice|extra voice|overlaid|synthetic speech|tts|artifact|background sound)"
    r"(?:e?s)?\b", re.I)


# A corrective prompt that redirects the OUTPUT is illegal however well it scores. The
# task is a multiple-choice question and the model must still select one given option in
# the required format. Measured 2026-08-25: an accepted candidate -- "tally all instances
# of distinct voices or sounds. Provide a count for each unique sound" -- won on dev while
# pushing the model into free-form enumeration, and some generations dropped the
# <Reasoning>/<Conclusion> format entirely. The question is data; a prompt may support
# answering it, never replace it.
_TASK_OVERRIDE = re.compile(
    r"\b(tally|enumerate|transcribe|summari[sz]e)\b|"
    r"\b(provide|give|report|write|output|return|state)\b[^.]{0,30}\b(count|number|list|"
    r"tally|transcript|description|summary)\b|"
    r"\bcount (each|every|all|the number)\b|"
    r"\blist (all|each|every|the)\b|"
    r"\bdescribe (all|each|every|what)\b|"
    r"\bfor each (unique |distinct )?(sound|voice|speaker|instrument|event)\b", re.I)


# ProTeGi optimizes a general task instruction, not a content-specific one. A candidate
# naming concrete subject matter has fitted the dev sample rather than the task: it was
# selected because the minibatch happened to contain animals, and it cannot help on
# music or speech items. Observed in a real run: "Focus on identifying the exact ANIMAL
# producing each sound" tied for the best score on a pooled minibatch spanning all four
# tracks, and would have been shipped to MMAU/MMAR where most items are not animals.
_TOO_SPECIFIC = re.compile(
    r"\b(animal|dog|cat|bird|rooster|cow|frog|hen|"
    r"instrument|guitar|piano|violin|drum|genre|melody|chord|"
    r"emotion|happy|sad|angry|gender|male|female|man|woman|"
    r"language|english|spanish|french|german|chinese|accent)\b", re.I)


def too_specific(text: str) -> str | None:
    """Return the offending content word if the candidate names concrete subject matter."""
    m = _TOO_SPECIFIC.search(text or "")
    return m.group(0) if m else None


def overrides_task(text: str) -> str | None:
    """Return the offending phrase if the candidate redirects the model's output."""
    m = _TASK_OVERRIDE.search(text or "")
    return m.group(0) if m else None


# Instructing the model to DISREGARD part of what it hears presumes that something
# extraneous is present -- which is intervention knowledge the checker does not have at
# inference, and is wrong on clean audio where nothing should be ignored. Caught after
# the plural fix: "disregard any secondary or background elements such as laughter,
# engines, or speech patterns" scored 79.2 and passed every word-level guard.
_IGNORE_AUDIO = re.compile(
    r"\b(ignore|disregard|filter out|tune out|screen out|set aside|skip|exclude)\b"
    r"[^.]{0,40}\b(secondary|background|surrounding|extraneous|irrelevant|other|"
    r"additional|unrelated|distract\w*|element|part|portion|segment)s?\b", re.I)


def ignores_audio(text: str) -> str | None:
    """Return the offending phrase if the candidate tells the model to ignore audio."""
    m = _IGNORE_AUDIO.search(text or "")
    return m.group(0) if m else None


def mentions_perturbation(text: str) -> str | None:
    m = _PERTURB_WORDS.search(text or "")
    return m.group(0) if m else None


@dataclass
class Candidate:
    text: str
    score: float = float("-inf")
    recovered: int = 0
    damaged: int = 0
    clean_damage: int = 0
    unfaithful: int = 0          # followed a CORRECT injection it could not perceive
    per_condition: dict = field(default_factory=dict)
    origin: str = "seed"
    gradient: str = ""
    history: list = field(default_factory=list)


SEED_PROMPTS = [
    "Listen to the entire recording before answering. Base your answer only on what you "
    "can actually hear, and say so if the recording does not contain enough information.",
]

GRAD_SYSTEM = (
    "You are optimizing a short instruction that is prepended to a question given to an "
    "audio-language model. Your job is to diagnose why the instruction is failing.\n\n"
    "You will see the current instruction and several cases where the model still answered "
    "incorrectly, including its own reasoning.\n\n"
    "Write a short, specific criticism of the instruction: what is it failing to make the "
    "model do? Be concrete about the reasoning error you observe. 3 sentences maximum. "
    "Do not rewrite the instruction yet."
)

EDIT_SYSTEM = (
    "You are optimizing a short instruction prepended to a question given to an "
    "audio-language model.\n\n"
    "Given the current instruction and a criticism of it, write {n} improved versions.\n\n"
    "HARD CONSTRAINTS — a version breaking any of these is discarded:\n"
    "- Never mention noise, masking, silence, corruption, injected or synthetic speech, "
    "extra voices, reordered or shuffled options, or any alteration of the input. The "
    "model reading your instruction will NOT know whether its input was altered, and "
    "will often receive a perfectly clean input. Advice that presumes an alteration is "
    "useless and illegal here.\n"
    "- MUST PRESERVE THE TASK. The model is answering a multiple-choice question and "
    "must still choose exactly ONE of the options it was given, in the required output "
    "format. Never instruct it to count, tally, enumerate, list, describe, transcribe "
    "or summarise anything as its answer. Your instruction helps it choose better; it "
    "never replaces the question.\n"
    "- Must not reduce accuracy on clean, unaltered inputs.\n"
    "- Never tell the model to IGNORE or DISREGARD any part of what it hears. On a "
    "clean input there is nothing to ignore, and it cannot know whether anything "
    "extraneous is present.\n"
    "- Must be GENERAL. Never name concrete subject matter (animals, instruments, "
    "emotions, genders, languages, genres). The same instruction must work for a "
    "question about music, a question about speech and a question about an "
    "environmental sound alike.\n"
    "- Must be a directly actionable instruction, at most 3 sentences.\n"
    "- Do not mention specific recordings, questions, or answer options.\n\n"
    "Output exactly {n} versions, one per line, each on its own line prefixed by "
    "'VERSION: '. No other text."
)

PARA_SYSTEM = (
    "Reword the following instruction {n} different ways, preserving its meaning exactly. "
    "Same hard constraints: the model must still choose exactly ONE of the given "
    "multiple-choice options, so never ask it to count, tally, enumerate, list, "
    "describe or transcribe as its answer; "
    "never mention noise, masking, silence, corruption, injected "
    "speech, extra voices, or reordered options; at most 3 sentences each. "
    "Output exactly {n} rewordings, one per line, each prefixed by 'VERSION: '. No other text."
)


# --------------------------------------------------------------------------- LLM backends
class Optimizer:
    """The agent's language model — writes gradients and edits. Never sees audio."""

    def __init__(self, backend: str, local_model: str):
        self.backend = backend
        self.local_model = local_model
        self._m = self._t = self._client = None

    def _ensure(self):
        if self.backend.startswith("claude"):
            if self._client is None:
                import anthropic
                self._client = anthropic.Anthropic()
            return
        if self._m is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._t = AutoTokenizer.from_pretrained(self.local_model)
            self._m = AutoModelForCausalLM.from_pretrained(
                self.local_model, dtype=torch.float16, device_map="auto")
            self._m.eval()

    def __call__(self, system: str, user: str, max_new_tokens: int = 700) -> str:
        self._ensure()
        if self.backend.startswith("claude"):
            _, _, mid = self.backend.partition(":")
            r = self._client.messages.create(
                model=mid or "claude-sonnet-5", max_tokens=max_new_tokens,
                system=system, messages=[{"role": "user", "content": user}])
            return "".join(b.text for b in r.content if b.type == "text")
        import torch
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self._t.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inp = self._t([text], return_tensors="pt").to(self._m.device)
        with torch.inference_mode():
            out = self._m.generate(**inp, max_new_tokens=max_new_tokens, do_sample=True,
                                   temperature=0.8, top_p=0.95)
        return self._t.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


MIN_PROMPT_CHARS = 20

# Leading decoration the optimizer LLM puts in front of its VERSION label. Qwen emits
# "**VERSION:**" and "1. VERSION:" routinely, and the original strict startswith check
# dropped both SILENTLY -- which does not error, it just returns no candidates, leaves
# the beam unchanged, and reports the hand-written SEED as the "learned" prompt.
_DECORATION = re.compile(r"^[\s>*_#-]*(?:\d+[.)]\s*)?[\s*_]*", re.UNICODE)
_VERSION_LABEL = re.compile(r"^VERSION\s*\d*\s*[:.\)]\s*", re.IGNORECASE)


def parse_versions(text: str, n: int) -> list[str]:
    """Pull candidate prompts out of the optimizer LLM's reply.

    Tolerant of markdown and list formatting, because the failure mode is silent: a
    parser that matches nothing yields an unchanged beam and a "learned" prompt that is
    really the seed. Returns at most n; callers should check for a short result.
    """
    out, dropped_short = [], 0
    for line in text.splitlines():
        line = _DECORATION.sub("", line.strip()).strip()
        line = line.rstrip("*_ ").strip()
        if not _VERSION_LABEL.match(line):
            continue
        v = _VERSION_LABEL.sub("", line).strip().strip('"').strip("*_ ").strip()
        if len(v) >= MIN_PROMPT_CHARS:
            out.append(v)
        elif v:
            dropped_short += 1
    if dropped_short:
        print(f"[agent] parse_versions dropped {dropped_short} candidate(s) under "
              f"{MIN_PROMPT_CHARS} chars", flush=True)
    if not out and text.strip():
        print(f"[agent] WARNING parse_versions matched NOTHING in a "
              f"{len(text)}-char reply. The beam cannot improve while this persists. "
              f"First 200 chars: {text.strip()[:200]!r}", flush=True)
    return out[:n]


# --------------------------------------------------------------------------- scoring
def score_candidate(cand: Candidate, minibatch, clean_ref, model, processor,
                    fmt: str, ordering: str, reasoning: bool, max_new_tokens: int,
                    objective: str = "repair") -> Candidate:
    """Run the frozen LALM over the minibatch with this corrective prompt prepended."""
    from src.model import generate

    recovered = damaged = clean_damage = unfaithful = 0
    by_cond: dict[str, list[int]] = defaultdict(list)
    errors = []
    for r in minibatch:
        stem = r["instruction"].split("(a)")[0].strip() or r["stem"]
        conv = build_conversation(stem, render_choices(r["choices_raw"]), r["audio_path"],
                                  fmt=fmt, ordering=ordering, reasoning=reasoning,
                                  context=cand.text)
        raw = generate(model, processor, conv, max_new_tokens=max_new_tokens)
        kind, letter = parse_outcome(raw, parse_options(r["instruction"]))
        ok = kind == "letter" and letter == r["gold"]
        # adv_correct is EXCLUDED from the accuracy average on purpose: the true answer
        # is spoken aloud there, so higher accuracy is injection-following, not skill.
        # Averaging it in would reward exactly the behaviour the unfaithful term
        # penalises. It is scored separately below.
        if r["variant"] not in FAITHFULNESS_VARIANTS:
            by_cond[r["variant"]].append(int(ok))
        was_ok = clean_ref.get((r["id"], r["hop"], r["variant"]), False)
        if r["variant"] == "clean":
            clean_damage += was_ok and not ok
        elif r["variant"] in FAITHFULNESS_VARIANTS:
            # INVERTED. The true answer is spoken in the injected audio, and this item
            # was WRONG on clean audio (see build_minibatch's unfaithful pool). Getting
            # it right here means the model read the injection rather than the source,
            # so it is a penalty. Scoring it as `recovered` would reward the optimizer
            # for increasing injection susceptibility — the opposite of the goal.
            unfaithful += ok
        else:
            recovered += (not was_ok) and ok
            damaged += was_ok and not ok
        if not ok:
            errors.append({"instruction": r["instruction"], "gold": r["gold"],
                           "gold_text": r["choices_raw"][r["gold"]], "raw": raw})
    cand.recovered, cand.damaged, cand.clean_damage = recovered, damaged, clean_damage
    cand.unfaithful = unfaithful
    cand.per_condition = {v: sum(h) / len(h) for v, h in by_cond.items()}
    # clean damage costs double; unfaithful injection-following is penalised at the same
    # weight as a genuine repair is rewarded, so a prompt cannot buy `recovered` points
    # by making the model attend harder to injected speech.
    if objective == "accuracy":
        # ROBUSTNESS objective: macro-average accuracy over CONDITION TYPES, so clean
        # and each perturbation count equally however many items each contributes. A
        # pooled accuracy would let a prompt win by helping the largest bucket while
        # wrecking a small one -- which is how the generic prompt scored 18.7% repair
        # while costing 18.3 points of clean accuracy.
        #
        # Non-answers count as wrong (`ok` requires a parsed letter), so hedging and
        # abstention are penalised, not rewarded: under the repair objective the
        # generic prompt abstained on 18% of CLEAN items and still scored well.
        n_unf = sum(1 for r in minibatch if r["variant"] in FAITHFULNESS_VARIANTS)
        terms = [100.0 * v for v in cand.per_condition.values()]
        if n_unf:
            # Faithfulness enters as ONE MORE TERM, not as an unbounded subtraction.
            # Subtracting the full 0-100 rate let it dominate every accuracy gain, so a
            # prompt could win by driving adv_correct to zero rather than by being more
            # accurate. As a term it is weighted like any single condition.
            terms.append(100.0 * (1.0 - unfaithful / n_unf))
        cand.score = sum(terms) / len(terms) if terms else float("-inf")
    else:
        # REPAIR objective (proposal Figure 2).
        cand.score = recovered - damaged - 2.0 * clean_damage - unfaithful
    return cand, errors


# --------------------------------------------------------------------------- the loop
def load_seed_prompts(path) -> list[str]:
    """Prompts from a previous curriculum stage, used to warm-start the beam."""
    data = yaml.safe_load(open(path))
    out = []
    for entry in (data.get("prompts") or {}).values():
        out.append(entry if isinstance(entry, str) else entry["content"])
    return [t for t in out if t]


def optimize_track(track, minibatch, clean_ref, model, processor, opt: Optimizer, args) -> dict:
    seeds = list(SEED_PROMPTS)
    if getattr(args, "seed_from", None):
        prior = load_seed_prompts(args.seed_from)
        # Prior stage FIRST: the curriculum keeps what already works and adapts it,
        # rather than restarting the search from the generic seed each stage.
        seeds = prior + seeds
        print(f"[agent] curriculum: warm-starting from {len(prior)} prior prompt(s) "
              f"in {args.seed_from}", flush=True)
    beam = [Candidate(text=t, origin="seed") for t in seeds]
    baseline = Candidate(text="", origin="no-prompt")
    all_seen: list[Candidate] = []

    scored, _ = score_candidate(baseline, minibatch, clean_ref, model, processor,
                                args.fmt, args.ordering, bool(args.reasoning),
                                args.max_new_tokens, args.objective)
    print(f"[agent:{track}] baseline (no corrective prompt): score={scored.score:.1f} "
          f"rec={scored.recovered} dmg={scored.damaged} cleandmg={scored.clean_damage}",
          flush=True)
    all_seen.append(scored)

    for rnd in range(args.rounds):
        t0 = time.time()
        fresh = []
        for cand in beam:
            c, errors = score_candidate(cand, minibatch, clean_ref, model, processor,
                                        args.fmt, args.ordering, bool(args.reasoning),
                                        args.max_new_tokens, args.objective)
            all_seen.append(c)
            if not errors:
                continue
            # --- textual gradient ---
            err_block = "\n\n".join(
                f"Question: {e['instruction']}\nCorrect answer: ({e['gold']}) {e['gold_text']}\n"
                f"Model response: {e['raw'][:600]}" for e in errors[: args.grad_examples])
            grad = opt(GRAD_SYSTEM,
                       f"Current instruction:\n\"{c.text}\"\n\nCases still answered "
                       f"incorrectly:\n\n{err_block}")
            c.gradient = grad.strip()[:800]
            # --- gradient step ---
            edits = parse_versions(
                opt(EDIT_SYSTEM.format(n=args.expansions),
                    f"Current instruction:\n\"{c.text}\"\n\nCriticism:\n{c.gradient}\n\n"
                    f"Write {args.expansions} improved versions."),
                args.expansions)
            # --- paraphrase for exploration ---
            paras = parse_versions(
                opt(PARA_SYSTEM.format(n=args.paraphrases),
                    f"Instruction:\n\"{c.text}\""), args.paraphrases) if args.paraphrases else []
            for t, origin in [(e, "gradient") for e in edits] + [(p, "paraphrase") for p in paras]:
                bad = mentions_perturbation(t)
                if bad:
                    print(f"[agent:{track}] REJECTED candidate naming '{bad}' "
                          f"(would need intervention identity at inference)", flush=True)
                    continue
                bad = ignores_audio(t)
                if bad:
                    print(f"[agent:{track}] REJECTED candidate telling the model to "
                          f"ignore audio ('{bad}') — presumes a distractor it cannot "
                          f"know about, and is wrong on clean input", flush=True)
                    continue
                bad = too_specific(t)
                if bad:
                    print(f"[agent:{track}] REJECTED candidate naming '{bad}' "
                          f"(content-specific; ProTeGi optimizes a general task "
                          f"instruction, and this cannot transfer)", flush=True)
                    continue
                bad = overrides_task(t)
                if bad:
                    print(f"[agent:{track}] REJECTED candidate redirecting the task "
                          f"('{bad}') — the model must still pick one given option",
                          flush=True)
                    continue
                fresh.append(Candidate(text=t, origin=origin, history=c.history + [c.text]))

        for cand in fresh:
            c, _ = score_candidate(cand, minibatch, clean_ref, model, processor,
                                   args.fmt, args.ordering, bool(args.reasoning),
                                   args.max_new_tokens, args.objective)
            all_seen.append(c)
        pool = {c.text: c for c in beam + fresh if c.score > float("-inf")}
        beam = sorted(pool.values(), key=lambda c: c.score, reverse=True)[: args.beam]
        print(f"[agent:{track}] round {rnd+1}/{args.rounds} "
              f"({time.time()-t0:.0f}s) best={beam[0].score:.1f} "
              f"rec={beam[0].recovered} dmg={beam[0].damaged} "
              f"cleandmg={beam[0].clean_damage} | {beam[0].text[:90]}", flush=True)

    best = beam[0]
    # A run where nothing was ever parsed, or where no learned candidate beat a seed,
    # returns the HAND-WRITTEN SEED while looking like a learned result. Say so in the
    # artifact rather than leaving it to be inferred from `origin`.
    learned = [c for c in all_seen if c.origin in ("gradient", "paraphrase")]
    seed_best = max((c.score for c in all_seen if c.origin == "seed"),
                    default=float("-inf"))
    improved = best.origin in ("gradient", "paraphrase") and best.score > seed_best
    if not learned:
        print(f"[agent:{track}] WARNING no candidate was ever generated — the reported "
              f"prompt is the SEED, not a learned one. Check the optimizer backend and "
              f"parse_versions warnings above.", flush=True)
    elif not improved:
        print(f"[agent:{track}] NOTE {len(learned)} candidates were generated but none "
              f"beat the seed (best seed {seed_best:.1f} vs best learned "
              f"{max(c.score for c in learned):.1f}). Reporting the seed.", flush=True)
    return {
        "title": f"Learned corrective prompt for {track}",
        "description": f"ProTeGi-style optimization, {args.rounds} rounds, beam {args.beam}.",
        "improved_over_seed": bool(improved),
        "n_learned_candidates": len(learned),
        "seed_best_score": None if seed_best == float("-inf") else seed_best,
        "content": best.text,
        "score": best.score, "recovered": best.recovered, "damaged": best.damaged,
        "clean_damage": best.clean_damage, "origin": best.origin,
        "baseline_score": all_seen[0].score,
        "n_candidates_scored": len(all_seen),
        "trace": [{"text": c.text[:300], "score": c.score, "origin": c.origin}
                  for c in sorted(all_seen, key=lambda c: c.score, reverse=True)[:10]],
    }


def build_minibatch(pool_fails, pool_cleans, pool_interv_ok, size: int,
                    clean_frac: float, interv_ok_frac: float,
                    pool_unfaithful=(), unfaithful_frac: float = 0.15) -> list:
    """Assemble the optimizer's minibatch from three pools.

    THE CONTROLS ARE NOT PADDING. The score is
        recovered - damaged - 2 * clean_damage
    and each negative term needs items that can actually trigger it:

      clean          items with no intervention. Without them `clean_damage` is
                     always 0 and a prompt that wrecks unperturbed audio scores free.
      unfaithful     adv_correct items the model got WRONG on clean audio but RIGHT
                     once the answer was spoken. Scored with the sign INVERTED, so a
                     prompt cannot buy `recovered` points by making the model attend
                     harder to injected speech.
      intervened-OK  items the intervention did NOT break. Without them every
                     non-clean item has was_ok == False, so `damaged` can never
                     increment and the term is dead. src/report.py counts exactly
                     that damage at evaluation, so omitting it lets a prompt win in
                     training and lose at eval.

    Failures fill whatever remains, since they are the repair target.
    """
    n_clean = min(max(2, int(size * clean_frac)) if pool_cleans else 0, len(pool_cleans))
    n_iok = min(max(2, int(size * interv_ok_frac)) if pool_interv_ok else 0,
                len(pool_interv_ok))
    n_unf = min(max(2, int(size * unfaithful_frac)) if len(pool_unfaithful) else 0,
                len(pool_unfaithful))
    n_fail = max(0, size - n_clean - n_iok - n_unf)
    return (pool_fails[:n_fail] + pool_cleans[:n_clean] + pool_interv_ok[:n_iok]
            + list(pool_unfaithful)[:n_unf])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--scored", type=Path, required=True,
                    help="dev characterization (original arm) — defines genuine failures")
    ap.add_argument("--tracks", nargs="+", default=None)
    ap.add_argument("--variants", nargs="+", default=None,
                    help="intervention variants to optimize against (default: recoverable ones)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--expansions", type=int, default=2)
    ap.add_argument("--paraphrases", type=int, default=1)
    ap.add_argument("--grad-examples", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=24)
    ap.add_argument("--clean-frac", type=float, default=0.20,
                    help="fraction of the minibatch that is CLEAN, to price in damage. "
                         "0.20 not 0.25: with three control pools (clean, intervened-OK, "
                         "unfaithful) the old default left failures at exactly half the "
                         "batch, so the repair target no longer dominated the objective.")
    ap.add_argument("--unfaithful-frac", type=float, default=0.15,
                    help="fraction that is adv_correct AND clean-wrong: the model got it "
                         "right only because the answer was spoken. Scored INVERTED, so "
                         "injection-following is penalised rather than rewarded.")
    ap.add_argument("--interv-ok-frac", type=float, default=0.15,
                    help="fraction that is INTERVENED BUT STILL CORRECT. Without these "
                         "the `damaged` term in the score is structurally always 0, and "
                         "the optimizer cannot see itself breaking items the "
                         "intervention left working — damage that report.py does count.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-from", type=Path, default=None,
                    help="warm-start the beam from a previous stage's learned prompt "
                         "(curriculum: solve clean first, then add perturbations while "
                         "keeping what already works).")
    ap.add_argument("--global-prompt", action="store_true",
                    help="learn ONE prompt pooled across tracks instead of one per track. "
                         "Required for MMAU/MMAR transfer, which have no SAKURA category.")
    ap.add_argument("--objective", default="repair", choices=("repair", "accuracy"),
                    help="repair = proposal Figure 2. accuracy = macro-average accuracy "
                         "across condition types, penalising hedging and clean damage.")
    ap.add_argument("--optimizer", default="local", help="'local' | 'claude' | 'claude:<model>'")
    ap.add_argument("--local-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--fmt", default="lenient")
    ap.add_argument("--ordering", default="audio_text")   # Gate A: beat text_audio 6/6
    ap.add_argument("--reasoning", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="cap minibatch (smoke)")
    ap.add_argument("--out", type=Path, default=Path("configs/phase1_prompts.yaml"))
    args = ap.parse_args()

    items = [json.loads(l) for l in args.items.read_text().splitlines() if l.strip()]
    scored = [json.loads(l) for l in args.scored.read_text().splitlines() if l.strip()]
    ref = {(r["id"], r["hop"], r["variant"]): r["correct"] for r in scored}
    clean_ok = {(r["id"], r["hop"]) for r in scored if r["variant"] == "clean" and r["correct"]}

    # Phase 1 scope: ADVERSARIAL injection only (see interventions.adversarial_variants).
    # Pass --variants explicitly to widen to the full audio set (audio_variants) or to
    # anything else; the default is deliberately the narrow, highest-signal target.
    variants = args.variants or adversarial_variants({r["variant"] for r in items})
    if not variants:
        raise SystemExit("no target variants in the manifest — check --items/--variants")
    print(f"[agent] optimizing against: {variants}", flush=True)

    by_track = defaultdict(list)
    for r in items:
        if r.get("split") != "dev" or r["hop"] != "single":
            continue
        key = (r["id"], r["hop"], r["variant"])
        if r["variant"] == "clean":
            if (r["id"], r["hop"]) in clean_ok:
                by_track[r["track"]].append(r)                # clean control
        elif r["variant"] in variants:
            # genuine failure: correct when clean, wrong under this intervention
            if (r["id"], r["hop"]) in clean_ok and ref.get(key) is False:
                by_track[r["track"]].append(r)
            # intervened but SURVIVED — the damage control (see build_minibatch)
            elif (r["id"], r["hop"]) in clean_ok and ref.get(key) is True:
                by_track[r["track"]].append(r)

    from src.model import load_model, env_report
    print("[agent] env:", json.dumps(env_report()), flush=True)
    model, processor = load_model()
    opt = Optimizer(args.optimizer, args.local_model)
    print(f"[agent] LALM loaded; optimizer backend={args.optimizer}", flush=True)

    # GLOBAL mode learns ONE prompt from a pool spanning every track. Required for
    # cross-benchmark transfer: MMAU/MMAR have no SAKURA category, so a per-track
    # library cannot be keyed on them (src/data_transfer.py sets track="_global" so a
    # per-track lookup fails loudly instead of silently resolving to Animal).
    # ACCURACY objective: sample the minibatch STRATIFIED across (benchmark, variant)
    # instead of from failure pools. Failure-driven pooling is a repair-objective idea --
    # it needs `ref` labels from a characterization pass, which exist for SAKURA but not
    # for MMAU/MMAR, and characterizing those purely to compose a batch would cost ~2.7 h
    # for information the objective never uses.
    if args.objective == "accuracy":
        buckets = defaultdict(list)
        # HONOUR --variants. The stratified sampler previously bucketed every dev row,
        # so a curriculum stage asking for clean-only silently trained on the full
        # perturbation set and stage A was indistinguishable from the unstaged run.
        # `clean` is always kept: it is the accuracy anchor every stage is measured against.
        allowed = set(variants) | {"clean"}
        for r in items:
            if r.get("split") != "dev" or r["variant"] in ILLEGAL_VARIANTS:
                continue
            if r["variant"] not in allowed:
                continue
            buckets[(r.get("benchmark", "sakura"), r["variant"])].append(r)
        if not buckets:
            raise SystemExit(f"no dev rows for variants {sorted(allowed)}")
        rng = random.Random(args.seed)
        for v in buckets.values():
            rng.shuffle(v)
        keys = sorted(buckets)
        mb, i = [], 0
        while len(mb) < args.minibatch and any(buckets[k] for k in keys):
            k = keys[i % len(keys)]
            if buckets[k]:
                mb.append(buckets[k].pop())
            i += 1
        print(f"[agent] STRATIFIED minibatch {len(mb)}: "
              f"{dict(Counter((r.get('benchmark','sakura'), r['variant']) for r in mb))}",
              flush=True)
        by_track = {GLOBAL_TRACK: mb}
        args.tracks = [GLOBAL_TRACK]
        args.global_prompt = False        # already pooled
        out = {GLOBAL_TRACK: optimize_track(GLOBAL_TRACK, mb, ref, model, processor,
                                            opt, args)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(yaml.safe_dump(
            {"method": "robustness prompt optimization (accuracy objective, diverse data)",
             "optimizer": args.optimizer, "rounds": args.rounds, "beam": args.beam,
             "objective": args.objective, "variants_optimized_against": variants,
             "prompts": out}, sort_keys=False, width=92))
        print(f"\n[agent] wrote {args.out}")
        return

    if args.global_prompt:
        merged = [r for rs in by_track.values() for r in rs]
        by_track = {GLOBAL_TRACK: merged}
        args.tracks = [GLOBAL_TRACK]
        print(f"[agent] GLOBAL mode: one prompt over {len(merged)} pooled items "
              f"from {len(set(r['track'] for r in merged))} tracks", flush=True)

    out = {}
    for track in (args.tracks or sorted(by_track)):
        pool = by_track.get(track, [])
        cleans = [r for r in pool if r["variant"] == "clean"]
        fails = [r for r in pool if r["variant"] != "clean"
                 and r["variant"] not in FAITHFULNESS_VARIANTS
                 and ref.get((r["id"], r["hop"], r["variant"])) is False]
        interv_ok = [r for r in pool if r["variant"] != "clean"
                     and r["variant"] not in FAITHFULNESS_VARIANTS
                     and ref.get((r["id"], r["hop"], r["variant"])) is True]
        unfaithful = [r for r in pool if r["variant"] in FAITHFULNESS_VARIANTS]
        mb = build_minibatch(fails, cleans, interv_ok, args.minibatch,
                             args.clean_frac, args.interv_ok_frac,
                             unfaithful, args.unfaithful_frac)
        if args.limit:
            mb = mb[: args.limit]
        if len(mb) < 4:
            print(f"[agent] {track}: only {len(mb)} minibatch items — skipping", flush=True)
            continue
        n_f = sum(1 for r in mb if r in fails)
        n_c = sum(1 for r in mb if r["variant"] == "clean")
        print(f"\n[agent] === {track} === minibatch {len(mb)} "
              f"({n_f} failures + {n_c} clean + {len(mb)-n_f-n_c} intervened-OK)",
              flush=True)
        out[track] = optimize_track(track, mb, ref, model, processor, opt, args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(
        {"method": "ProTeGi-style failure-driven prompt optimization (proposal Phase 1)",
         "optimizer": args.optimizer, "rounds": args.rounds, "beam": args.beam,
         "variants_optimized_against": variants,
         "prompts": out}, sort_keys=False, width=92))
    print(f"\n[agent] wrote {args.out} with {len(out)} learned prompts")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------- presupposition guard
# What the invariant forbids is a prompt that only makes sense if the perturbation is
# already known to be present. That is a property of the CLAIM, not of the vocabulary.
#
#   "ignore the injected second voice"                    presupposes one exists -> ILLEGAL
#   "the recording is noisy, so listen harder"            asserts it is         -> ILLEGAL
#   "consider the possibility of degraded audio"          asserts nothing       -> LEGAL
#   "if parts of the recording are unclear, say so"       conditional           -> LEGAL
#
# The distinction matters because the vocabulary rule is both too strict and too loose.
# Too strict: it rejects every hedged insight the fixed reflection framing produces (8/18
# from qwen2.5-7b, 10/70 from mistral-24b, jobs 10526261/10526262), which are applied
# identically to clean items and therefore assert nothing about any one item -- their cost
# is clean collateral, which we already measure. Too loose: it never matched the old
# pool's "verify the INSERTED misleading claims", which presupposes exactly as much as the
# phrases it does catch.
_HEDGES = re.compile(
    r"\b(may|might|could|possib\w+|potential\w*|perhaps|maybe|if|whether|when|unless|"
    r"in case|any|some|uncertain\w*|ambigu\w+|risk of|chance of|consider)\b", re.I)

# Directives to discard part of the input. Illegal even when hedged: on a clean item there
# is nothing extraneous to drop, so the instruction can only do harm.
_DISCARD = re.compile(
    r"\b(ignore|disregard|discount|filter out|screen out|set aside|tune out|"
    r"strip out|exclude)\b", re.I)

# What is being discarded. The discard rule must NOT depend on the perturbation
# vocabulary: "filter out the added speech" is illegal and contains no listed term, so a
# vocabulary-gated check misses it (it did). It must also not fire on ordinary reasoning
# about the OPTIONS -- "exclude implausible options" is exactly what we want the model to
# do -- so the object has to be audio.
_AUDIO_OBJECT = re.compile(
    r"\b(sound|sounds|speech|voice|voices|noise|noises|audio|recording|clip|segment|"
    r"segments|part|parts|element|elements|background|utterance|utterances)\b", re.I)


def presupposes_perturbation(text: str) -> str | None:
    """Return the offending phrase if `text` only makes sense when the attack is present.

    A hedged mention is allowed; an assertion or a discard-directive is not. Sentence by
    sentence, because one hedged clause must not license an asserted one elsewhere.
    """
    for sentence in re.split(r"(?<=[.;!?])\s+", (text or "").strip()):
        # Discarding audio is illegal on its own terms, whatever it is called: on an
        # untouched item there is nothing extraneous to drop, so the instruction can only
        # do harm. Checked BEFORE the vocabulary, which is why it catches "filter out the
        # added speech" -- a phrase containing no listed perturbation term.
        if (d := _DISCARD.search(sentence)) and _AUDIO_OBJECT.search(sentence):
            return f"{d.group(0)} (discards audio)"
        hit = _PERTURB_WORDS.search(sentence)
        if not hit:
            continue
        if not _HEDGES.search(sentence):
            return f"{hit.group(0)} (asserted, not hedged)"
    return None


# A note on "the answer options may have been reordered, so re-read them", which
# `mentions_perturbation` rejects and this guard does not. Both are right, for different
# arms. It presupposes nothing -- it is hedged, and re-reading the options is harmless on
# an untouched item -- so it clears the presupposition bar. What is wrong with it is that
# the model CANNOT ACT on it: a permuted choice list is indistinguishable from an original
# one, so "may have been reordered" is only meaningful to someone who knows permutation is
# in the experiment. That is unactionability, which `pool._inactionable` already covers,
# not a presupposition. Keeping the two separate stops one guard from being stretched to
# do the other's job badly.


def discards_audio(text: str) -> str | None:
    """Return the offending phrase if `text` tells the model to throw away audio content.

    Separate from every leak concern, and enforced under all leak policies. Naming the
    perturbation can be earned from the signal; discarding part of the recording cannot be
    justified that way, because on an untouched item -- the majority of any deployment --
    there is nothing extraneous to drop and the instruction can only do harm.
    """
    for sentence in re.split(r"(?<=[.;!?])\s+", (text or "").strip()):
        if (d := _DISCARD.search(sentence)) and _AUDIO_OBJECT.search(sentence):
            return d.group(0)
    return None
