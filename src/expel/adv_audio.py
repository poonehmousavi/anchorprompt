"""SAKURA adversarial audio injection: a TTS voice speaking an answer, mixed in.

Joining the manifest to SAKURA is the whole difficulty. The manifest id
(`sakura_animal_0_single`) is NOT positional into metadata.json -- joining that way
agrees on 2-11% of choice sets. What identifies an item is its CONTENT: (hop, question,
choice set) is 495-distinct across 500 Animal items on both sides, and the residual
collisions are genuine duplicates within SAKURA itself.

The join is verified, not assumed. `build_index` checks that the manifest's correct
answer TEXT equals SAKURA's gold text for every joined item, because the two sides
render choices in different orders and different case, so agreeing on a letter would
prove nothing.

Only the audio changes. SAKURA's own instruction and choices go to the model untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

MANIFEST_ROOT = Path("data/variants/audio_variants")
ADV_ROOT = Path("data/variants/audio_variants/data/adversarial_aug_data")


def _norm(s: str) -> str:
    return " ".join(s.lower().split()).strip(" .?!")


def _key(question: str, choice_texts, hop: str) -> tuple:
    """Order-SENSITIVE. A set of option texts is not an identity: many SAKURA items share
    both the question and the option set and differ only in which one is correct, so a
    frozenset key silently matched a dog item to a cat row -- 37 of 240 with a gold that
    disagreed. The order the options are rendered in is what distinguishes them."""
    return (hop, _norm(question), tuple(_norm(c) for c in choice_texts))


def _gold_text(m: dict) -> str:
    """The manifest's correct option text, from its own letter into its own list."""
    letter = (m.get("true_letter") or "").strip().lower()
    if letter and letter in "abcd" and len(m.get("choices_list", [])) > "abcd".index(letter):
        return _norm(m["choices_list"]["abcd".index(letter)])
    ans = (m.get("answer") or "").strip()
    return _norm(ans[3:] if ans.startswith("(") and len(ans) > 3 else ans)


def build_index(track: str, mode: str = "wrong", root: Path = MANIFEST_ROOT) -> dict:
    """{content key -> adv wav path} for one track. Raises if the gold disagrees."""
    path = root / track.lower() / "adv_manifests" / f"manifest_adv_{mode}.jsonl"
    index: dict = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        hop = "single" if m["id"].endswith("_single") else "multi"
        wav = ADV_ROOT / f"{track.lower()}_concat" / m["id"] / f"{mode}.wav"
        index.setdefault(_key(m["question"], m.get("choices_list", []), hop),
                         []).append((str(wav), _gold_text(m)))
    return index


def resolve(index: dict, task) -> tuple[str, str] | None:
    """(wav path, manifest gold text) for this task, or None if it did not join.

    Some keys still collide -- SAKURA contains items identical in question and rendered
    option order. Taking the first row there attached the wrong recording to 8.8% of
    items with no error raised. The item's own gold text disambiguates: it is data-prep
    metadata, not something the model sees, and a collision that cannot be resolved is
    reported as a MISS rather than guessed at.
    """
    hits = index.get(_key(task.stem, task.choices.values(), task.hop))
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    want = _norm(task.choices[task.gold])
    exact = [h for h in hits if h[1] == want]
    return exact[0] if exact else None


class AdvAudioResolver:
    """Lazily builds one index per (track, mode) and checks the gold on every lookup."""

    def __init__(self, mode: str = "wrong"):
        self.mode = mode
        self._by_track: dict = {}
        self.misses: list = []
        self.gold_mismatches: list = []

    def __call__(self, task) -> str:
        if task.track not in self._by_track:
            self._by_track[task.track] = build_index(task.track, self.mode)
        hit = resolve(self._by_track[task.track], task)
        if hit is None:
            self.misses.append(task.id)
            raise KeyError(f"no adv_{self.mode} audio joined for {task.id}")
        wav, gold_text = hit
        # SAKURA's gold text must equal the manifest's. Letters differ between the two
        # renderings, so matching on a letter would agree even when the items do not.
        if gold_text and _norm(task.choices[task.gold]) != gold_text:
            self.gold_mismatches.append(task.id)
            raise ValueError(
                f"adv_{self.mode} gold disagrees for {task.id}: "
                f"sakura={task.choices[task.gold]!r} manifest={gold_text!r}")
        return wav
