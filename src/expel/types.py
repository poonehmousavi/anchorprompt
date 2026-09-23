"""Core records for the ExpeL pipeline. Plain dataclasses, JSON-round-trippable."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Task:
    """One SAKURA question. `condition` is 'clean' or an attack name."""
    id: str                       # "Animal/dog28:single"
    track: str                    # Animal | Emotion | Gender | Language
    hop: str                      # single | multi
    audio_path: str
    stem: str                     # question without the choices
    choices: dict                 # {"a": "dog", ...} original casing
    gold: str                     # "a"
    condition: str = "clean"
    # False when the intervention destroyed the evidence (mask_100, noise_-20dB). The
    # faithful behaviour is then to DECLINE, so these are scored by abstention, not
    # accuracy. Training-side only: the actor is never told which regime it is in -- the
    # audio fingerprint is what routes it. See src/expel/retrieval.py.
    answerable: bool = True
    # What the attack did. Training-side telemetry only: never rendered into a prompt.
    attack_meta: dict = field(default_factory=dict)

    @property
    def choices_block(self) -> str:
        return " ".join(f"({L}) {t}" for L, t in sorted(self.choices.items()))


@dataclass
class Trajectory:
    """One actor rollout."""
    task_id: str
    track: str
    hop: str
    condition: str
    answerable: bool
    family: str                    # attack family; the pool scope this rollout belongs to
    context: str                  # knowledge injected into the prompt ("" if none)
    raw: str                      # decoded model output
    reasoning: str
    outcome: str                  # letter | abstain | unparseable
    pred: str | None
    gold: str
    correct: bool
    retrieved: list = field(default_factory=list)   # insight ids used, for credit assignment
    # Two-pass (CorrectionActor) only: what the model said it could actually hear, before
    # it saw the options. Kept so a wrong answer can be traced to a wrong description --
    # the failure mode of a correction stage is that pass 2 reasons confidently from a
    # confabulated pass 1, which is invisible if only the final letter is stored.
    evidence: str = ""
    # Routed arm only: the family the ROUTER predicted, never the true one. Stored so
    # routing errors can be counted per item against the label held elsewhere.
    route: str | None = None
    # The decoded output BEFORE stop-string truncation, verbatim. `raw` is what the parser
    # saw; this is what the model said. Kept so a parsing bug can be fixed by re-parsing
    # the rows instead of re-generating them.
    raw_full: str = ""

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Insight:
    """One entry in the knowledge pool. `count` is ExpeL's importance counter."""
    id: str
    scope: str                    # "<track>|<attack_family>", or "*|<family>" for cross-track
    text: str
    count: int = 2                # ADD starts at 2 so a single DOWNVOTE cannot erase it
    origin: str = "reflection"    # reflection | seed
    n_add: int = 1
    n_up: int = 0
    n_down: int = 0
    n_edit: int = 0
    # Mean (audio, instruction) key of the failures this insight was learned from. What
    # similarity retrieval matches against; empty for scope-only insights.
    centroid: list = field(default_factory=list)
    n_keys: int = 0

    def to_json(self) -> dict:
        return asdict(self)

    def absorb(self, key) -> None:
        """Move the centroid toward another failure's key (running mean)."""
        if key is None or not len(key):
            return
        k = [float(x) for x in key]
        if not self.centroid:
            self.centroid, self.n_keys = k, 1
            return
        n = self.n_keys + 1
        self.centroid = [(c * self.n_keys + v) / n for c, v in zip(self.centroid, k)]
        self.n_keys = n


@dataclass
class PoolOp:
    """An edit proposed by the reflector and applied by the pool."""
    op: str                       # ADD | UPVOTE | DOWNVOTE | EDIT
    scope: str = ""
    insight_id: str | None = None
    text: str | None = None
    reason: str = ""
    key: list = field(default_factory=list)   # (audio, instruction) key of the failure
    applied: bool = False
    rejected: str = ""            # non-empty when a guard refused it

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class ContrastPair:
    """The unit of learning: the same task, clean vs attacked."""
    clean: Trajectory
    attacked: Trajectory

    @property
    def is_attack_induced_failure(self) -> bool:
        """Correct on clean audio, wrong once the text was injected."""
        return self.clean.correct and not self.attacked.correct

    @property
    def is_resisted(self) -> bool:
        """Correct on both: evidence that whatever knowledge was in play worked."""
        return self.clean.correct and self.attacked.correct
