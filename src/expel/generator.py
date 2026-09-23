"""The text model that WRITES the insights. Never the actor, never scored.

RoP has GPT-4o author the instructions and GPT-3.5-turbo execute them
(arXiv:2506.03627). We had Qwen2.5-Omni reflecting on itself, and the four insights that
survived are vacuous -- "Always verify the emotional tone of the speaker and ensure it
aligns with the given options" has the worst net of any insight in the pool (-27). A
weak author is the most likely cause of the Phase 2 null, so the author becomes a knob.

THE ACTOR STAYS FROZEN QWEN2.5-OMNI. This model never sees audio, never answers a
benchmark question, and never touches the actor's internals, so the black-box claim is
unchanged. What DOES change is the claim's wording: with a separate author it is "a text
LLM writes prompts for a frozen LALM" -- RoP's own arrangement -- not "the model repairs
itself". Keep `qwen-omni` runnable so both can be reported.

WHY THIS IS A SEPARATE PROCESS FROM THE ACTOR. Reflection is text-only, so co-loading the
audio model for it is pure waste, and the arithmetic forbids it anyway: Mistral-24B is
45 GB of weights and Qwen2.5-Omni needs 19 GB, which does not fit the 48 GB tier. Reading
trajectories off disk and reflecting in a second job (src/expel/reflect_offline.py) costs
nothing and lets the author be chosen independently of the card.

Availability was checked, not assumed -- `meta-llama/Meta-Llama-3-70B-Instruct` is in the
cache as 21 KB of metadata with no weights, so it is NOT usable here.

    mistral-24b   45 GB on disk, ~48 GB bf16 -> needs the 80 GB tier ALONE
    qwen2.5-7b    15 GB on disk, ~15 GB bf16 -> fits the 48 GB tier; the cheap control
    qwen-omni     the incumbent: the actor reflecting on itself, for comparison
"""
from __future__ import annotations

GENERATORS = {
    "mistral-24b": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}

# Weights only, from `du -sh` on the HF cache. Used to fail loudly at submit time rather
# than after a model load has already burned the allocation.
APPROX_VRAM_GB = {"mistral-24b": 48, "qwen2.5-7b": 15, "qwen-omni": 19}


class HFTextGenerator:
    """A cached instruction-tuned text model, called as `gen(prompt) -> str`."""

    def __init__(self, name: str, max_new_tokens: int = 256, dtype=None):
        if name not in GENERATORS:
            raise ValueError(f"unknown generator {name!r}; have {sorted(GENERATORS)}")
        self.name = name
        self.model_id = GENERATORS[name]
        self.max_new_tokens = max_new_tokens
        self._dtype = dtype
        self.model = self.tokenizer = None

    def load(self) -> "HFTextGenerator":
        """Load by trying the causal-LM head, then the multimodal one.

        `Mistral-Small-3.1-24B-Instruct` is a VISION-language model: its config is
        `Mistral3Config` and `AutoModelForCausalLM` refuses it outright
        (`Unrecognized configuration class ... for this kind of AutoModel`, job 10526203).
        We only ever feed it text, exactly as we use Qwen2.5-Omni's thinker without audio,
        so the image tower is dead weight but harmless. Trying both heads keeps one code
        path for plain text models (Qwen2.5-7B) and multimodal ones.
        """
        from transformers import AutoTokenizer

        from src.model import pick_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        kwargs = dict(dtype=self._dtype or pick_dtype(), device_map="auto",
                      attn_implementation="sdpa")
        errors = []
        for loader in self._auto_classes():
            try:
                self.model = loader.from_pretrained(self.model_id, **kwargs)
                break
            except (ValueError, KeyError) as e:      # wrong head for this config
                errors.append(f"{loader.__name__}: {str(e)[:120]}")
        else:
            raise RuntimeError(
                f"no AutoModel class could load {self.model_id}:\n  " + "\n  ".join(errors))
        self.model.eval()
        for p in self.model.parameters():        # authors prompts; is never trained
            p.requires_grad_(False)
        return self

    def _as_chat(self, prompt: str) -> str:
        """Apply the tokenizer's chat template, or fall back to a plain instruct format.

        `Mistral-Small-3.1-24B` ships no `tokenizer.chat_template` -- it expects the
        mistral-common tokenizer, and `apply_chat_template` raises (job 10526217). The
        fallback is Mistral's own [INST] format, which the model was instruction-tuned on.
        Falling back is safe for an AUTHOR: the worst case is a slightly off-format prompt
        producing worse insights, which is visible in the insights themselves. It would
        NOT be safe for the actor, whose output format is parsed.
        """
        msgs = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                                      tokenize=False)
        except (ValueError, AttributeError, TypeError) as e:
            if not getattr(self, "_warned_template", False):
                print(f"[generator] {self.model_id} has no chat template ({e}); "
                      "falling back to the [INST] instruct format", flush=True)
                self._warned_template = True
            return f"<s>[INST] {prompt} [/INST]"

    @staticmethod
    def _auto_classes():
        from transformers import AutoModelForCausalLM
        out = [AutoModelForCausalLM]
        try:
            from transformers import AutoModelForImageTextToText
            out.append(AutoModelForImageTextToText)
        except ImportError:
            pass
        return out

    def __call__(self, prompt: str, sample: bool = False, temperature: float = 0.9,
                 seed: int | None = None) -> str:
        """`sample=False` is greedy and reproducible; `sample=True` gives DIVERSE draws.

        APE selects the best of N candidate instructions, which requires the N to differ.
        Greedy decoding returns the SAME string every time, so calling this N times on one
        prompt produced 4 identical candidates and spent 4x the scoring budget choosing
        between copies (job 10530592: all in_ec and all in_opt candidates byte-identical).
        Candidate 0 stays greedy so the run has a deterministic anchor; the rest sample.
        """
        import torch

        if self.model is None:
            self.load()
        text = self._as_chat(prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        kwargs = dict(max_new_tokens=self.max_new_tokens,
                      pad_token_id=self.tokenizer.eos_token_id)
        if sample:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
            if seed is not None:
                torch.manual_seed(seed)
        else:
            kwargs.update(do_sample=False)
        with torch.inference_mode():
            out = self.model.generate(**inputs, **kwargs)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen[0], skip_special_tokens=True).strip()


class OmniTextGenerator:
    """The incumbent: the actor itself, used text-only. Kept as the same-family control.

    Reporting only the strong-author numbers would silently change what the paper claims;
    this is what makes "the model repairs itself" still measurable.
    """
    name = "qwen-omni"

    def __init__(self, model, processor, max_new_tokens: int = 256):
        self.model, self.processor = model, processor
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt: str) -> str:
        from src.expel.extractor import generate_text
        return generate_text(self.model, self.processor, prompt,
                             max_new_tokens=self.max_new_tokens)


class ScriptedGenerator:
    """CPU stand-in. Returns a fixed op list so the plumbing can be exercised with no GPU."""
    name = "scripted"

    def __init__(self, reply: str = "ADD: Base your answer only on what you can hear."):
        self.reply = reply
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.reply


def build_generator(name: str, *, model=None, processor=None, max_new_tokens: int = 256):
    if name == "scripted":
        return ScriptedGenerator()
    if name == "qwen-omni":
        if model is None:
            raise ValueError("qwen-omni generator needs the loaded actor model")
        return OmniTextGenerator(model, processor, max_new_tokens)
    return HFTextGenerator(name, max_new_tokens)
