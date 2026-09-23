"""Frozen LALM loading and generation (Qwen2.5-Omni-7B by default; see MODELS).

Thinker-only (Qwen2_5OmniThinkerForConditionalGeneration): Phase 0 never needs speech
output, and skipping the TTS head saves a few GB of VRAM. Loading follows
alas/scripts/get_qwen_omni_hiddenstates.py and listen-to-reason/src/mmar_eval.py:357-390,
with two deliberate departures:

  attn_implementation="sdpa", not "flash_attention_2". The prior code hardcoded FA2,
  which does not exist on rtx8000 (Turing, sm_75) — currently the only GPU class with
  real free capacity on this cluster.

  dtype auto-selected. torch 2.12.0+cu130 ships sm_75/80/86/90; l40s (sm_89) runs on
  the sm_86 cubin. Turing has no bf16, so fall back to fp16 there rather than failing.

The model is never updated. No optimizer, no gradients — inference_mode throughout.
"""
from __future__ import annotations

import re
import os
import torch

MODEL_ID = "Qwen/Qwen2.5-Omni-7B"

# Frozen LALMs the pipeline can drive. `kind` selects the input-building path in
# `generate`: the two Qwen-Omni models take qwen_omni_utils conversations; Audio Flamingo 3
# takes raw 16 kHz arrays through its own processor. All three use a Qwen2/3 tokenizer, so
# the soft-prompt marker `<|quad_start|>` is a single token in each (id 151650).
MODELS = {
    "qwen2.5-omni": {"id": "Qwen/Qwen2.5-Omni-7B", "kind": "qwen_omni", "vram_gb": 19},
    "qwen3-omni": {"id": "Qwen/Qwen3-Omni-30B-A3B-Instruct", "kind": "qwen_omni", "vram_gb": 60},
    "af3": {"id": "nvidia/audio-flamingo-3-hf", "kind": "af3", "vram_gb": 17},
    # AF-Think: the same weights plus the `think` LoRA adapter shipped in the repo.
    "af3-think": {"id": "nvidia/audio-flamingo-3-hf", "kind": "af3", "vram_gb": 17, "adapter": "think"},
}
DEFAULT_MODEL = os.environ.get("REPROMPT_MODEL", "qwen2.5-omni")


def resolve_model(name: str | None) -> tuple[str, dict]:
    """Registry name (or a raw HF id, treated as qwen_omni) -> (name, spec)."""
    name = name or DEFAULT_MODEL
    if name in MODELS:
        return name, MODELS[name]
    for n, spec in MODELS.items():
        if spec["id"] == name and "adapter" not in spec:
            return n, spec
    raise SystemExit(f"unknown model {name!r}; choose from {sorted(MODELS)}")


def model_kind(model) -> str:
    return getattr(model, "_reprompt_kind", "qwen_omni")


def native_bf16() -> bool:
    """True only for NATIVE bf16 (Ampere sm_80+).

    Do not use torch.cuda.is_bf16_supported(): on torch 2.12 it defaults to
    including_emulation=True and returns True on a Quadro RTX 8000 (Turing, sm_75),
    which has no bf16 hardware. Verified on this cluster:
        capability (7,5) | is_bf16_supported() -> True | (including_emulation=False) -> False
    Selecting bf16 there gives slow emulated math, not a fast path.
    """
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0)[0] >= 8


def pick_dtype() -> torch.dtype:
    return torch.bfloat16 if native_bf16() else torch.float16


def load_model(model_id: str | None = None, dtype: torch.dtype | None = None):
    """Returns (model, processor). Frozen and in eval mode. `model_id` is a registry name
    (`qwen2.5-omni` | `qwen3-omni` | `af3`) or the HF id of one of them; default from
    `REPROMPT_MODEL`. Thinker-only for the Omni models (no speech head)."""
    import transformers as tf
    from transformers import AutoProcessor

    name, spec = resolve_model(model_id)
    dtype = dtype or pick_dtype()
    processor = AutoProcessor.from_pretrained(spec["id"])
    cls = {"qwen2.5-omni": "Qwen2_5OmniThinkerForConditionalGeneration",
           "qwen3-omni": "Qwen3OmniMoeThinkerForConditionalGeneration",
           "af3": "AudioFlamingo3ForConditionalGeneration",
           "af3-think": "AudioFlamingo3ForConditionalGeneration"}[name]
    model = _retry(lambda: getattr(tf, cls).from_pretrained(
        spec["id"], dtype=dtype, device_map="auto", attn_implementation="sdpa"))
    if spec.get("adapter"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, spec["id"], subfolder=spec["adapter"])
        model = model.merge_and_unload()      # plain module again: hooks/embeddings unchanged
    if spec["kind"] == "af3" and dtype != torch.float32:
        # transformers 5.10 keeps AF3's audio `embed_positions` in fp32 (strict list), so
        # under bf16 the encoder adds fp32 positions to bf16 features and the next LayerNorm
        # raises "expected scalar type Float but found BFloat16" (smoke 10747645). The
        # sinusoids live in [-1, 1]; casting the whole model to one dtype is the fix.
        model.to(dtype)
    model.eval()
    for p in model.parameters():          # belt and braces: the base LALM stays frozen
        p.requires_grad_(False)
    model._reprompt_kind = spec["kind"]
    model._reprompt_name = name
    return model, processor


def _retry(fn, attempts: int = 4, wait_s: float = 45.0):
    """Shard reads off the shared filesystem fail transiently under many concurrent
    loads ("does not appear to have a file named model-0000x-of-0000y.safetensors" for a
    file that exists; jobs 10748192, 10751890). Retry a few times before giving up."""
    import time
    last = None
    for i in range(attempts):
        try:
            return fn()
        except OSError as e:                     # HF raises OSError/EnvironmentError here
            last = e
            print(f"load attempt {i + 1}/{attempts} failed: {str(e)[:160]}", flush=True)
            if i + 1 < attempts:
                time.sleep(wait_s)
    raise last


def load_audio_16k(path: str):
    """Mono float32 waveform at 16 kHz (what every processor here expects)."""
    import librosa
    wav, _ = librosa.load(path, sr=16000, mono=True)
    return wav


def build_inputs(model, processor, conversation: list[dict]):
    """Tokenised + audio-featurised inputs for one conversation, by model kind. The
    conversation format is shared (system text; user = audio item + text item)."""
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    kind = model_kind(model)
    if kind == "af3":
        audios = [load_audio_16k(c["audio"]) for m in conversation
                  if isinstance(m.get("content"), list) for c in m["content"] if c.get("type") == "audio"]
        return processor(text=text, audio=audios or None, return_tensors="pt")
    from qwen_omni_utils import process_mm_info
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    kw = {"use_audio_in_video": False} if getattr(model, "_reprompt_name", "qwen2.5-omni") == "qwen2.5-omni" else {}
    return processor(text=text, audio=audios, images=images, videos=videos,
                     return_tensors="pt", padding=True, **kw)


# Qwen2.5-Omni does not reliably stop after the answer: it emits a correct
# <Conclusion> block and then runs on, inventing "Human: What is the capital city of
# France?" turns for hundreds of tokens. This is the failure _OMNI_SYSTEM is meant to
# suppress, and the identity system prompt alone does NOT suppress it. Observed in the
# Gate A smoke run (job 10461202): ~1700 chars emitted where ~250 were content.
# Left unchecked it costs roughly 4x the generation time across the whole experiment.
# "\nHuman\n" is not redundant with "Human:". The colon-less form ESCAPED the guard: the
# evidence pass of job 10525748 ran on into a column of bare "Human" lines, 129 words on
# one row, because the model emitted "Human" on its own line with no colon and
# `"Human:"` cannot match it. Same shape as the plural that escaped the leak guard
# (\bnoise\b vs "noises") -- a literal that is one character away from the thing it was
# written to catch.
# Qwen3-Omni runs on with its OWN template's role names instead ("(d)\nuser",
# "(d)\nassistant\n(d)", smoke 10747650), so the lowercase turn markers are stopped too.
STOP_STRINGS = ("</Conclusion>", "\nHuman:", "Human:", "\nHuman\n", "\nuser", "\nassistant")

# A line that is EXACTLY "Human" (or "Assistant"), which is the run-on turn marker. Anchored
# to line boundaries so ordinary prose -- "a human voice", "the human speaker" -- survives.
_RUNON_TURN = re.compile(r"\n[ \t]*(?:Human|Assistant|user|assistant)[ \t]*(?:\n|$)")


def _truncate(text: str) -> str:
    """Cut the response at the end of the answer, discarding any run-on dialogue."""
    end = text.find("</Conclusion>")
    if end != -1:
        return text[: end + len("</Conclusion>")].strip()
    for marker in ("\nHuman:", "Human:"):
        cut = text.find(marker)
        if cut != -1:
            return text[:cut].strip()
    # The colon-less run-on. Reached only by turns with no <Conclusion> -- which is every
    # evidence pass, since that prompt asks for prose rather than the answer format.
    if m := _RUNON_TURN.search(text):
        return text[:m.start()].strip()
    return text.strip()


@torch.inference_mode()
def generate(model, processor, conversation: list[dict], *, max_new_tokens: int = 512,
             sample: bool = False, seed: int | None = None, return_full: bool = False):
    """Run one conversation, return the decoded assistant text, truncated at the answer.

    Greedy by default -- every measurement in this project assumes a deterministic actor.
    `sample=True` exists for self-consistency, which needs the SAME prompt to produce
    DIFFERENT answers; the seed is set per call so a k-sample vote is reproducible. The
    text generator learned this lesson already: greedy decoding made N candidates N
    identical strings and the optimiser paid N times to choose between copies.
    """
    inputs = build_inputs(model, processor, conversation)
    inputs = inputs.to(model.device).to(model.dtype)
    tok = getattr(processor, "tokenizer", None)
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=bool(sample))
    if getattr(model, "_reprompt_name", "qwen2.5-omni") == "qwen2.5-omni":
        kwargs["use_audio_in_video"] = False
    if sample:
        if seed is not None:
            torch.manual_seed(seed)
        kwargs.update(temperature=0.7, top_p=0.9)
    if tok is not None:
        kwargs.update(stop_strings=list(STOP_STRINGS), tokenizer=tok)
    out = model.generate(**inputs, **kwargs)
    # Strip the prompt: keep only newly generated tokens.
    gen = out[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(gen, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]
    # `return_full` hands back the UNTRUNCATED decode as well, so a run can be re-parsed
    # offline if the parser turns out to be the problem (src.expel.softpool_reparse).
    return (_truncate(raw), raw) if return_full else _truncate(raw)


def env_report() -> dict:
    return {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_native": native_bf16(),
        "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "dtype": str(pick_dtype()),
        "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "HF_HOME": os.environ.get("HF_HOME"),
    }
