"""Where the datasets live. Nothing else in the repo hard-codes a data location.

Every root is read from an environment variable and falls back to a directory under
`data/` (git-ignored), so a fresh clone works by either exporting the variables or
symlinking the datasets into `data/`:

    REPROMPT_SAKURA_ROOT   SAKURA checkout: <root>/data/<Track>/metadata.json + wavs
    REPROMPT_MMAR_ROOT     MMAR download:   <root>/MMAR-meta.json + <root>/audio/
    REPROMPT_MMAU_AUDIO    MMAU audio dir (the metadata ships in data/benchmarks/)

The intervention renders (noise, masking, adversarial injection) always live under
`data/variants/audio_variants`; see approaches/README.md for how to obtain them.
"""
from __future__ import annotations

import os
from pathlib import Path


def _root(env: str, default: str) -> Path:
    return Path(os.environ.get(env, default))


SAKURA_ROOT = _root("REPROMPT_SAKURA_ROOT", "data/sakura")
MMAR_ROOT = _root("REPROMPT_MMAR_ROOT", "data/mmar")
MMAU_AUDIO_ROOT = _root("REPROMPT_MMAU_AUDIO", "data/mmau_audio")
