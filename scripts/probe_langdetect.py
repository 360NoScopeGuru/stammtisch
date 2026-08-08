"""Does constraining Whisper's language choice to {de, en} actually help?

Mentor-mode auto-detect picked Urdu, Arabic, Chinese, Italian and Indonesian on
seven of twenty-one utterances in one real session, and transcribed them as
such. `Transcriber.transcribe` now restricts the choice to
`stt.detect_languages`. This checks two things:

  1. that clean speech is unaffected, and
  2. that when detection *does* go off the rails, the restriction catches it.

Clean Piper speech is detected correctly every time, which is precisely why the
bug was invisible to `test_pipeline.py`. So the audio here is deliberately
degraded — noise, pitch shift, clipping — until Whisper starts guessing, which
is what a beginner's accent through laptop mic does in practice.

No microphone needed. `python scripts/probe_langdetect.py`
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.config import load_config  # noqa: E402
from app.stt import Transcriber  # noqa: E402
from app.tts import Synthesizer  # noqa: E402

CLIPS = [
    ("de", "Guten Morgen, wie geht es dir?"),
    ("de", "Ich heiße Vamshee."),
    ("de", "Ja."),
    ("en", "How do I say good morning?"),
    ("en", "Can we practise now?"),
]

# Progressively nastier. `clean` is the control.
DEGRADATIONS = [
    ("clean", 0.0, 1.0),
    ("noisy", 0.05, 1.0),
    ("noisy+pitched", 0.08, 1.12),
    ("wrecked", 0.15, 0.88),
]


def to_16k(pcm: np.ndarray, src_rate: int) -> np.ndarray:
    n = int(len(pcm) * 16000 / src_rate)
    return np.interp(
        np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm
    ).astype(np.float32)


def degrade(audio: np.ndarray, noise: float, rate: float) -> np.ndarray:
    """Noise plus a resample, which shifts pitch and duration together —
    a crude stand-in for an unfamiliar accent."""
    if rate != 1.0:
        n = int(len(audio) / rate)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
        ).astype(np.float32)
    if noise:
        rng = np.random.default_rng(0)
        audio = audio + rng.normal(0, noise, len(audio)).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def main() -> int:
    cfg = load_config()
    tts = Synthesizer(cfg)
    stt = Transcriber(cfg)
    stt.warm()

    print(f"\nmodel: {cfg.stt.model}   candidates: {cfg.stt.detect_languages}\n")

    clean = {text: to_16k(tts.synthesize(text, want), tts.sample_rate)
             for want, text in CLIPS}

    overall_raw = overall_fixed = total = 0

    for label, noise, rate in DEGRADATIONS:
        print(f"  {label}:")
        raw_wrong = fixed_wrong = 0

        for want, text in CLIPS:
            audio = degrade(clean[text], noise, rate)

            # What unconstrained detection would have said.
            _, info = stt._run(audio, None)
            raw = info.language
            probs = dict(info.all_language_probs or [])

            # What the production path now says.
            t0 = time.perf_counter()
            heard = stt.transcribe(audio, None)
            ms = (time.perf_counter() - t0) * 1000
            fixed = stt.last_language

            raw_wrong += raw != want
            fixed_wrong += fixed != want
            mark = "  " if fixed == want else "<-"
            print(f"    {text[:30]:<32} raw={raw:<3}{probs.get(raw, 0):.2f}  "
                  f"-> {fixed:<3} {ms:>5.0f}ms {mark} {heard[:40]!r}")

        total += len(CLIPS)
        overall_raw += raw_wrong
        overall_fixed += fixed_wrong
        print(f"    wrong: unconstrained {raw_wrong}/{len(CLIPS)}, "
              f"constrained {fixed_wrong}/{len(CLIPS)}\n")

    print(f"  TOTAL  unconstrained wrong : {overall_raw}/{total}")
    print(f"         constrained   wrong : {overall_fixed}/{total}")
    ok = overall_fixed <= overall_raw
    print(f"\n  {'PASS' if ok else 'FAIL'} — restriction never does worse\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
