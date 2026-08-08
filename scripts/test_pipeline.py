"""Offline check of TTS -> VAD -> STT without a microphone or an LLM.

Synthesizes German with Piper, runs it through the VAD as if it were mic input,
then transcribes it with Whisper. If this passes, every local model works and
only the LLM server is left to verify.

    python scripts/test_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config  # noqa: E402
from app.vad import FRAME_SAMPLES, SileroVad  # noqa: E402

PHRASES = [
    "Guten Morgen, was darf es denn sein?",
    "Ich hätte gerne zwei Brötchen und einen Kaffee.",
    "Wie war dein Wochenende in Berlin?",
]


from app.tts import resample_linear  # noqa: E402


def main() -> int:
    cfg = load_config()
    ok = True

    print("\n[1/3] Piper ...")
    from app.tts import Synthesizer
    t0 = time.perf_counter()
    tts = Synthesizer(cfg)
    print(f"      loaded in {time.perf_counter() - t0:.1f}s")
    print(f"      voices: {{k: v.name for k, v in tts.voices.items()}}"
          .replace("{k: v.name for k, v in tts.voices.items()}",
                   str({k: v.name for k, v in tts.voices.items()})))

    # Mentor mode mixes languages inside one sentence; each piece must reach
    # the right voice and the pieces must concatenate at one sample rate.
    print("\n      mixed-language routing:")
    from app.segments import EN, split_segments
    mixed = 'Try saying «Ich heiße Anna», which means "my name is Anna".'
    total = 0
    for lang, chunk in split_segments(mixed, EN):
        pcm = tts.synthesize(chunk, lang)
        total += len(pcm)
        status = "OK  " if len(pcm) else "FAIL"
        ok &= bool(len(pcm))
        print(f"      {status}  [{lang}] {len(pcm) / tts.sample_rate * 1000:5.0f} ms  "
              f"{chunk[:40]}")
    print(f"            total {total / tts.sample_rate * 1000:.0f} ms at "
          f"{tts.sample_rate} Hz")

    clips = []
    for text in PHRASES:
        t0 = time.perf_counter()
        pcm = tts.synthesize(text, "de")
        gen_ms = (time.perf_counter() - t0) * 1000
        dur_ms = len(pcm) / cfg.tts.sample_rate * 1000
        if len(pcm) == 0:
            print(f"      FAIL  produced no audio for {text!r}")
            ok = False
            continue
        print(
            f"      {dur_ms:6.0f} ms audio in {gen_ms:5.0f} ms "
            f"(rtf {gen_ms / dur_ms:.3f})  {text[:38]}"
        )
        clips.append((text, pcm))

    print("\n[2/3] Silero VAD (speech should score high) ...")
    vad = SileroVad(cfg.models_root / "vad" / "silero_vad.onnx",
                    cfg.audio.input_sample_rate)
    vad_ready = []
    for text, pcm in clips:
        mono16k = resample_linear(pcm, tts.sample_rate,
                                  cfg.audio.input_sample_rate)
        vad.reset()
        probs = [
            vad(mono16k[i : i + FRAME_SAMPLES])
            for i in range(0, len(mono16k) - FRAME_SAMPLES + 1, FRAME_SAMPLES)
        ]
        speech_frac = sum(p >= cfg.vad.threshold for p in probs) / max(len(probs), 1)
        peak = max(probs) if probs else 0.0
        verdict = "OK  " if peak > 0.8 and speech_frac > 0.3 else "FAIL"
        ok &= verdict == "OK  "
        print(
            f"      {verdict}  peak={peak:.3f}  speech_frames={speech_frac:6.1%}  "
            f"{text[:34]}"
        )
        vad_ready.append((text, mono16k))

    print("\n[3/3] Whisper (round-trip: does it hear what Piper said?) ...")
    from app.stt import Transcriber
    stt = Transcriber(cfg)
    for text, mono16k in vad_ready:
        t0 = time.perf_counter()
        heard = stt.transcribe(mono16k)
        ms = (time.perf_counter() - t0) * 1000
        dur = len(mono16k) / cfg.audio.input_sample_rate
        print(f"      said  : {text}")
        print(f"      heard : {heard}")
        print(f"      {ms:.0f} ms for {dur:.1f}s audio ({dur * 1000 / ms:.0f}x rt)\n")

    print("PASS — local models all work.\n" if ok else "FAIL — see above.\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
