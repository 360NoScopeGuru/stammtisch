"""Speech-to-text via faster-whisper, constrained to the languages in play."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .config import Config

log = logging.getLogger(__name__)


class Transcriber:
    def __init__(self, cfg: Config) -> None:
        from .cuda_paths import register

        if cfg.stt.device == "cuda":
            register()  # must happen before faster_whisper loads CTranslate2

        from faster_whisper import WhisperModel

        self.cfg = cfg
        cache = cfg.models_root / "whisper"
        cache.mkdir(parents=True, exist_ok=True)

        log.info("loading whisper %s on %s...", cfg.stt.model, cfg.stt.device)
        t0 = time.perf_counter()
        self.model = WhisperModel(
            cfg.stt.model,
            device=cfg.stt.device,
            compute_type=cfg.stt.compute_type,
            download_root=str(cache),
        )
        log.info("whisper ready in %.1fs", time.perf_counter() - t0)
        self.last_language: str | None = cfg.stt.language

    def warm(self) -> None:
        """First inference carries CUDA graph/kernel setup cost. Pay it at startup."""
        silence = np.zeros(self.cfg.audio.input_sample_rate, dtype=np.float32)
        self.transcribe(silence)

    def _run(self, audio: np.ndarray, lang: str | None):
        segments, info = self.model.transcribe(
            audio,
            language=lang,
            beam_size=self.cfg.stt.beam_size,
            vad_filter=self.cfg.stt.vad_filter,
            condition_on_previous_text=False,  # avoids runaway repetition loops
        )
        # faster-whisper yields segments lazily and only fills `info.language`
        # once they are consumed, so this join is load-bearing, not cosmetic.
        return " ".join(seg.text.strip() for seg in segments).strip(), info

    def _restrict(self, info, fallback: str) -> str:
        """Best of the candidate languages, ignoring the other ninety-seven."""
        probs = dict(getattr(info, "all_language_probs", None) or [])
        candidates = [c for c in self.cfg.stt.detect_languages if c in probs]
        if not candidates:
            return fallback
        return max(candidates, key=probs.__getitem__)

    def transcribe(self, audio: np.ndarray, language: str | None = "") -> str:
        """`audio` is mono float32 in [-1, 1] at the configured input rate.

        `language` overrides the configured one. Pass None to auto-detect,
        which is what mentor mode needs — the learner speaks mostly English
        there, and forcing German onto English speech does not fail loudly. It
        produces confident German-shaped nonsense ("my name is Vamshee" ->
        "Bamsi Taran ist ein bisschen komisch") which then reaches the
        corrector as if the learner had really said it.

        Auto-detect is constrained to `stt.detect_languages`. Left
        unconstrained it is far worse than it looks: on one real session
        Whisper labelled seven of twenty-one utterances Urdu, Chinese, Arabic,
        Italian or Indonesian, at confidences as low as 0.30, and transcribed
        them accordingly. A beginner's accent on a two-second clip is simply
        not enough signal to pick from ninety-nine languages, but it is plenty
        to pick from two.

        The restriction is nearly free. The first pass already computes the
        full distribution, so a second pass is needed only when the global
        argmax lands outside the candidate set — exactly the case that was
        producing garbage anyway.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        forced = self.cfg.stt.language if language == "" else language

        t0 = time.perf_counter()
        text, info = self._run(audio, forced)
        detected = getattr(info, "language", forced) or forced

        redone = False
        if forced is None:
            wanted = self._restrict(info, fallback=detected)
            if wanted != detected:
                log.info("stt: detected %r, overriding to %r", detected, wanted)
                text, _ = self._run(audio, wanted)
                redone = True
            detected = wanted

        self.last_language = detected
        log.debug(
            "stt %.0fms [%s%s] -> %r",
            (time.perf_counter() - t0) * 1000,
            self.last_language, " re-run" if redone else "", text,
        )
        return text
