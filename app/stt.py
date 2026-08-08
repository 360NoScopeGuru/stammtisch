"""Speech-to-text via faster-whisper, pinned to German."""

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

    def transcribe(self, audio: np.ndarray, language: str | None = "") -> str:
        """`audio` is mono float32 in [-1, 1] at the configured input rate.

        `language` overrides the configured one. Pass None to auto-detect —
        necessary in mentor mode, where the learner speaks mostly English.
        Forcing German onto English speech does not fail loudly; it produces
        confident German-shaped nonsense ("how are you" -> "Of are you"), which
        then reaches the corrector as if the learner had really said it.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        lang = self.cfg.stt.language if language == "" else language

        t0 = time.perf_counter()
        segments, info = self.model.transcribe(
            audio,
            language=lang,
            beam_size=self.cfg.stt.beam_size,
            vad_filter=self.cfg.stt.vad_filter,
            condition_on_previous_text=False,  # avoids runaway repetition loops
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        self.last_language = getattr(info, "language", lang) or lang
        log.debug(
            "stt %.0fms [%s] -> %r",
            (time.perf_counter() - t0) * 1000, self.last_language, text,
        )
        return text
