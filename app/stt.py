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

    def warm(self) -> None:
        """First inference carries CUDA graph/kernel setup cost. Pay it at startup."""
        silence = np.zeros(self.cfg.audio.input_sample_rate, dtype=np.float32)
        self.transcribe(silence)

    def transcribe(self, audio: np.ndarray) -> str:
        """`audio` is mono float32 in [-1, 1] at the configured input rate."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        t0 = time.perf_counter()
        segments, _info = self.model.transcribe(
            audio,
            language=self.cfg.stt.language,
            beam_size=self.cfg.stt.beam_size,
            vad_filter=self.cfg.stt.vad_filter,
            condition_on_previous_text=False,  # avoids runaway repetition loops
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        log.debug("stt %.0fms -> %r", (time.perf_counter() - t0) * 1000, text)
        return text
