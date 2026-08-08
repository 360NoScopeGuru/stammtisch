"""German text-to-speech via Piper.

Piper's Python API changed shape across releases (synthesize_stream_raw ->
synthesize chunk objects), so we probe for what's available at load time rather
than pinning to one signature.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .config import Config

log = logging.getLogger(__name__)


class Synthesizer:
    def __init__(self, cfg: Config) -> None:
        from piper import PiperVoice

        self.cfg = cfg
        voice_dir = cfg.models_root / "piper"
        onnx = voice_dir / f"{cfg.tts.voice}.onnx"
        if not onnx.exists():
            raise FileNotFoundError(
                f"Piper voice not found: {onnx}\n"
                f"Run: python scripts/fetch_models.py"
            )

        log.info("loading piper voice %s...", cfg.tts.voice)
        self.voice = PiperVoice.load(str(onnx))

        # Trust the voice's own config over ours where they disagree.
        model_sr = getattr(getattr(self.voice, "config", None), "sample_rate", None)
        if model_sr and model_sr != cfg.tts.sample_rate:
            log.warning(
                "config tts.sample_rate=%d but voice is %d Hz; using %d",
                cfg.tts.sample_rate, model_sr, model_sr,
            )
            cfg.tts.sample_rate = model_sr

        self._mode = self._detect_api()
        log.info("piper ready (api: %s)", self._mode)

    def _detect_api(self) -> str:
        if hasattr(self.voice, "synthesize_stream_raw"):
            return "stream_raw"
        if hasattr(self.voice, "synthesize"):
            return "synthesize"
        raise RuntimeError("unrecognised piper-tts API; pin piper-tts>=1.2.0")

    def _synth_kwargs(self) -> dict:
        return {
            "length_scale": self.cfg.tts.length_scale,
            "noise_scale": self.cfg.tts.noise_scale,
            "noise_w": self.cfg.tts.noise_w,
        }

    def warm(self) -> None:
        self.synthesize("Hallo.")

    def synthesize(self, text: str) -> np.ndarray:
        """Render one sentence to mono float32 PCM in [-1, 1]."""
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)

        t0 = time.perf_counter()
        raw = b"".join(self._raw_chunks(text))
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        dur_ms = len(pcm) / self.cfg.tts.sample_rate * 1000
        log.debug(
            "tts %.0fms for %.0fms audio (rtf %.2f)",
            (time.perf_counter() - t0) * 1000, dur_ms,
            (time.perf_counter() - t0) * 1000 / max(dur_ms, 1),
        )
        return pcm

    def _raw_chunks(self, text: str):
        if self._mode == "stream_raw":
            yield from self.voice.synthesize_stream_raw(text, **self._synth_kwargs())
            return

        # Newer API yields AudioChunk objects.
        try:
            result = self.voice.synthesize(text, **self._synth_kwargs())
        except TypeError:
            result = self.voice.synthesize(text)

        for chunk in result:
            data = getattr(chunk, "audio_int16_bytes", None)
            if data is None:
                arr = getattr(chunk, "audio_int16_array", None)
                data = arr.tobytes() if arr is not None else bytes(chunk)
            yield data
