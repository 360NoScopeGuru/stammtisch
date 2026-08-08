"""Text-to-speech via Piper, with one voice per language.

A German voice reading English produces mangled nonsense (it phonemises the
text with German rules), and vice versa. In mentor mode the tutor speaks mostly
English with German examples, so we keep both voices loaded and route each
segment to the right one.

Piper's Python API changed shape across releases (synthesize_stream_raw ->
synthesize chunk objects), so we probe for what's available at load time.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .config import Config

log = logging.getLogger(__name__)


def resample_linear(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or len(x) == 0:
        return x
    n = int(round(len(x) * dst / src))
    return np.interp(
        np.linspace(0, len(x) - 1, n, dtype=np.float64),
        np.arange(len(x), dtype=np.float64),
        x,
    ).astype(np.float32)


class _Voice:
    def __init__(self, name: str, voice, sample_rate: int, mode: str) -> None:
        self.name = name
        self.voice = voice
        self.sample_rate = sample_rate
        self.mode = mode


class Synthesizer:
    def __init__(self, cfg: Config) -> None:
        from piper import PiperVoice

        self.cfg = cfg
        voice_dir = cfg.models_root / "piper"
        self.voices: dict[str, _Voice] = {}

        for lang, name in cfg.tts.voices.items():
            onnx = voice_dir / f"{name}.onnx"
            if not onnx.exists():
                raise FileNotFoundError(
                    f"Piper voice not found: {onnx}\n"
                    f"Run: python scripts/fetch_models.py"
                )
            log.info("loading piper voice %s (%s)...", name, lang)
            v = PiperVoice.load(str(onnx))
            sr = getattr(getattr(v, "config", None), "sample_rate", None) or 22050
            self.voices[lang] = _Voice(name, v, sr, self._detect_api(v))

        if cfg.tts.primary not in self.voices:
            raise ValueError(
                f"tts.primary={cfg.tts.primary!r} has no matching entry in "
                f"tts.voices ({sorted(self.voices)})"
            )

        # Everything is resampled to the primary voice's rate so a single
        # output stream can play mixed-language replies without reopening.
        self.sample_rate = self.voices[cfg.tts.primary].sample_rate
        cfg.tts.sample_rate = self.sample_rate

        rates = {lang: v.sample_rate for lang, v in self.voices.items()}
        if len(set(rates.values())) > 1:
            log.info("voice rates %s -> resampling to %d Hz", rates, self.sample_rate)
        log.info("piper ready: %s", {k: v.name for k, v in self.voices.items()})

    @staticmethod
    def _detect_api(voice) -> str:
        if hasattr(voice, "synthesize_stream_raw"):
            return "stream_raw"
        if hasattr(voice, "synthesize"):
            return "synthesize"
        raise RuntimeError("unrecognised piper-tts API; pin piper-tts>=1.2.0")

    def _synth_kwargs(self) -> dict:
        return {
            "length_scale": self.cfg.tts.length_scale,
            "noise_scale": self.cfg.tts.noise_scale,
            "noise_w": self.cfg.tts.noise_w,
        }

    def warm(self) -> None:
        for lang in self.voices:
            self.synthesize("Hallo." if lang == "de" else "Hello.", lang)

    def synthesize(self, text: str, lang: str | None = None) -> np.ndarray:
        """Render one segment to mono float32 PCM at self.sample_rate."""
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)

        v = self.voices.get(lang or self.cfg.tts.primary)
        if v is None:  # unknown language: fall back rather than fail a turn
            v = self.voices[self.cfg.tts.primary]

        t0 = time.perf_counter()
        raw = b"".join(self._raw_chunks(v, text))
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        pcm = resample_linear(pcm, v.sample_rate, self.sample_rate)

        dur_ms = len(pcm) / self.sample_rate * 1000
        log.debug(
            "tts[%s] %.0fms for %.0fms audio", v.name,
            (time.perf_counter() - t0) * 1000, dur_ms,
        )
        return pcm

    def _raw_chunks(self, v: _Voice, text: str):
        if v.mode == "stream_raw":
            yield from v.voice.synthesize_stream_raw(text, **self._synth_kwargs())
            return

        try:
            result = v.voice.synthesize(text, **self._synth_kwargs())
        except TypeError:
            result = v.voice.synthesize(text)

        for chunk in result:
            data = getattr(chunk, "audio_int16_bytes", None)
            if data is None:
                arr = getattr(chunk, "audio_int16_array", None)
                data = arr.tobytes() if arr is not None else bytes(chunk)
            yield data
