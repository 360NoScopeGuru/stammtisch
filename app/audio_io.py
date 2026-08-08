"""Microphone capture with Silero VAD endpointing, plus interruptible playback.

IMPORTANT: there is no acoustic echo cancellation here. On laptop speakers the
tutor's own voice will re-enter the mic and (with barge-in on) it will interrupt
itself. Use headphones, or set tutor.barge_in: false.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import queue
import threading

import numpy as np
import sounddevice as sd

from .config import Config
from .vad import FRAME_SAMPLES as SILERO_FRAME
from .vad import SileroVad

log = logging.getLogger(__name__)


class MicListener:
    """Captures speech utterances and pushes each one onto `self.utterances`.

    Runs the sounddevice callback on its own thread; the queue is the handoff
    to the async side.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.sr = cfg.audio.input_sample_rate
        self.vad = SileroVad(cfg.models_root / "vad" / "silero_vad.onnx", self.sr)

        self.utterances: queue.Queue[np.ndarray] = queue.Queue()
        # Optional asyncio mirror of `utterances`, so callers can await audio
        # alongside other events instead of parking a thread on a blocking get.
        self._loop: asyncio.AbstractEventLoop | None = None
        self.async_utterances: asyncio.Queue[np.ndarray] | None = None
        # Set while the tutor is speaking, so we can detect barge-in.
        self.tutor_speaking = threading.Event()
        # Set when the user interrupts during playback.
        self.interrupted = threading.Event()

        self._stream: sd.InputStream | None = None
        self._reset_state()

        pre_roll_frames = max(1, int(cfg.vad.pre_roll_ms / cfg.audio.frame_ms))
        self._pre_roll: collections.deque[np.ndarray] = collections.deque(
            maxlen=pre_roll_frames
        )
        self._residual = np.zeros(0, dtype=np.float32)

    def _reset_state(self) -> None:
        self._collecting = False
        self._voiced: list[np.ndarray] = []
        self._speech_ms = 0
        self._silence_ms = 0

    # ---- lifecycle ----------------------------------------------------

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        """Mirror captured utterances onto an asyncio queue on `loop`."""
        self._loop = loop
        self.async_utterances = asyncio.Queue()
        return self.async_utterances

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            blocksize=SILERO_FRAME,
            device=self.cfg.audio.input_device,
            callback=self._on_audio,
        )
        self._stream.start()
        log.info("mic listening @ %d Hz", self.sr)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # ---- capture ------------------------------------------------------

    def _on_audio(self, indata, _frames, _time, status) -> None:
        if status:
            log.debug("input stream status: %s", status)

        # sounddevice honours blocksize, but guard against short/long blocks
        # by buffering into exact 512-sample frames for Silero.
        self._residual = np.concatenate([self._residual, indata[:, 0].copy()])
        while len(self._residual) >= SILERO_FRAME:
            frame, self._residual = (
                self._residual[:SILERO_FRAME],
                self._residual[SILERO_FRAME:],
            )
            self._process_frame(frame)

    def _process_frame(self, frame: np.ndarray) -> None:
        prob = self.vad(frame)
        is_speech = prob >= self.cfg.vad.threshold
        frame_ms = int(SILERO_FRAME / self.sr * 1000)

        if not self._collecting:
            self._pre_roll.append(frame)
            if not is_speech:
                self._speech_ms = 0
                return

            self._speech_ms += frame_ms
            if self._speech_ms < self.cfg.vad.min_speech_ms:
                return

            # Confirmed speech onset. Seed with pre-roll so we keep the attack.
            self._collecting = True
            self._voiced = list(self._pre_roll)
            self._pre_roll.clear()
            self._silence_ms = 0

            if self.tutor_speaking.is_set() and self.cfg.tutor.barge_in:
                log.info("barge-in detected")
                self.interrupted.set()
            return

        # --- collecting ---
        self._voiced.append(frame)
        self._silence_ms = 0 if is_speech else self._silence_ms + frame_ms

        total_ms = len(self._voiced) * frame_ms
        ended = self._silence_ms >= self.cfg.vad.end_silence_ms
        overlong = total_ms >= self.cfg.vad.max_utterance_ms

        if ended or overlong:
            audio = np.concatenate(self._voiced)
            self._reset_state()
            if overlong:
                log.warning("utterance hit max length, cutting at %d ms", total_ms)
            self.utterances.put(audio)

            # This runs on the sounddevice callback thread — hop to the loop.
            if self._loop is not None and self.async_utterances is not None:
                try:
                    self._loop.call_soon_threadsafe(
                        self.async_utterances.put_nowait, audio
                    )
                except RuntimeError:  # loop closed during shutdown
                    pass


class Speaker:
    """Queued PCM playback that can be cut off mid-sentence for barge-in."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.sr = cfg.tts.sample_rate
        self._chunks: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stop = threading.Event()
        self._drained = threading.Event()
        self._drained.set()
        self._thread: threading.Thread | None = None
        self._stream: sd.OutputStream | None = None

    def start(self) -> None:
        self._stream = sd.OutputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            device=self.cfg.audio.output_device,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            chunk = self._chunks.get()
            if chunk is None:
                return
            if self._stop.is_set():
                continue  # drain without playing
            try:
                assert self._stream is not None
                self._stream.write(chunk)
            except Exception:  # pragma: no cover - device hiccups
                log.exception("playback write failed")
            finally:
                if self._chunks.empty():
                    self._drained.set()

    def play(self, pcm: np.ndarray) -> None:
        self._drained.clear()
        self._chunks.put(pcm)

    def wait_until_done(self, timeout: float | None = None) -> bool:
        return self._drained.wait(timeout)

    def cancel(self) -> None:
        """Drop everything queued and silence the current chunk boundary."""
        self._stop.set()
        try:
            while True:
                self._chunks.get_nowait()
        except queue.Empty:
            pass
        self._drained.set()

    def resume(self) -> None:
        self._stop.clear()

    def close(self) -> None:
        self._chunks.put(None)
        if self._thread:
            self._thread.join(timeout=2)
        if self._stream:
            self._stream.stop()
            self._stream.close()


def list_devices() -> str:
    return str(sd.query_devices())
