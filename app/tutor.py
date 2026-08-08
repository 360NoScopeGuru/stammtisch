r"""The conversation loop.

Turn structure:
    utterance -> STT -> [ LLM stream -> sentence split -> TTS -> speaker ]
                     \-> (async, off critical path) correction pass

The bracketed part is pipelined: sentence 1 is already playing while the model
is still generating sentence 3. That is what keeps time-to-first-audio ~1s.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from . import prompts, scenarios
from .audio_io import MicListener, Speaker
from .config import Config
from .events import EventBus
from .llm import LlmClient
from .review import SessionLog
from .sentences import SentenceStreamer
from .stt import Transcriber
from .tts import Synthesizer

log = logging.getLogger(__name__)


@dataclass
class Turn:
    role: str
    text: str
    latency_ms: float | None = None
    corrections: list[dict] = field(default_factory=list)


class Tutor:
    def __init__(self, cfg: Config, bus: EventBus | None = None) -> None:
        self.cfg = cfg
        self.bus = bus or EventBus()
        self.scenario = scenarios.get(cfg.tutor.scenario)
        self.history: list[dict[str, str]] = []
        self.session = SessionLog(cfg, level=cfg.tutor.level, scenario=self.scenario.key)

        self.stt = Transcriber(cfg)
        self.tts = Synthesizer(cfg)
        self.llm = LlmClient(cfg)
        self.mic = MicListener(cfg)
        self.speaker = Speaker(cfg)

        self._pending: set[asyncio.Task] = set()

    # ---- setup --------------------------------------------------------

    async def preflight(self) -> None:
        ok, msg = await self.llm.health()
        if not ok:
            raise RuntimeError(f"LLM preflight failed: {msg}")
        log.info("warming models...")
        t0 = time.perf_counter()
        self.stt.warm()
        self.tts.warm()
        log.info("warm in %.1fs", time.perf_counter() - t0)

    def _messages(self) -> list[dict[str, str]]:
        system = prompts.tutor_system_prompt(
            self.cfg.tutor.level, self.scenario.description
        )
        keep = self.cfg.tutor.history_turns * 2
        return [{"role": "system", "content": system}, *self.history[-keep:]]

    # ---- speaking -----------------------------------------------------

    async def _speak(self, text: str) -> None:
        """Synthesize and enqueue one sentence, off the event loop thread."""
        pcm = await asyncio.to_thread(self.tts.synthesize, text)
        if len(pcm):
            self.speaker.play(pcm)

    async def say(self, text: str) -> None:
        """Speak a fixed line (openers, errors). Not streamed."""
        self.mic.tutor_speaking.set()
        self.speaker.resume()
        self.mic.interrupted.clear()
        try:
            await self._speak(text)
            await asyncio.to_thread(self.speaker.wait_until_done, 30.0)
        finally:
            self.mic.tutor_speaking.clear()

    async def respond(self, user_text: str) -> Turn:
        """Stream one tutor turn, speaking sentence-by-sentence."""
        self.history.append({"role": "user", "content": user_text})

        streamer = SentenceStreamer()
        spoken: list[str] = []
        t0 = time.perf_counter()
        first_audio_ms: float | None = None

        self.mic.interrupted.clear()
        self.speaker.resume()
        self.mic.tutor_speaking.set()

        try:
            async for delta in self.llm.stream(self._messages()):
                if self.mic.interrupted.is_set():
                    log.info("cutting generation — user interrupted")
                    break
                for sentence in streamer.feed(delta):
                    if first_audio_ms is None:
                        first_audio_ms = (time.perf_counter() - t0) * 1000
                    spoken.append(sentence)
                    await self._speak(sentence)

            if not self.mic.interrupted.is_set():
                for sentence in streamer.flush():
                    spoken.append(sentence)
                    await self._speak(sentence)
                await asyncio.to_thread(self.speaker.wait_until_done, 60.0)
            else:
                self.speaker.cancel()

        except Exception as e:
            log.exception("generation failed")
            self.speaker.cancel()
            self.history.pop()  # don't poison history with a failed turn
            raise RuntimeError(str(e)) from e
        finally:
            self.mic.tutor_speaking.clear()

        reply = " ".join(spoken).strip()
        if reply:
            self.history.append({"role": "assistant", "content": reply})

        if first_audio_ms is not None:
            log.info("time-to-first-audio: %.0f ms", first_audio_ms)

        return Turn(role="assistant", text=reply, latency_ms=first_audio_ms)

    # ---- corrections (never on the critical path) ----------------------

    def queue_correction(self, user_text: str) -> None:
        task = asyncio.create_task(self._correct(user_text))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _correct(self, user_text: str) -> None:
        result = await self.llm.complete_json(
            prompts.corrector_messages(self.cfg.tutor.level, user_text)
        )
        if not result:
            return
        corrections = [c for c in result.get("corrections", []) if isinstance(c, dict)]
        vocab = [v for v in result.get("vocab", []) if isinstance(v, str)]
        self.session.add_feedback(user_text, corrections, vocab)
        for c in corrections:
            log.info("correction: %r -> %r", c.get("original"), c.get("corrected"))

        # Arrives after the spoken turn, so the UI patches it in retroactively.
        if corrections or vocab:
            self.bus.publish(
                "corrections",
                utterance=user_text,
                items=corrections,
                vocab=vocab,
            )

    # ---- runtime controls ---------------------------------------------

    def set_level(self, level: str) -> bool:
        level = level.upper()
        if level not in prompts.LEVELS or level == self.cfg.tutor.level:
            return False
        self.cfg.tutor.level = level
        # History stays: the tutor simply adapts its register from here on.
        self.bus.publish("config", level=level, scenario=self.scenario.key)
        log.info("level -> %s", level)
        return True

    def set_scenario(self, key: str) -> bool:
        if key not in scenarios.SCENARIOS or key == self.scenario.key:
            return False
        self.scenario = scenarios.get(key)
        self.cfg.tutor.scenario = key
        self.history.clear()  # a new roleplay is a new conversation
        self.bus.publish("config", level=self.cfg.tutor.level, scenario=key)
        log.info("scenario -> %s", key)
        return True

    # ---- lifecycle ----------------------------------------------------

    async def aclose(self) -> None:
        if self._pending:
            log.info("finishing %d correction pass(es)...", len(self._pending))
            await asyncio.gather(*self._pending, return_exceptions=True)
        self.mic.stop()
        self.speaker.close()
        await self.llm.aclose()
        self.session.save()
