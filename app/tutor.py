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

from . import prompts, scenarios, segments
from .audio_io import MicListener, Speaker
from .config import Config
from .corrections import filter_corrections
from .events import EventBus
from .llm import LlmClient
from .review import SessionLog
from .sentences import SentenceStreamer
from .stt import Transcriber
from .tts import Synthesizer

log = logging.getLogger(__name__)

# The playback queue drains when the last chunk is handed to the audio device,
# not when it finishes leaving the speaker. Keep the mic gated a moment longer
# so the tail of the tutor's own voice does not open a new utterance.
SPEECH_TAIL_MS = 250


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

    @property
    def mode(self) -> str:
        return self.cfg.tutor.mode

    # ---- setup --------------------------------------------------------

    async def preflight(self) -> None:
        ok, msg = await self.llm.health()
        if not ok:
            raise RuntimeError(f"LLM preflight failed: {msg}")
        if err := await self.llm.pin_model():
            log.info("could not pin model (not Ollama?): %s", err)
        log.info("warming models...")
        t0 = time.perf_counter()
        self.stt.warm()
        self.tts.warm()
        log.info("warm in %.1fs", time.perf_counter() - t0)

    def _messages(self) -> list[dict[str, str]]:
        system = prompts.system_prompt(
            self.mode, self.cfg.tutor.level, self.scenario.description
        )
        keep = self.cfg.tutor.history_turns * 2
        return [{"role": "system", "content": system}, *self.history[-keep:]]

    # ---- speaking -----------------------------------------------------

    async def _speak(self, text: str) -> None:
        """Synthesize one sentence, routing each language to its own voice.

        In mentor mode a single sentence is often mixed — English prose with a
        German phrase inside guillemets — so it may become several segments.
        """
        # Each fragment is classified by its own words (app/langid.py). The
        # mode only breaks ties on fragments too short to call, where practice
        # mode means an unmarked "Ja." is German and mentor mode means it is
        # more likely an English aside.
        tie_break = segments.DE if self.mode == prompts.PRACTICE else segments.EN

        for lang, chunk in segments.split_segments(text, tie_break):
            if self.mic.interrupted.is_set():
                return
            pcm = await asyncio.to_thread(self.tts.synthesize, chunk, lang)
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
            await self._end_speaking()

    async def _end_speaking(self) -> None:
        await asyncio.sleep(SPEECH_TAIL_MS / 1000)
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

        async def emit(sentence: str) -> None:
            """Strip any mode token, then speak what remains."""
            nonlocal first_audio_ms
            # Strip any stray mode token so it is never read aloud. We do not
            # act on it — see the note in prompts.py.
            clean, stray = segments.extract_mode(sentence)
            if stray:
                log.debug("stripped stray mode token (%s) from reply", stray)
            if not clean:
                return
            if first_audio_ms is None:
                first_audio_ms = (time.perf_counter() - t0) * 1000
            spoken.append(clean)
            await self._speak(clean)

        try:
            async for delta in self.llm.stream(self._messages()):
                if self.mic.interrupted.is_set():
                    log.info("cutting generation — user interrupted")
                    break
                for sentence in streamer.feed(delta):
                    await emit(sentence)

            if not self.mic.interrupted.is_set():
                for sentence in streamer.flush():
                    await emit(sentence)
                await asyncio.to_thread(self.speaker.wait_until_done, 60.0)
            else:
                self.speaker.cancel()

        except Exception as e:
            log.exception("generation failed")
            self.speaker.cancel()
            self.history.pop()  # don't poison history with a failed turn
            raise RuntimeError(str(e)) from e
        finally:
            await self._end_speaking()

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
        name = self.cfg.learner.name
        result = await self.llm.complete_json(
            prompts.corrector_messages(self.cfg.tutor.level, user_text, name)
        )
        if not result:
            return
        corrections = [c for c in result.get("corrections", []) if isinstance(c, dict)]
        vocab = [v for v in result.get("vocab", []) if isinstance(v, str)]

        # The prompt asks the model to leave names alone; it does not always
        # listen, and it cannot recognise a name it has never seen. Enforce it
        # here, where the check is deterministic.
        corrections, dropped = filter_corrections(corrections, name, user_text)
        for c in dropped:
            log.info("dropped correction (proper noun or no-op): %r -> %r",
                     c.get("original"), c.get("corrected"))

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
        self._publish_config()
        log.info("level -> %s", level)
        return True

    def set_scenario(self, key: str) -> bool:
        if key not in scenarios.SCENARIOS or key == self.scenario.key:
            return False
        self.scenario = scenarios.get(key)
        self.cfg.tutor.scenario = key
        self.history.clear()  # a new roleplay is a new conversation
        self._publish_config()
        log.info("scenario -> %s", key)
        return True

    def set_mode(self, mode: str) -> bool:
        mode = mode.lower()
        if mode not in (prompts.MENTOR, prompts.PRACTICE) or mode == self.mode:
            return False
        self.cfg.tutor.mode = mode
        # History carries over deliberately: switching to practice right after
        # being taught a phrase should let the tutor use that phrase.
        self._publish_config()
        log.info("mode -> %s", mode)
        return True

    def _publish_config(self) -> None:
        self.bus.publish(
            "config",
            mode=self.mode,
            level=self.cfg.tutor.level,
            scenario=self.scenario.key,
        )

    # ---- lifecycle ----------------------------------------------------

    async def aclose(self) -> None:
        if self._pending:
            log.info("finishing %d correction pass(es)...", len(self._pending))
            await asyncio.gather(*self._pending, return_exceptions=True)
        self.mic.stop()
        self.speaker.close()
        await self.llm.aclose()
        # The session has been on disk since the first turn; this final pass
        # just folds in any correction that landed during the gather above.
        if saved := self.session.save():
            log.info("session saved: %s", saved[0])
