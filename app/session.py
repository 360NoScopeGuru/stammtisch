"""The conversation driver, shared by the CLI and the web UI.

Both front-ends run this exact loop and differ only in how they render the
event stream. Controls (level/scenario changes) arrive on an asyncio queue and
are serviced between turns.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import re
import time

from . import scenarios
from .events import EventBus
from .intents import requested_mode
from .prompts import MENTOR, PRACTICE
from .segments import strip_markers
from .tutor import Tutor

log = logging.getLogger(__name__)

# Status values the UI renders as a pill.
IDLE, LISTENING, THINKING, SPEAKING = "idle", "listening", "thinking", "speaking"

# Without acoustic echo cancellation, laptop speakers feed the tutor's own voice
# back into the mic. Whisper transcribes it cleanly and it enters the history as
# a user turn, so the tutor ends up talking to itself. Any transcript that
# closely matches what we just said, and arrives soon after we said it, is echo.
ECHO_SIMILARITY = 0.75
ECHO_WINDOW_S = 2.5


def _normalize(text: str) -> str:
    return re.sub(r"[^\wäöüß ]+", "", text.lower()).strip()


def is_echo(heard: str, spoken: str) -> float:
    """Similarity of a heard utterance to what the tutor just said (0..1)."""
    if not heard or not spoken:
        return 0.0
    a, b = _normalize(heard), _normalize(spoken)
    if not a or not b:
        return 0.0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    # A short echo fragment ("wie war denn dein Tag") scores poorly against the
    # full sentence, so also check whether it is contained in it.
    if len(a) >= 12 and a in b:
        ratio = max(ratio, 0.95)
    return ratio


class ConversationRunner:
    def __init__(self, tutor: Tutor, bus: EventBus) -> None:
        self.tutor = tutor
        self.bus = bus
        self.controls: asyncio.Queue[dict] = asyncio.Queue()
        self._stop = asyncio.Event()
        # What the tutor last said, and when it finished saying it.
        self._last_spoken = ""
        self._spoke_at = 0.0
        self.echo_rejections = 0

    def request_stop(self) -> None:
        self._stop.set()
        self.controls.put_nowait({"action": "_wake"})

    def submit(self, msg: dict) -> None:
        self.controls.put_nowait(msg)

    # ---- helpers ------------------------------------------------------

    def _publish_config(self) -> None:
        self.bus.publish(
            "config",
            mode=self.tutor.mode,
            level=self.tutor.cfg.tutor.level,
            scenario=self.tutor.scenario.key,
            scenarios=[
                {"key": s.key, "title": s.title, "min_level": s.min_level}
                for s in scenarios.SCENARIOS.values()
            ],
            barge_in=self.tutor.cfg.tutor.barge_in,
        )

    async def _speak_opener(self) -> None:
        """Greet in the current mode's language: English to teach, German to
        practise."""
        sc = self.tutor.scenario
        if self.tutor.mode == PRACTICE:
            opener = f"«{sc.opener}»"   # marked so it routes to the German voice
        else:
            opener = sc.intro

        self.bus.publish("status", state=SPEAKING)
        await self.tutor.say(opener)

        plain = strip_markers(opener)
        self._mark_spoken(plain)
        self.tutor.session.add_turn("assistant", plain)
        self.bus.publish("tutor_turn", text=opener, latency_ms=None, opener=True)

    def _mark_spoken(self, text: str) -> None:
        self._last_spoken = text
        self._spoke_at = time.monotonic()

    def _reject_as_echo(self, text: str) -> bool:
        if time.monotonic() - self._spoke_at > ECHO_WINDOW_S:
            return False
        score = is_echo(text, self._last_spoken)
        if score < ECHO_SIMILARITY:
            return False
        self.echo_rejections += 1
        log.info("discarded echo (%.2f similar to own speech): %r", score, text)
        self.bus.publish("echo_rejected", text=text, similarity=round(score, 3))
        return True

    async def _apply_control(self, msg: dict) -> None:
        action = msg.get("action")
        if action == "set_level":
            self.tutor.set_level(str(msg.get("value", "")))
        elif action == "set_scenario":
            if self.tutor.set_scenario(str(msg.get("value", ""))):
                self.bus.publish("scenario_changed", scenario=self.tutor.scenario.key)
                await self._speak_opener()
        elif action == "set_chapter":
            if self.tutor.set_chapter(int(msg.get("value", 0) or 0)):
                self.bus.publish("scenario_changed",
                                 scenario=self.tutor.scenario.key)
                await self._speak_opener()
        elif action == "next_chapter":
            if self.tutor.next_chapter():
                self.bus.publish("scenario_changed",
                                 scenario=self.tutor.scenario.key)
                await self._speak_opener()
        elif action == "set_mode":
            await self._switch_mode(str(msg.get("value", "")))
        elif action == "stop":
            self.request_stop()

    async def _switch_mode(self, mode: str) -> None:
        """Change mode and greet in the new one, so the switch is audible."""
        if not self.tutor.set_mode(mode):
            return
        self.bus.publish("mode_changed", mode=self.tutor.mode)
        await self._speak_opener()

    async def _drain_controls(self) -> None:
        while not self.controls.empty():
            await self._apply_control(self.controls.get_nowait())

    # ---- main loop ----------------------------------------------------

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        audio_q = self.tutor.mic.attach_loop(loop)

        await self.tutor.preflight()
        self.tutor.mic.start()
        self.tutor.speaker.start()

        self._publish_config()
        await self._speak_opener()

        while not self._stop.is_set():
            self.bus.publish("status", state=LISTENING)

            # Wait for speech or a control message, whichever lands first.
            audio_task = asyncio.ensure_future(audio_q.get())
            ctrl_task = asyncio.ensure_future(self.controls.get())
            done, _ = await asyncio.wait(
                {audio_task, ctrl_task}, return_when=asyncio.FIRST_COMPLETED
            )

            if ctrl_task in done:
                audio_task.cancel()
                await self._apply_control(ctrl_task.result())
                continue

            ctrl_task.cancel()
            audio = audio_task.result()

            if self._stop.is_set():
                break

            self.bus.publish("status", state=THINKING)
            # In mentor mode the learner speaks mostly English, so let Whisper
            # detect. In practice mode pin it to German: the learner's accent is
            # a beginner's and auto-detect drifts to English on short replies.
            lang = None if self.tutor.mode == MENTOR else self.tutor.cfg.stt.language
            text = await asyncio.to_thread(self.tutor.stt.transcribe, audio, lang)
            if not text or len(text) < 2:
                continue

            if self._reject_as_echo(text):
                continue

            self.tutor.session.add_turn("user", text)
            self.bus.publish("user_turn", text=text)

            # Only grade German. Running the corrector on an English sentence
            # produced feedback like 'Hof Arjo' -> 'Haus Arjo': confident,
            # authoritative, and about a word the learner never said.
            heard_german = (
                self.tutor.mode == PRACTICE
                or getattr(self.tutor.stt, "last_language", None) == "de"
            )
            if heard_german:
                self.tutor.queue_correction(text)

            try:
                self.bus.publish("status", state=SPEAKING)
                turn = await self.tutor.respond(text)
            except RuntimeError as e:
                log.error("turn failed: %s", e)
                self.bus.publish("error", message=str(e))
                continue

            if turn.text:
                # Echo comparison works on what was actually spoken, so the
                # guillemets have to come off first.
                self._mark_spoken(strip_markers(turn.text))
                self.tutor.session.add_turn(
                    "assistant", strip_markers(turn.text), turn.latency_ms
                )
                self.bus.publish(
                    "tutor_turn", text=turn.text, latency_ms=turn.latency_ms
                )

            # Switch modes based on what the learner asked for, once the turn
            # has finished speaking. Driven by the transcript rather than the
            # model, which is not reliable enough — see prompts.py.
            wanted = requested_mode(text, self.tutor.mode)
            if wanted:
                await self._switch_mode(wanted)

            await self._drain_controls()

            # Without barge-in, anything the mic caught during playback is echo.
            if not self.tutor.cfg.tutor.barge_in:
                while not audio_q.empty():
                    audio_q.get_nowait()
                while not self.tutor.mic.utterances.empty():
                    self.tutor.mic.utterances.get_nowait()

        self.bus.publish("status", state=IDLE)
        self.bus.publish("session_end", summary=self.tutor.session.summary())
