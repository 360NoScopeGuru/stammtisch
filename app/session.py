"""The conversation driver, shared by the CLI and the web UI.

Both front-ends run this exact loop and differ only in how they render the
event stream. Typed messages and controls arrive on an asyncio queue; speech
arrives on the microphone's queue when voice input is on. Whichever lands
first is handled, so the learner can type and talk in the same session.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import re
import time

from . import documents, drills, objectives, scenarios, study
from .events import EventBus
from .intents import requested_mode, wants_drill
from .prompts import MENTOR, PRACTICE
from .segments import strip_markers
from .tutor import Tutor

log = logging.getLogger(__name__)

# Status values the UI renders as a pill. READY is the text-mode resting
# state: nothing is listening, the learner is expected to type.
IDLE, LISTENING, THINKING, SPEAKING, READY = (
    "idle", "listening", "thinking", "speaking", "ready")

# Without acoustic echo cancellation, laptop speakers feed the tutor's own voice
# back into the mic. Whisper transcribes it cleanly and it enters the history as
# a user turn, so the tutor ends up talking to itself. Any transcript that
# closely matches what we just said, and arrives soon after we said it, is echo.
ECHO_SIMILARITY = 0.75
ECHO_WINDOW_S = 2.5

# How many goes at one phrase before moving on. Drilling the same words past
# this stops being practice and starts being a wall.
DRILL_MAX_ATTEMPTS = 3

# Goes at one chapter goal before the check ends. A second try after feedback
# is fair; a third is the learner copying the feedback back.
GOAL_MAX_ATTEMPTS = 2

MAX_TYPED_CHARS = 4000


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
        # Repeat-after-me state. While a target is set, the next thing heard is
        # an attempt at it rather than a turn in the conversation.
        self._drill_target = ""
        self._drill_attempts = 0
        # Chapter-goal check state: (goal, task, attempts). While set, the
        # next message is an answer to the task.
        self._check: tuple[objectives.GoalStatus, str, int] | None = None

    def request_stop(self) -> None:
        self._stop.set()
        self.controls.put_nowait({"action": "_wake"})

    def submit(self, msg: dict) -> None:
        self.controls.put_nowait(msg)

    def interrupt(self) -> None:
        """Stop the current reply now — not queued behind it."""
        self.tutor.interrupt()

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
            voice_input=self.tutor.listening,
            speak=self.tutor.speaking_aloud,
        )

    def _rest(self) -> None:
        self.bus.publish("status", state=LISTENING if self.tutor.listening else READY)

    async def _tutor_line(self, text: str, **extra) -> None:
        """Show and speak a fixed line, and keep it in the transcript."""
        self.bus.publish("status", state=SPEAKING)
        self.bus.publish("tutor_turn", text=text, latency_ms=None, **extra)
        await self.tutor.say(text)
        plain = strip_markers(text)
        self._mark_spoken(plain)
        self.tutor.session.add_turn("assistant", plain)

    async def _speak_opener(self) -> None:
        """Greet in the current mode's language: English to teach, German to
        practise."""
        sc = self.tutor.scenario
        if self.tutor.mode == PRACTICE:
            opener = f"«{sc.opener}»"   # marked so it routes to the German voice
        else:
            opener = sc.intro
        await self._tutor_line(opener, opener=True)

    # ---- repeat-after-me ----------------------------------------------

    async def _start_drill(self, target: str) -> None:
        """Say the phrase, then wait for the learner to say it back."""
        if not self.tutor.listening:
            self.bus.publish("error", message="Repeat-after-me needs the "
                             "microphone. Turn voice input on first.")
            return
        self._drill_target = target
        self._drill_attempts = 0
        self.bus.publish("drill_start", target=target)
        self.bus.publish("status", state=SPEAKING)
        await self.tutor.say(f"Say this after me. «{target}»")
        # Deliberately not marked as spoken for the echo guard: the learner is
        # about to repeat this phrase on purpose, and the guard would throw the
        # attempt away as an echo of the prompt.
        self.bus.publish("status", state=LISTENING)

    def _end_drill(self) -> None:
        self._drill_target = ""
        self._drill_attempts = 0

    async def _handle_attempt(self, text: str) -> None:
        target = self._drill_target
        attempt = drills.compare(target, text)
        self._drill_attempts += 1

        self.bus.publish(
            "drill_result",
            target=target,
            heard=text,
            score=attempt.score,
            verdict=attempt.verdict,
            words=[{"target": w.target, "heard": w.heard, "status": w.status}
                   for w in attempt.results],
        )
        log.info("drill %.2f (%s): %r -> %r",
                 attempt.score, attempt.verdict, target, text)

        feedback = attempt.feedback()
        keep_going = (attempt.verdict != "good"
                      and self._drill_attempts < DRILL_MAX_ATTEMPTS)
        if keep_going:
            spoken = f"{feedback} «{target}»"
        else:
            if attempt.verdict != "good":
                # Three goes is enough. Dwelling on one phrase is how a lesson
                # turns into a wall, and it can be revisited later.
                feedback += " Let us come back to that one another time."
            spoken = feedback
            self._end_drill()

        self.bus.publish("status", state=SPEAKING)
        await self.tutor.say(spoken)
        self.tutor.session.add_turn("assistant", strip_markers(spoken))
        if not keep_going:
            self.bus.publish("drill_end", target=target)

    # ---- chapter goals ------------------------------------------------

    def publish_goals(self) -> None:
        goals = self.tutor.goals()
        self.bus.publish(
            "goals",
            chapter=self.tutor.chapter.number if self.tutor.chapter else None,
            done=objectives.chapter_done(goals),
            goals=[{"index": g.index, "text": g.text,
                    "demonstrated": g.demonstrated, "attempts": g.attempts,
                    "by": g.by, "note": g.last_note} for g in goals],
        )

    async def _start_check(self, index: int) -> None:
        goals = self.tutor.goals()
        if not 0 <= index < len(goals):
            return
        goal = goals[index]
        self.bus.publish("status", state=THINKING)
        task = await study.goal_task(
            self.tutor.llm, self.tutor.cfg.tutor.level, goal.text,
            self.tutor.scenario.description)
        if not task:
            self.bus.publish("error", message="Could not set a check just now "
                             "— try again.")
            self._rest()
            return
        self._check = (goal, task, 0)
        self.bus.publish("goal_check", index=goal.index, goal=goal.text, task=task)
        line = f"Let us check one goal: «{goal.text}». {task}"
        await self._tutor_line(line, check=True)
        # In the history too, so the tutor knows what the next answer is for.
        self.tutor.history.append({"role": "assistant", "content": line})

    async def _handle_check_answer(self, answer: str) -> None:
        goal, task, attempts = self._check
        self.bus.publish("status", state=THINKING)
        verdict = await study.judge_goal(
            self.tutor.llm, self.tutor.cfg.tutor.level, goal.text, task, answer)
        if verdict is None:
            self.bus.publish("error", message="The check did not work this "
                             "time — nothing was recorded. Try answering again.")
            self._rest()
            return
        passed, note = verdict
        attempts += 1
        course = self.tutor.course.title if self.tutor.course else ""
        objectives.record(self.tutor.cfg.study_path, course,
                          self.tutor.chapter.number, goal.text, passed,
                          note=note, by=objectives.CHECK, answer=answer)
        self.bus.publish("goal_result", index=goal.index, goal=goal.text,
                         passed=passed, note=note)

        if passed or attempts >= GOAL_MAX_ATTEMPTS:
            self._check = None
            ending = (" That goal is done." if passed
                      else " Let us leave it there and practise it some more.")
        else:
            self._check = (goal, task, attempts)
            ending = " Have another go."
        line = (note or ("Well done." if passed else "Not quite.")) + ending
        await self._tutor_line(line)
        self.tutor.history.append({"role": "user", "content": answer})
        self.tutor.history.append({"role": "assistant", "content": line})
        self.publish_goals()
        if passed and objectives.chapter_done(self.tutor.goals()):
            self.bus.publish("chapter_done",
                             chapter=self.tutor.chapter.number)

    def _mark_goal(self, index: int, done: bool) -> None:
        goals = self.tutor.goals()
        if not 0 <= index < len(goals) or not self.tutor.chapter:
            return
        course = self.tutor.course.title if self.tutor.course else ""
        objectives.record(self.tutor.cfg.study_path, course,
                          self.tutor.chapter.number, goals[index].text, done,
                          note="marked by hand", by=objectives.MANUAL)
        self.publish_goals()

    # ---- echo ----------------------------------------------------------

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

    # ---- controls -----------------------------------------------------

    async def _apply_control(self, msg: dict) -> None:
        action = msg.get("action")
        if action == "say":
            await self._typed(str(msg.get("value", "")),
                              msg.get("attachments") or [])
        elif action == "set_level":
            self.tutor.set_level(str(msg.get("value", "")))
        elif action == "set_scenario":
            if self.tutor.set_scenario(str(msg.get("value", ""))):
                self._check = None
                self.bus.publish("scenario_changed", scenario=self.tutor.scenario.key)
                await self._speak_opener()
        elif action in ("set_chapter", "next_chapter"):
            changed = (self.tutor.next_chapter() if action == "next_chapter"
                       else self.tutor.set_chapter(int(msg.get("value", 0) or 0)))
            if changed:
                self._check = None
                self.bus.publish("scenario_changed",
                                 scenario=self.tutor.scenario.key)
                self.publish_goals()
                await self._speak_opener()
        elif action == "drill":
            target = str(msg.get("value", "")) or self.tutor.last_german
            if target:
                await self._start_drill(target)
        elif action == "stop_drill":
            self._end_drill()
            self.bus.publish("drill_end", target="")
        elif action == "set_mode":
            await self._switch_mode(str(msg.get("value", "")))
        elif action == "set_input":
            await self.tutor.set_voice_input(str(msg.get("value", "")) == "voice")
            self._publish_config()
        elif action == "set_speak":
            await self.tutor.set_speak(bool(msg.get("value")))
            self._publish_config()
        elif action == "check_goal":
            await self._start_check(int(msg.get("value", -1)))
        elif action == "cancel_check":
            self._check = None
            self.bus.publish("goal_check_end")
        elif action == "mark_goal":
            self._mark_goal(int(msg.get("value", -1)), bool(msg.get("done", True)))
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

    # ---- turns --------------------------------------------------------

    async def _resolve(self, refs: list) -> tuple[list[documents.Page], list[dict], str]:
        """Attachment references -> pages for the model, a description for the
        UI, and a note for the transcript. Raises DocumentError."""
        pages: list[documents.Page] = []
        shown: list[dict] = []
        notes: list[str] = []
        for ref in refs[:4]:
            if not isinstance(ref, dict):
                continue
            doc = self.tutor.documents.get(str(ref.get("id", "")))
            if doc is None:
                raise documents.DocumentError(
                    "That file is no longer loaded — attach it again.")
            # Rendering a PDF page is CPU work measured in hundreds of ms.
            chosen = await asyncio.to_thread(doc.select, str(ref.get("pages", "")))
            pages += chosen
            shown.append({"id": str(ref.get("id")), "name": doc.name,
                          "pages": [p.number for p in chosen]})
            notes.append(documents.describe(doc, chosen))
        if len(pages) > documents.MAX_PAGES_PER_MESSAGE:
            raise documents.DocumentError(
                f"at most {documents.MAX_PAGES_PER_MESSAGE} pages per message")
        return pages, shown, " ".join(notes)

    async def _typed(self, text: str, refs: list) -> None:
        text = text.strip()[:MAX_TYPED_CHARS]
        try:
            pages, shown, note = await self._resolve(refs)
        except documents.DocumentError as e:
            self.bus.publish("error", message=str(e))
            return
        if not text and not pages:
            return
        if self._drill_target:
            # Typing in the middle of a spoken drill means the learner has
            # moved on; a typed "attempt" says nothing about pronunciation.
            self._end_drill()
            self.bus.publish("drill_end", target="")
        await self._handle_user(text, typed=True, pages=pages, shown=shown,
                                note=note)

    async def _handle_user(self, text: str, *, typed: bool,
                           pages: list[documents.Page] | None = None,
                           shown: list[dict] | None = None,
                           note: str = "") -> None:
        """One learner message, however it arrived."""
        self.tutor.session.add_turn("user", f"{note} {text}".strip())
        self.bus.publish("user_turn", text=text, typed=typed,
                         attachments=shown or [])

        if self._check is not None and text:
            await self._handle_check_answer(text)
            return

        # "Let me try that" drills the last German phrase rather than
        # sending it to the model, which would answer the request in words
        # instead of actually making the learner say anything.
        if (not typed and not pages and wants_drill(text)
                and self.tutor.last_german):
            await self._start_drill(self.tutor.last_german)
            return

        if text and self.tutor.should_correct(text, typed):
            self.tutor.queue_correction(text, typed=typed)

        try:
            self.bus.publish("status", state=THINKING)
            self.bus.publish("tutor_start")
            turn = await self.tutor.respond(text or "(see the attached pages)",
                                            pages=pages, note=note)
        except RuntimeError as e:
            log.error("turn failed: %s", e)
            self.bus.publish("error", message=str(e))
            return

        if turn.text:
            # Echo comparison works on what was actually spoken, so the
            # guillemets have to come off first.
            self._mark_spoken(strip_markers(turn.text))
            self.tutor.session.add_turn(
                "assistant", strip_markers(turn.text), turn.latency_ms
            )
        self.bus.publish("tutor_turn", text=turn.text, latency_ms=turn.latency_ms)

        # Switch modes based on what the learner asked for, once the turn
        # has finished. Driven by the message rather than the model, which is
        # not reliable enough — see prompts.py.
        wanted = requested_mode(text, self.tutor.mode)
        if wanted:
            await self._switch_mode(wanted)

    async def _heard(self, audio) -> None:
        """One utterance from the microphone."""
        self.bus.publish("status", state=THINKING)
        # In mentor mode the learner speaks mostly English, so let Whisper
        # detect. In practice mode pin it to German: the learner's accent is
        # a beginner's and auto-detect drifts to English on short replies.
        # A drill attempt is German by definition — the learner was just
        # asked to say a German phrase — so pin it there too, or a shaky
        # first attempt gets detected as English and scored as nonsense.
        drilling = bool(self._drill_target)
        checking = self._check is not None
        lang = (self.tutor.cfg.stt.language
                if drilling or checking or self.tutor.mode != MENTOR else None)
        text = await asyncio.to_thread(self.tutor.stt.transcribe, audio, lang)
        if not text or len(text) < 2:
            return

        # Echo rejection is skipped mid-drill: the learner is repeating the
        # phrase the tutor just said, so of course it matches.
        if not drilling and self._reject_as_echo(text):
            return

        if drilling:
            self.tutor.session.add_turn("user", text)
            self.bus.publish("user_turn", text=text, drill=True)
            await self._handle_attempt(text)
            return

        await self._handle_user(text, typed=False)

    # ---- main loop ----------------------------------------------------

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self.tutor.mic.attach_loop(loop)

        await self.tutor.preflight()
        self.tutor.mic.start()
        self.tutor.speaker.start()

        self._publish_config()
        self.publish_goals()
        await self._speak_opener()

        while not self._stop.is_set():
            self._rest()

            # Wait for a message, a control, or speech, whichever lands first.
            ctrl_task = asyncio.ensure_future(self.controls.get())
            waits = {ctrl_task}
            audio_task = None
            if (audio_q := self.tutor.audio_queue) is not None:
                audio_task = asyncio.ensure_future(audio_q.get())
                waits.add(audio_task)
            done, _ = await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)

            if ctrl_task in done:
                if audio_task is not None:
                    audio_task.cancel()
                await self._apply_control(ctrl_task.result())
                continue

            ctrl_task.cancel()
            if self._stop.is_set():
                break
            await self._heard(audio_task.result())
            await self._drain_controls()

            # Without barge-in, anything the mic caught during playback is echo.
            if self.tutor.listening and not self.tutor.cfg.tutor.barge_in:
                q = self.tutor.audio_queue
                while q is not None and not q.empty():
                    q.get_nowait()
                while not self.tutor.mic.utterances.empty():
                    self.tutor.mic.utterances.get_nowait()

        self.bus.publish("status", state=IDLE)
        self.bus.publish("session_end", summary=self.tutor.session.summary())
