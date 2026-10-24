r"""The conversation loop.

Turn structure:
    typed text or (utterance -> STT)
        -> [ LLM stream -> screen, and sentence split -> TTS -> speaker ]
        \-> (async, off critical path) correction pass

Text is the primary channel. The reply streams to the screen token by token,
and if replies are read aloud, each finished sentence goes to the speaker while
the model is still writing the next one.

The two directions are independent. Typing while hearing the tutor's German
pronounced is the normal setup; the microphone and Whisper are loaded only
when voice input is switched on, so a text session starts in seconds and needs
neither.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import re
import threading
import time
from datetime import date, datetime
from dataclasses import dataclass, field

from . import (classlog, curriculum, documents, drills, homework, langid,
               objectives, progress, prompts, scenarios, segments, srs)
from .config import Config
from .corrections import filter_corrections
from .events import EventBus
from .llm import LlmClient
from .review import SessionLog
from .sentences import SentenceStreamer

log = logging.getLogger(__name__)

# The playback queue drains when the last chunk is handed to the audio device,
# not when it finishes leaving the speaker. Keep the mic gated a moment longer
# so the tail of the tutor's own voice does not open a new utterance.
SPEECH_TAIL_MS = 250

# Pages stay in the conversation for this many further turns, so "and number
# 3?" still has the worksheet in view. After that they are dropped from what
# is sent: each image is ~256 tokens and seconds of encoding on every turn.
ATTACHMENT_TURNS = 6


@dataclass
class Turn:
    role: str
    text: str
    latency_ms: float | None = None
    corrections: list[dict] = field(default_factory=list)


class _NoMic:
    """Stands in for MicListener while voice input is off.

    The turn logic only ever touches these attributes, so it runs unchanged
    with or without a microphone instead of branching on one everywhere.
    """

    def __init__(self) -> None:
        self.tutor_speaking = threading.Event()
        self.interrupted = threading.Event()
        self.utterances: queue.Queue = queue.Queue()
        self.async_utterances = None

    def attach_loop(self, loop):
        return None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _Silent:
    """Stands in for Speaker while replies are not read aloud."""

    def start(self) -> None:
        pass

    def play(self, pcm) -> None:
        pass

    def wait_until_done(self, timeout: float | None = None) -> bool:
        return True

    def cancel(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def close(self) -> None:
        pass


class Tutor:
    def __init__(self, cfg: Config, bus: EventBus | None = None) -> None:
        self.cfg = cfg
        self.bus = bus or EventBus()
        # A textbook, if one has been ingested. When present it supplies the
        # session's topic in place of the stock scenarios, so the app teaches
        # the same chapters in the same order as the learner's actual class.
        self.course = curriculum.find(cfg.courses_path, cfg.tutor.course)
        # Everything the learner has done before, rebuilt from the saved
        # sessions. Without it the tutor re-teaches «Guten Morgen» every launch
        # and never notices a mistake becoming a habit.
        self.progress = progress.build(cfg.sessions_path, cfg.homework_path)
        if not self.progress.is_empty:
            log.info("resuming: %d past session(s), %d words met, "
                     "%d recurring mistake(s)",
                     self.progress.sessions, len(self.progress.vocab),
                     len(self.progress.recurring(limit=99)))
        cfg.tutor.chapter = self.resume_chapter()
        self.chapter = self._resolve_chapter(cfg.tutor.chapter)
        self._follow_book_level()
        self.scenario = self._scenario_for(self.chapter)
        self.history: list[dict] = []
        # The most recent German phrase spoken, so "let me try that" has
        # something to drill.
        self.last_german = ""
        self.session = SessionLog(cfg, level=cfg.tutor.level, scenario=self.scenario.key)
        # Files the learner has shared this session, by upload id.
        self.documents: dict[str, documents.Document] = {}
        # The worked answers to the exercise on the latest page, if it has
        # blanks. Never shown to the learner; see _start_key.
        self.answer_key = ""
        self._key_task: asyncio.Task | None = None

        self.llm = LlmClient(cfg)
        self.stt = None
        self.tts = None
        self.mic = _NoMic()
        self.speaker = _Silent()
        self._speech_q: asyncio.Queue[str] | None = None
        self._speech_task: asyncio.Task | None = None

        if cfg.voice_input:
            self._load_voice_input()
        if cfg.tutor.speak:
            self._load_speech()

        self._pending: set[asyncio.Task] = set()

    @property
    def mode(self) -> str:
        return self.cfg.tutor.mode

    # ---- channels -----------------------------------------------------

    @property
    def listening(self) -> bool:
        """Is a real microphone attached?"""
        return self.stt is not None and not isinstance(self.mic, _NoMic)

    @property
    def speaking_aloud(self) -> bool:
        return self.tts is not None and self.cfg.tutor.speak

    @property
    def typed(self) -> bool:
        """Does the learner type rather than speak? Shapes the prompt."""
        return not self.listening

    def _load_voice_input(self) -> None:
        """Whisper and the microphone. Seconds to load and ~2 GB of VRAM,
        which is why a text session never pays for them."""
        from .audio_io import MicListener
        from .stt import Transcriber

        if self.stt is None:
            self.stt = Transcriber(self.cfg)
        self.mic = MicListener(self.cfg)

    def _load_speech(self) -> bool:
        """Piper and the speaker. Missing voices degrade to text, loudly."""
        if self.tts is not None:
            self.cfg.tutor.speak = True
            return True
        try:
            from .audio_io import Speaker
            from .tts import Synthesizer

            self.tts = Synthesizer(self.cfg)
            self.speaker = Speaker(self.cfg)
        except Exception as e:
            log.warning("replies will not be read aloud: %s", e)
            self.cfg.tutor.speak = False
            self.tts, self.speaker = None, _Silent()
            return False
        self.cfg.tutor.speak = True
        return True

    async def set_voice_input(self, on: bool) -> bool:
        """Switch the microphone on or off mid-session."""
        if on == self.listening:
            return False
        if on:
            self.bus.publish("notice", message="Loading speech recognition…")
            try:
                await asyncio.to_thread(self._load_voice_input)
                await asyncio.to_thread(self.stt.warm)
                self.mic.attach_loop(asyncio.get_running_loop())
                self.mic.start()
            except Exception as e:
                log.exception("could not start voice input")
                self.mic = _NoMic()
                self.bus.publish("error", message=f"Microphone unavailable: {e}")
                return False
            self.cfg.tutor.input = "voice"
        else:
            self.mic.stop()
            self.mic = _NoMic()
            self.cfg.tutor.input = "text"
        self._publish_config()
        return True

    async def set_speak(self, on: bool) -> bool:
        if on == self.speaking_aloud:
            return False
        if on:
            started = self.tts is None
            ok = await asyncio.to_thread(self._load_speech)
            if not ok:
                self.bus.publish("error", message="No voices available — run "
                                 "python scripts/fetch_models.py")
                return False
            if started:
                self.speaker.start()
        else:
            self.speaker.cancel()
            self._clear_speech_queue()
            self.cfg.tutor.speak = False
        self._publish_config()
        return True

    @property
    def audio_queue(self):
        return self.mic.async_utterances if self.listening else None

    def interrupt(self) -> None:
        """Stop generating and stop talking. Safe from any thread."""
        self.mic.interrupted.set()
        self.speaker.cancel()
        self._clear_speech_queue()

    # ---- the course ---------------------------------------------------

    def resume_chapter(self) -> int:
        """Which chapter this session should open on.

        Shared with the homework commands, which build a cut-down Tutor and
        would otherwise silently fall back to the stock scenarios — setting
        homework on "Absolute basics" while the lessons were on chapter 3.
        """
        if not (self.cfg.tutor.resume and self.course) or self.cfg.tutor.chapter:
            return self.cfg.tutor.chapter
        # The class sets the pace: if the learner has logged a class on a
        # later chapter than the last session, that is where they are now.
        noted = classlog.latest_chapter(self.cfg.study_path)
        last = self.progress.last_chapter
        if noted and self.course.get(noted) and (not last or noted > last):
            return noted
        # Carry on where the last session stopped; on a first ever run that is
        # simply the start of the book.
        first = self.course.first()
        return last or (first.number if first else 0)

    def _resolve_chapter(self, number: int) -> curriculum.Chapter | None:
        """Chapter `number`, or None to fall back to the stock scenarios."""
        if not self.course or number <= 0:
            return None
        chapter = self.course.get(number)
        if chapter is None:
            log.warning("chapter %s is not in %s — falling back to scenario %r",
                        number, self.course.title, self.cfg.tutor.scenario)
        return chapter

    def _follow_book_level(self) -> None:
        """Teach at the level of the volume the chapter is in.

        The course runs from A1 to B2 across four books. A learner on
        chapter 20 is in the B1 book, and A1 explanations of the B1 grammar
        would be both too slow and wrong for what the class expects.
        """
        level = getattr(self.chapter, "level", "") if self.chapter else ""
        if level and level.upper() in prompts.LEVELS:
            self.cfg.tutor.level = level.upper()

    def _scenario_for(self, chapter: curriculum.Chapter | None):
        if chapter is None:
            return scenarios.get(self.cfg.tutor.scenario)
        return curriculum.as_scenario(chapter, self.cfg.tutor.level)

    def _course_reference(self) -> str:
        """The book's own end-of-chapter summary, for the system prompt."""
        return self.chapter.reference_extract() if self.chapter else ""

    def goals(self) -> list[objectives.GoalStatus]:
        if not (self.course and self.chapter):
            return []
        return objectives.status(self.cfg.study_path, self.course.title,
                                 self.chapter)

    def _open_goals(self) -> list[str]:
        return [g.text for g in self.goals() if not g.demonstrated]

    # ---- setup --------------------------------------------------------

    async def preflight(self) -> None:
        ok, msg = await self.llm.health()
        if not ok:
            # The most common first-run failure by a wide margin, and the least
            # self-explanatory. Name the likely cause and the command.
            raise RuntimeError(
                f"cannot reach the LLM at {self.cfg.llm.base_url} — {msg}\n"
                f"     Is the server running? For Ollama: ollama serve\n"
                f"     Run `python main.py --doctor` to check everything."
            )
        if err := await self.llm.pin_model():
            log.info("could not pin model (not Ollama?): %s", err)
        log.info("warming models...")
        t0 = time.perf_counter()
        if self.stt is not None:
            await asyncio.to_thread(self.stt.warm)
        if self.tts is not None:
            await asyncio.to_thread(self.tts.warm)
        log.info("warm in %.1fs", time.perf_counter() - t0)

    def _messages(self, attachments: bool = False) -> list[dict]:
        today = date.today()
        study = srs.brief(self.cfg.study_path, self.progress, today)
        history = "\n".join(x for x in (self.progress.brief(), study) if x)
        system = prompts.system_prompt(
            self.mode, self.cfg.tutor.level, self.scenario.description,
            reference=self._course_reference(),
            history=history,
            typed=self.typed,
            spoken=self.speaking_aloud,
            attachments=attachments,
            class_notes=classlog.brief(self.cfg.study_path, today),
            goals=self._open_goals(),
        )
        if attachments and self.answer_key:
            system += "\n" + prompts.KEY_BLOCK.format(key=self.answer_key)
        keep = self.cfg.tutor.history_turns * 2
        return [{"role": "system", "content": system},
                *self._trim_images(self.history[-keep:])]

    def _trim_images(self, turns: list[dict]) -> list[dict]:
        """Keep only recent pages in what is sent to the model.

        The text that went with them stays, along with a note of what was
        attached, so the conversation still makes sense without the pixels.
        """
        out = []
        for age, msg in enumerate(reversed(turns)):
            content = msg.get("content")
            if isinstance(content, list) and age >= ATTACHMENT_TURNS * 2:
                text = " ".join(p.get("text", "") for p in content
                                if p.get("type") == "text")
                msg = {**msg, "content": text + " (pages no longer shown)"}
            out.append(msg)
        return list(reversed(out))

    # ---- speaking -----------------------------------------------------

    async def _speak(self, text: str) -> None:
        """Synthesize one sentence, routing each language to its own voice.

        In mentor mode a single sentence is often mixed — English prose with a
        German phrase inside guillemets — so it may become several segments.
        """
        if self.tts is None:
            return
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

    def _remember_german(self, text: str) -> None:
        """Keep the last drillable German phrase for "let me try that"."""
        tie_break = segments.DE if self.mode == prompts.PRACTICE else segments.EN
        german = [chunk for lang, chunk in segments.split_segments(text, tie_break)
                  if lang == segments.DE]
        if german:
            candidate = " ".join(german).strip()
            if len(drills.words(candidate)) >= 2:
                self.last_german = candidate

    # In text mode speech runs beside the conversation rather than in it: the
    # learner reads the reply and can type the next message while it is still
    # being spoken. A new turn cuts off whatever is left.

    def _ensure_speech_worker(self) -> None:
        if self._speech_task is None or self._speech_task.done():
            self._speech_q = asyncio.Queue()
            self._speech_task = asyncio.create_task(self._speech_worker())

    async def _speech_worker(self) -> None:
        assert self._speech_q is not None
        while True:
            sentence = await self._speech_q.get()
            try:
                await self._speak(sentence)
            except Exception:
                log.exception("speech failed (non-fatal)")

    def _clear_speech_queue(self) -> None:
        q = self._speech_q
        if q is None:
            return
        while not q.empty():
            q.get_nowait()

    def _queue_speech(self, sentence: str) -> None:
        self._ensure_speech_worker()
        assert self._speech_q is not None
        self._speech_q.put_nowait(sentence)

    async def say(self, text: str) -> None:
        """Speak a fixed line (openers, errors). Not streamed."""
        self._remember_german(text)
        if not self.speaking_aloud:
            return
        if not self.listening:
            self.speaker.resume()
            self.mic.interrupted.clear()
            self._queue_speech(text)
            return
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

    async def respond(self, user_text: str,
                      pages: list[documents.Page] | None = None,
                      note: str = "") -> Turn:
        """Stream one tutor turn to the screen, and to the speaker if on.

        `pages` are images shown to the model with this message; `note` names
        them ("[attached worksheet.pdf, page 3 of 4]") so that the turn still
        makes sense once the images have aged out of the context.
        """
        text = f"{note}\n{user_text}".strip() if note else user_text
        if pages:
            # Read first, teach second. Asked to do both at once the model
            # skims the page and teaches from what it expects to be there;
            # asked only to transcribe, it reads. The transcript also stays
            # in the conversation after the images age out of it, and the
            # learner is shown it, so a misreading is visible rather than
            # silently taught from.
            transcript = await self.read_pages(user_text, pages)
            self._start_key(transcript)
            if transcript:
                text = (f"{note}\nWhat the page says, transcribed exactly:\n"
                        f"{transcript}\n\n{user_text}").strip()
            # Whole pages here, not halves: the reading is done, and these
            # are for layout and pictures. Halves would double the images
            # re-sent on every turn while the page stays in view.
            content: str | list = [{"type": "text", "text": text},
                                   *documents.image_parts(pages, tile=False)]
        else:
            content = text
        self.history.append({"role": "user", "content": content})

        recent_pages = any(isinstance(m.get("content"), list)
                           for m in self.history[-ATTACHMENT_TURNS * 2:])
        # Voice: sentences are spoken in line, and the mic stays gated until
        # the speaker is done, or the tutor hears itself. Text: speech runs
        # beside the reply and never holds the next turn up.
        inline_speech = self.listening and self.speaking_aloud
        if self.typed:
            max_tokens = self.cfg.llm.text_max_tokens
            if recent_pages:
                max_tokens = int(max_tokens * 1.5)
        else:
            max_tokens = self.cfg.llm.max_tokens * (2 if recent_pages else 1)

        streamer = SentenceStreamer()
        written: list[str] = []
        t0 = time.perf_counter()
        first_ms: float | None = None

        self.mic.interrupted.clear()
        self.speaker.cancel()          # a new turn cuts off the old one
        self._clear_speech_queue()
        self.speaker.resume()
        if inline_speech:
            self.mic.tutor_speaking.set()

        async def emit(sentence: str) -> None:
            """Strip any mode token, then speak what remains."""
            clean, stray = segments.extract_mode(sentence)
            if stray:
                log.debug("stripped stray mode token (%s) from reply", stray)
            clean = clean.lstrip("-•* ").strip()
            if not clean or not any(c.isalnum() for c in clean):
                return
            if inline_speech:
                await self._speak(clean)
            elif self.speaking_aloud:
                self._queue_speech(clean)

        try:
            async for delta in self.llm.stream(
                    self._messages(attachments=recent_pages),
                    max_tokens=max_tokens):
                if self.mic.interrupted.is_set():
                    log.info("cutting generation — interrupted")
                    break
                if first_ms is None:
                    first_ms = (time.perf_counter() - t0) * 1000
                written.append(delta)
                self.bus.publish("tutor_delta", text=delta)
                # A line break ends a spoken sentence even without a full
                # stop, or each line of a conjugation list runs into the next.
                for sentence in streamer.feed(delta.replace("\n", ". \n")):
                    await emit(sentence)

            if not self.mic.interrupted.is_set():
                for sentence in streamer.flush():
                    await emit(sentence)
                if inline_speech:
                    await asyncio.to_thread(self.speaker.wait_until_done, 60.0)
            else:
                self.speaker.cancel()

        except Exception as e:
            log.exception("generation failed")
            self.speaker.cancel()
            self.history.pop()  # don't poison history with a failed turn
            raise RuntimeError(str(e)) from e
        finally:
            if inline_speech:
                await self._end_speaking()

        reply = _tidy("".join(written))
        if reply:
            self.history.append({"role": "assistant", "content": reply})
            self._remember_german(reply)

        if first_ms is not None:
            log.info("time-to-first-token: %.0f ms", first_ms)

        return Turn(role="assistant", text=reply, latency_ms=first_ms)

    def _start_key(self, transcript: str) -> None:
        """Solve an exercise with blanks in the background.

        Solving in passing while teaching, the model told a learner the first
        gap of «Wir ___ Lisa und Marie» was «kennen»; asked only to solve,
        it got 16 of 19 gaps on that page right. The first reply does not
        need the key — it explains the exercise and asks for item one — so
        the solving runs while that streams, and is ready by the time the
        learner has typed an answer.
        """
        if self._key_task is not None:
            self._key_task.cancel()
        self.answer_key = ""
        if not re.search(r"_{2,}|\.{4,}", transcript or ""):
            self._key_task = None
            return
        self._key_task = asyncio.create_task(self._solve(transcript))

    async def _solve(self, transcript: str) -> None:
        raw = await self.llm.complete_json(
            prompts.solve_messages(self.cfg.tutor.level, transcript),
            temperature=0.0, max_tokens=1500)
        lines = []
        for item in (raw or {}).get("items") or []:
            if not isinstance(item, dict) or not str(item.get("answer", "")).strip():
                continue
            also = [str(a) for a in item.get("also") or [] if str(a).strip()]
            lines.append(f"({item.get('n', '?')}) {str(item['answer']).strip()}"
                         + (f" (also: {', '.join(also)})" if also else "")
                         + (f" — {str(item.get('why', '')).strip()}"
                            if item.get("why") else ""))
        self.answer_key = "\n".join(lines)[:3000]
        if self.answer_key:
            log.info("answer key ready: %d item(s)", len(lines))

    async def read_pages(self, question: str,
                         pages: list[documents.Page]) -> str:
        """An exact transcription of the part of the pages being asked about."""
        self.bus.publish("notice", message="Reading the page…")
        t0 = time.perf_counter()
        text = await self.llm.complete(prompts.read_messages(question, pages),
                                       max_tokens=1200)
        log.info("read %d page(s) in %.1fs", len(pages), time.perf_counter() - t0)
        text = (text or "").strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1].strip()
        if text:
            self.bus.publish("page_read", text=text)
        return text[:6000]

    # ---- corrections (never on the critical path) ----------------------

    def queue_correction(self, user_text: str, typed: bool = False) -> None:
        task = asyncio.create_task(self._correct(user_text, typed))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    def should_correct(self, text: str, typed: bool) -> bool:
        """Only grade German.

        Running the corrector on an English sentence produced feedback like
        'Hof Arjo' -> 'Haus Arjo': confident, authoritative, and about a word
        the learner never said.
        """
        if self.mode == prompts.PRACTICE:
            return True
        if typed:
            return langid.detect(text, default=langid.EN) == langid.DE
        return getattr(self.stt, "last_language", None) == "de"

    async def _correct(self, user_text: str, typed: bool = False) -> None:
        name = self.cfg.learner.name
        result = await self.llm.complete_json(
            prompts.corrector_messages(self.cfg.tutor.level, user_text, name,
                                       typed=typed)
        )
        if not result:
            return
        corrections = [c for c in result.get("corrections", []) if isinstance(c, dict)]
        vocab, glossary = parse_vocab(result.get("vocab", []))

        # The prompt asks the model to leave names alone; it does not always
        # listen, and it cannot recognise a name it has never seen. Enforce it
        # here, where the check is deterministic.
        corrections, dropped = filter_corrections(corrections, name, user_text)
        for c in dropped:
            log.info("dropped correction (proper noun or no-op): %r -> %r",
                     c.get("original"), c.get("corrected"))

        self.session.add_feedback(user_text, corrections, vocab, glossary)
        for c in corrections:
            log.info("correction: %r -> %r", c.get("original"), c.get("corrected"))

        # Arrives after the turn, so the UI patches it in retroactively.
        if corrections or vocab:
            self.bus.publish(
                "corrections",
                utterance=user_text,
                items=corrections,
                vocab=vocab,
                glossary=glossary,
            )

    # ---- homework ------------------------------------------------------

    async def set_homework(self, count: int = 5) -> homework.Assignment | None:
        """Ask the model for an assignment on the current topic."""
        weaknesses = [m.label for m in self.progress.recurring(limit=3)]
        topic = self.scenario.description
        if notes := classlog.brief(self.cfg.study_path, date.today()):
            topic += "\n\nWhat the class covered most recently:\n" + notes
        raw = await self.llm.complete_json(
            prompts.homework_messages(
                self.cfg.tutor.level, topic,
                count=count, weaknesses=weaknesses,
            )
        )
        if not raw:
            log.warning("homework generation returned nothing")
            return None

        assignment = homework.parse_generated(
            raw,
            chapter=self.chapter.number if self.chapter else None,
            chapter_title=self.scenario.title,
            level=self.cfg.tutor.level,
        )
        if not assignment.exercises:
            log.warning("homework generation produced no usable exercises")
            return None

        homework.save(assignment, self.cfg.homework_path)
        log.info("homework set: %d exercises (%d closed)",
                 len(assignment.exercises),
                 sum(1 for e in assignment.exercises if e.kind == homework.CLOSED))
        self.bus.publish("homework_set", id=assignment.id,
                         count=len(assignment.exercises),
                         title=assignment.chapter_title)
        return assignment

    async def mark_homework(self, assignment: homework.Assignment
                            ) -> homework.Assignment:
        """Mark what the rules can, then send only the rest to the model."""
        open_indices = assignment.mark_closed()

        if open_indices:
            items = [
                {"index": i,
                 "prompt": assignment.exercises[i].prompt,
                 "given": assignment.answer_for(i).given}
                for i in open_indices
            ]
            raw = await self.llm.complete_json(
                prompts.marking_messages(assignment.level, items)
            )
            if raw:
                assignment.apply_model_marks(raw.get("marks") or [],
                                             expected=open_indices)
                assignment.comment = str(raw.get("comment", ""))[:300]
            else:
                log.warning("marking returned nothing; open answers left unmarked")
                assignment.marked = datetime.now().isoformat(timespec="seconds")
        else:
            assignment.marked = datetime.now().isoformat(timespec="seconds")

        homework.save(assignment, self.cfg.homework_path)
        right, total = assignment.score
        log.info("homework marked: %d/%d", right, total)
        self.bus.publish("homework_marked", id=assignment.id,
                         right=right, total=total,
                         comment=assignment.comment)
        return assignment

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
        self.chapter = None          # a stock scenario leaves the book behind
        self.cfg.tutor.chapter = 0
        self.history.clear()  # a new roleplay is a new conversation
        self._publish_config()
        log.info("scenario -> %s", key)
        return True

    def set_chapter(self, number: int) -> bool:
        """Move to another chapter of the textbook."""
        if not self.course:
            return False
        chapter = self.course.get(int(number))
        if chapter is None or (self.chapter and chapter.number == self.chapter.number):
            return False
        self.chapter = chapter
        self.cfg.tutor.chapter = chapter.number
        self._follow_book_level()
        self.scenario = self._scenario_for(chapter)
        self.cfg.tutor.scenario = self.scenario.key
        self.history.clear()  # a new chapter is a new lesson
        self._publish_config()
        log.info("chapter -> %s (%s)", chapter.number, chapter.title)
        return True

    def next_chapter(self) -> bool:
        if not (self.course and self.chapter):
            return False
        nxt = self.course.next_after(self.chapter.number)
        return self.set_chapter(nxt.number) if nxt else False

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
            title=self.scenario.title,
            chapter=self.chapter.number if self.chapter else None,
            course=self.course.title if self.course else None,
            voice_input=self.listening,
            speak=self.speaking_aloud,
        )

    # ---- lifecycle ----------------------------------------------------

    async def aclose(self) -> None:
        if self._pending:
            log.info("finishing %d correction pass(es)...", len(self._pending))
            await asyncio.gather(*self._pending, return_exceptions=True)
        for task in (self._speech_task, self._key_task):
            if task is not None:
                task.cancel()
        self.mic.stop()
        self.speaker.close()
        await self.llm.aclose()
        # The session has been on disk since the first turn; this final pass
        # just folds in any correction that landed during the gather above.
        if saved := self.session.save():
            log.info("session saved: %s", saved[0])


def parse_vocab(items) -> tuple[list[str], dict[str, str]]:
    """The corrector's vocabulary, as words plus a German -> English glossary.

    Asked for {"de": ..., "en": ...} objects, a model will sometimes still
    return bare strings. Both are kept; only the meaning is optional.
    """
    words: list[str] = []
    glossary: dict[str, str] = {}
    for item in items or []:
        if isinstance(item, str):
            de, en = item.strip(), ""
        elif isinstance(item, dict):
            de = str(item.get("de", "")).strip()
            en = str(item.get("en", "")).strip()
        else:
            continue
        if not de or len(de) > 60:
            continue
        words.append(de)
        if en:
            glossary[de] = en[:80]
    return words, glossary


def _tidy(text: str) -> str:
    """Strip mode tokens and stray whitespace from a finished reply, keeping
    its line breaks — a conjugation list is unreadable flattened."""
    lines = [segments.extract_mode(line)[0].rstrip() for line in text.split("\n")]
    out = "\n".join(lines).strip()
    # Told not to, gemma3 still writes *before* for emphasis now and then;
    # the page shows plain text, so the asterisks would show as asterisks.
    out = re.sub(r"(?<![\w*])\*{1,2}([^*\n]+?)\*{1,2}(?![\w*])", r"\1", out)
    while "\n\n\n" in out:
        out = out.replace("\n\n\n", "\n\n")
    return out
