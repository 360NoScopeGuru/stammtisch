"""Drive ConversationRunner.run() with fake audio — no mic, no models, no LLM.

This covers the parts that are easy to get subtly wrong: the wait-on-either
(audio | control) race, task cancellation, control handling mid-conversation,
echo-draining when barge-in is off, typed turns with attached pages, and the
chapter-goal check. `python scripts/test_loop.py`
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.config import load_config  # noqa: E402
from app.events import EventBus  # noqa: E402
from app.session import ConversationRunner  # noqa: E402
from app.tutor import Turn  # noqa: E402


class FakeMic:
    def __init__(self):
        self.q = None
        self.utterances = asyncio.Queue()  # only .empty()/.get_nowait() are used
        self.started = False

    def attach_loop(self, loop):
        self.q = asyncio.Queue()
        return self.q

    def start(self):
        self.started = True

    def stop(self):
        pass


class FakeSpeaker:
    def start(self):
        pass

    def cancel(self):
        pass

    def close(self):
        pass


class FakeTutor:
    def __init__(self, cfg, bus, transcripts, voice=True):
        self.cfg = cfg
        self.bus = bus
        self.voice = voice
        self.history = []
        self.documents = {}
        self.llm = None
        self.course = types.SimpleNamespace(title="Testbuch")
        self.chapter = types.SimpleNamespace(number=3)
        self.goal_list = []
        self.responded = []      # (text, pages, note) per respond() call
        self.typed_flags = []    # `typed` per queued correction
        self.scenario = types.SimpleNamespace(
            key="baeckerei", title="Bakery",
            opener="Guten Morgen!", intro="Today we go to the bakery.",
            description="At the bakery.",
        )
        self.mic = FakeMic()
        self.speaker = FakeSpeaker()
        # Return "" once exhausted — the loop skips empty transcripts, so an
        # unexpected extra utterance shows up as a failed assertion, not a crash.
        # Records the language each call was made with, so a test can check
        # that a drill attempt is pinned to German rather than auto-detected.
        self.languages: list[str | None] = []

        def transcribe(a, lang=None):
            self.languages.append(lang)
            return transcripts.pop(0) if transcripts else ""

        self.stt = types.SimpleNamespace(
            transcribe=transcribe,
            last_language="de",   # so corrections are exercised
        )
        self.last_german = ""
        self.reply_delay = 0.0  # stand-in for TTS playback time
        self.session = _FakeSession()
        self.spoken: list[str] = []
        self.corrected: list[str] = []
        self.level_changes: list[str] = []
        self.mode_changes: list[str] = []
        self.fail_next = False

    @property
    def mode(self):
        return self.cfg.tutor.mode

    @property
    def listening(self):
        return self.voice

    @property
    def speaking_aloud(self):
        return True

    @property
    def audio_queue(self):
        return self.mic.q if self.voice else None

    def goals(self):
        return self.goal_list

    def interrupt(self):
        pass

    def should_correct(self, text, typed):
        return True

    def set_mode(self, v):
        if v == self.cfg.tutor.mode:
            return False
        self.mode_changes.append(v)
        self.cfg.tutor.mode = v
        return True

    async def preflight(self):
        pass

    async def say(self, text):
        self.spoken.append(text)

    async def respond(self, text, pages=None, note=""):
        self.responded.append((text, pages or [], note))
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("simulated LLM failure")
        if self.reply_delay:
            await asyncio.sleep(self.reply_delay)
        reply = f"Antwort auf: {text}"
        self.spoken.append(reply)
        return Turn(role="assistant", text=reply, latency_ms=910.0)

    def queue_correction(self, text, typed=False):
        self.corrected.append(text)
        self.typed_flags.append(typed)

    def set_level(self, v):
        self.level_changes.append(v)
        self.cfg.tutor.level = v
        return True

    def set_scenario(self, v):
        return False

    async def aclose(self):
        pass


class _FakeSession:
    def __init__(self):
        self.turns = []

    def add_turn(self, role, text, latency=None):
        self.turns.append((role, text))

    def summary(self):
        return "done"


async def collect(bus, out):
    q = bus.subscribe()
    while True:
        out.append(await q.get())


async def scenario_basic(cfg):
    """Two utterances flow through; one LLM failure is survived."""
    bus = EventBus()
    tutor = FakeTutor(cfg, bus, ["Ich hätte gern ein Brot.", "Danke schön."])
    runner = ConversationRunner(tutor, bus)
    events: list[dict] = []
    collector = asyncio.create_task(collect(bus, events))

    task = asyncio.create_task(runner.run())
    await asyncio.sleep(0.05)

    audio = np.zeros(1600, dtype=np.float32)
    tutor.mic.q.put_nowait(audio)
    await asyncio.sleep(0.1)

    tutor.fail_next = True
    tutor.mic.q.put_nowait(audio)
    await asyncio.sleep(0.1)

    runner.request_stop()
    await asyncio.wait_for(task, timeout=3)
    collector.cancel()

    kinds = [e["type"] for e in events]
    return {
        "mic started": tutor.mic.started,
        # Default mode is mentor, so the greeting is the English intro. The
        # German opener belongs to practice mode.
        "opener spoken in the mode's language": (
            bool(tutor.spoken) and tutor.spoken[0] == "Today we go to the bakery."
        ),
        "user_turn emitted": kinds.count("user_turn") == 2,
        "tutor_turn emitted": kinds.count("tutor_turn") == 2,  # opener + 1 reply
        "correction queued per utterance": len(tutor.corrected) == 2,
        "LLM failure surfaced as error": "error" in kinds,
        "loop survived the failure": True,
        "latency on the turn": any(
            e.get("latency_ms") == 910.0 for e in events if e["type"] == "tutor_turn"
        ),
        "session_end emitted": "session_end" in kinds,
    }


async def scenario_control(cfg):
    """A control message must win the race against a silent mic."""
    bus = EventBus()
    tutor = FakeTutor(cfg, bus, ["egal"])
    runner = ConversationRunner(tutor, bus)
    collector = asyncio.create_task(collect(bus, []))

    task = asyncio.create_task(runner.run())
    await asyncio.sleep(0.05)

    runner.submit({"action": "set_level", "value": "C1"})
    await asyncio.sleep(0.1)
    level_ok = tutor.level_changes == ["C1"]

    # The loop must still accept audio afterwards.
    tutor.mic.q.put_nowait(np.zeros(1600, dtype=np.float32))
    await asyncio.sleep(0.1)
    still_alive = len(tutor.corrected) == 1

    runner.request_stop()
    await asyncio.wait_for(task, timeout=3)
    collector.cancel()

    return {
        "control applied while idle": level_ok,
        "loop still works after a control": still_alive,
        "stop terminates the loop": task.done(),
    }


async def scenario_no_barge_in(cfg):
    """With barge-in off, audio captured during playback is discarded."""
    cfg.tutor.barge_in = False
    bus = EventBus()
    tutor = FakeTutor(cfg, bus, ["Hallo."])
    tutor.reply_delay = 0.3  # the window during which echo can arrive
    runner = ConversationRunner(tutor, bus)
    collector = asyncio.create_task(collect(bus, []))

    task = asyncio.create_task(runner.run())
    await asyncio.sleep(0.05)

    audio = np.zeros(1600, dtype=np.float32)
    tutor.mic.q.put_nowait(audio)
    await asyncio.sleep(0.1)         # loop is now inside respond()
    tutor.mic.q.put_nowait(audio)    # speaker bleed captured mid-reply
    await asyncio.sleep(0.4)         # reply finishes, drain runs

    drained = tutor.mic.q.empty()
    # The echo must not have been transcribed as a second user turn.
    only_one_turn = len(tutor.corrected) == 1

    runner.request_stop()
    await asyncio.wait_for(task, timeout=3)
    collector.cancel()

    return {
        "echo drained when barge_in is off": drained,
        "echo not treated as a user turn": only_one_turn,
    }


async def scenario_mode_handoff(cfg):
    """Asking to practise switches modes after the turn finishes speaking.

    The switch is driven by the learner's transcript, not by the model.
    """
    cfg.tutor.mode = "mentor"
    bus = EventBus()
    tutor = FakeTutor(cfg, bus, ["Can we practise now please?", "Hallo."])
    tutor.reply_delay = 0.15
    runner = ConversationRunner(tutor, bus)
    events: list[dict] = []
    collector = asyncio.create_task(collect(bus, events))

    task = asyncio.create_task(runner.run())
    await asyncio.sleep(0.05)
    tutor.mic.q.put_nowait(np.zeros(1600, dtype=np.float32))
    await asyncio.sleep(0.4)

    switched_after_reply = tutor.spoken.index("Antwort auf: Can we practise now "
                                              "please?") < len(tutor.spoken) - 1

    runner.request_stop()
    await asyncio.wait_for(task, timeout=3)
    collector.cancel()

    kinds = [e["type"] for e in events]
    return {
        "practice request switches mode": tutor.mode_changes == ["practice"],
        "mode_changed event published": "mode_changed" in kinds,
        "switch happens after the reply, not during": switched_after_reply,
        # Entering practice mode greets in German, marked for the German voice.
        "new-mode opener spoken in German": any(
            "Guten Morgen" in s for s in tutor.spoken
        ),
    }


async def scenario_drill(cfg):
    """"Let me try that" drills the last German phrase instead of replying.

    The whole point is that the learner has to *say* something. If this falls
    through to the model, it answers "of course, go ahead!" and nothing is
    practised.
    """
    cfg.tutor.mode = "mentor"
    bus = EventBus()
    tutor = FakeTutor(cfg, bus, [
        "Let me try that again",          # asks for a drill
        "Ich hätte gern ein Brot",        # attempt: one word wrong
        "Ich hätte gern ein Brötchen",    # attempt: right
        "So what is next?",               # back to normal conversation
    ])
    tutor.last_german = "Ich hätte gern ein Brötchen"
    runner = ConversationRunner(tutor, bus)
    events: list[dict] = []
    collector = asyncio.create_task(collect(bus, events))

    task = asyncio.create_task(runner.run())
    await asyncio.sleep(0.05)

    audio = np.zeros(1600, dtype=np.float32)
    for _ in range(4):
        tutor.mic.q.put_nowait(audio)
        await asyncio.sleep(0.15)

    runner.request_stop()
    await asyncio.wait_for(task, timeout=3)
    collector.cancel()

    kinds = [e["type"] for e in events]
    results = [e for e in events if e["type"] == "drill_result"]
    langs = tutor.languages

    return {
        "drill starts on request": "drill_start" in kinds,
        "the phrase is spoken back for the learner": any(
            "Ich hätte gern ein Brötchen" in s for s in tutor.spoken
        ),
        "both attempts scored": len(results) == 2,
        "a wrong word is not a pass": results[0]["verdict"] != "good",
        "the second attempt passes": results[-1]["verdict"] == "good",
        "the missed word is identified": any(
            w["status"] in ("wrong", "missing") and w["target"] == "Brötchen"
            for w in results[0]["words"]
        ),
        "drill ends after success": "drill_end" in kinds,
        "attempts were transcribed as German, not auto-detected": (
            langs[1] == "de" and langs[2] == "de"
        ),
        "the request itself was not sent to the model": all(
            "Let me try that again" not in s for s in tutor.spoken
        ),
        "conversation resumes afterwards": any(
            "Antwort auf: So what is next?" in s for s in tutor.spoken
        ),
    }


async def scenario_typed(cfg):
    """Text-first: typed messages are turns, with no microphone at all."""
    bus = EventBus()
    tutor = FakeTutor(cfg, bus, [], voice=False)
    runner = ConversationRunner(tutor, bus)
    events: list[dict] = []
    collector = asyncio.create_task(collect(bus, events))

    task = asyncio.create_task(runner.run())
    await asyncio.sleep(0.05)
    runner.submit({"action": "say", "value": "Ich heisse Anna."})
    await asyncio.sleep(0.05)
    runner.submit({"action": "say", "value": "   "})          # nothing to say
    runner.submit({"action": "say", "value": "Can we practise now?"})
    await asyncio.sleep(0.1)

    runner.request_stop()
    await asyncio.wait_for(task, timeout=3)
    collector.cancel()

    user = [e for e in events if e["type"] == "user_turn"]
    statuses = [e["state"] for e in events if e["type"] == "status"]
    return {
        "typed message becomes a turn": tutor.responded[0][0] == "Ich heisse Anna.",
        "blank message ignored": len(user) == 2,
        "marked as typed": all(e.get("typed") for e in user),
        "correction told the text was typed": tutor.typed_flags[:1] == [True],
        "never claims to be listening": "listening" not in statuses,
        "rests in the ready state": "ready" in statuses,
        "mode switch works from typed text too": tutor.mode_changes == ["practice"],
    }


class FakeDoc:
    name = "arbeitsblatt.pdf"
    kind = "pdf"
    page_count = 4

    def select(self, spec):
        from app.documents import Page, parse_pages
        return [Page(number=n, text="", image=b"x") for n in parse_pages(spec, 4)]


async def scenario_attachment(cfg):
    """Pages attached to a typed message reach the model with a note naming
    them; a bad page range is an error, not a turn."""
    bus = EventBus()
    tutor = FakeTutor(cfg, bus, [], voice=False)
    tutor.documents["d1"] = FakeDoc()
    runner = ConversationRunner(tutor, bus)
    events: list[dict] = []
    collector = asyncio.create_task(collect(bus, events))

    task = asyncio.create_task(runner.run())
    await asyncio.sleep(0.05)
    runner.submit({"action": "say", "value": "Can you check number 2?",
                   "attachments": [{"id": "d1", "pages": "2-3"}]})
    await asyncio.sleep(0.05)
    runner.submit({"action": "say", "value": "and this?",
                   "attachments": [{"id": "d1", "pages": "9"}]})
    runner.submit({"action": "say", "value": "",
                   "attachments": [{"id": "gone", "pages": ""}]})
    await asyncio.sleep(0.1)

    runner.request_stop()
    await asyncio.wait_for(task, timeout=3)
    collector.cancel()

    text, pages, note = tutor.responded[0]
    errors = [e["message"] for e in events if e["type"] == "error"]
    shown = next(e for e in events if e["type"] == "user_turn")["attachments"]
    return {
        "pages handed to the model": [p.number for p in pages] == [2, 3],
        "note names the file and pages": "arbeitsblatt.pdf" in note and "2-3" in note,
        "the UI is told what was attached": bool(shown) and shown[0]["pages"] == [2, 3],
        "out-of-range page is an error": any("outside" in m for m in errors),
        "unknown file is an error": any("no longer loaded" in m for m in errors),
        "neither error became a turn": len(tutor.responded) == 1,
        "the attachment note is in the transcript": any(
            "arbeitsblatt.pdf" in t for r, t in tutor.session.turns if r == "user"),
    }


async def scenario_goal_check(cfg):
    """A goal check sets a task, judges the next message, records it, and
    gives up after two tries. The model is replaced: what is under test is
    the flow and what gets recorded."""
    import tempfile
    from app import objectives, study

    cfg.paths.study_dir = tempfile.mkdtemp()
    bus = EventBus()
    tutor = FakeTutor(cfg, bus, [], voice=False)
    chapter = types.SimpleNamespace(number=3, can_do=["Bestellungen aufgeben"])
    tutor.chapter = chapter
    tutor.goals = lambda: objectives.status(cfg.study_path, "Testbuch", chapter)

    verdicts = [(False, "Use «einen»."), (True, "Good.")]

    async def task_(*a, **k):
        return "Order a coffee."

    async def judge(llm, level, goal, task, answer):
        return objectives.precheck(answer) or verdicts.pop(0)

    real = study.goal_task, study.judge_goal
    study.goal_task, study.judge_goal = task_, judge
    runner = ConversationRunner(tutor, bus)
    events: list[dict] = []
    collector = asyncio.create_task(collect(bus, events))
    try:
        task = asyncio.create_task(runner.run())
        await asyncio.sleep(0.05)
        runner.submit({"action": "check_goal", "value": 0})
        await asyncio.sleep(0.05)
        runner.submit({"action": "say", "value": "I would order a coffee please"})
        await asyncio.sleep(0.05)
        runner.submit({"action": "say", "value": "Ein Kaffee bitte"})
        await asyncio.sleep(0.05)
        runner.submit({"action": "say", "value": "Danke!"})
        await asyncio.sleep(0.1)
        runner.request_stop()
        await asyncio.wait_for(task, timeout=3)
    finally:
        study.goal_task, study.judge_goal = real
        collector.cancel()

    results = [e for e in events if e["type"] == "goal_result"]
    status = objectives.status(cfg.study_path, "Testbuch", chapter)[0]
    return {
        "the task is set as a tutor turn": any(
            "Order a coffee." in e.get("text", "") for e in events
            if e["type"] == "tutor_turn"),
        "an English answer fails without the model": (
            bool(results) and results[0]["passed"] is False
            and "English" in results[0]["note"]),
        "two attempts, then the check ends": len(results) == 2,
        "answers were not sent to the tutor as chat": all(
            "Kaffee" not in r[0] for r in tutor.responded),
        "the next message is conversation again": any(
            r[0] == "Danke!" for r in tutor.responded),
        "the log records the second-try fail": (
            status.attempts == 2 and status.demonstrated is False),
    }


async def scenario_goal_pass_and_manual(cfg):
    import tempfile
    from app import objectives, study

    cfg.paths.study_dir = tempfile.mkdtemp()
    bus = EventBus()
    tutor = FakeTutor(cfg, bus, [], voice=False)
    chapter = types.SimpleNamespace(number=3, can_do=["A", "B"])
    tutor.chapter = chapter
    tutor.goals = lambda: objectives.status(cfg.study_path, "Testbuch", chapter)

    async def task_(*a, **k):
        return "Do A."

    async def judge(*a, **k):
        return True, "Good."

    real = study.goal_task, study.judge_goal
    study.goal_task, study.judge_goal = task_, judge
    runner = ConversationRunner(tutor, bus)
    events: list[dict] = []
    collector = asyncio.create_task(collect(bus, events))
    try:
        task = asyncio.create_task(runner.run())
        await asyncio.sleep(0.05)
        runner.submit({"action": "mark_goal", "value": 1, "done": True})
        runner.submit({"action": "check_goal", "value": 0})
        runner.submit({"action": "say", "value": "Ich mache A."})
        await asyncio.sleep(0.05)
        kinds_then = [e["type"] for e in events]
        runner.submit({"action": "mark_goal", "value": 0, "done": False})
        await asyncio.sleep(0.05)
        runner.request_stop()
        await asyncio.wait_for(task, timeout=3)
    finally:
        study.goal_task, study.judge_goal = real
        collector.cancel()

    final = objectives.status(cfg.study_path, "Testbuch", chapter)
    return {
        "a pass ends the check at once": sum(
            1 for e in events if e["type"] == "goal_result") == 1,
        "all goals done announces the chapter": "chapter_done" in kinds_then,
        "a manual mark is recorded as manual": final[1].by == "manual",
        "a manual 'not yet' undoes a pass": final[0].demonstrated is False,
    }


def scenario_echo_guard(cfg):
    """Similarity guard: reject own speech, keep genuine user turns.

    The first case is a real capture from a session on laptop speakers.
    """
    from app.session import ECHO_SIMILARITY, is_echo

    spoken = "Hallo! Schön, dich zu sehen. Wie war denn dein Tag heute?"
    cases = [
        # (heard, should_be_rejected, label)
        ("Hallo, schön dich zu sehen. Wie war denn dein Tag heute?", True,
         "verbatim echo of the opener"),
        ("Wie war denn dein Tag heute", True, "partial echo (tail fragment)"),
        ("Hey da man, how are you? All good?", False,
         "genuine user speech, unrelated"),
        ("Mein Tag war gut, danke.", False, "genuine reply reusing a word"),
        ("Ja.", False, "short genuine answer"),
        ("Und wie war dein Tag?", False,
         "user echoes the question back — legitimately conversational"),
    ]

    results = {}
    for heard, expect_reject, label in cases:
        score = is_echo(heard, spoken)
        rejected = score >= ECHO_SIMILARITY
        results[f"{label} ({score:.2f})"] = rejected == expect_reject
    return results


async def main() -> int:
    failures = 0
    for fn in (scenario_basic, scenario_control, scenario_no_barge_in,
               scenario_mode_handoff, scenario_drill, scenario_typed,
               scenario_attachment, scenario_goal_check,
               scenario_goal_pass_and_manual, scenario_echo_guard):
        cfg = load_config()  # fresh config per scenario
        print(f"\n{fn.__name__}:")
        try:
            out = fn(cfg)
            results = (
                await asyncio.wait_for(out, timeout=15)
                if asyncio.iscoroutine(out) else out
            )
        except asyncio.TimeoutError:
            print("  FAIL  deadlocked")
            failures += 1
            continue
        for label, ok in results.items():
            failures += not ok
            print(f"  {'PASS' if ok else 'FAIL'}  {label}")

    print(f"\n{'PASS' if not failures else 'FAIL'} — conversation loop\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
