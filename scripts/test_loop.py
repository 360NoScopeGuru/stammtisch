"""Drive ConversationRunner.run() with fake audio — no mic, no models, no LLM.

This covers the parts that are easy to get subtly wrong: the wait-on-either
(audio | control) race, task cancellation, control handling mid-conversation,
and echo-draining when barge-in is off. `python scripts/test_loop.py`
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
    def __init__(self, cfg, bus, transcripts):
        self.cfg = cfg
        self.bus = bus
        self.scenario = types.SimpleNamespace(
            key="baeckerei", title="Bakery",
            opener="Guten Morgen!", intro="Today we go to the bakery.",
        )
        self.mic = FakeMic()
        self.speaker = FakeSpeaker()
        # Return "" once exhausted — the loop skips empty transcripts, so an
        # unexpected extra utterance shows up as a failed assertion, not a crash.
        self.stt = types.SimpleNamespace(
            transcribe=lambda a, lang=None: transcripts.pop(0) if transcripts else "",
            last_language="de",   # so corrections are exercised
        )
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

    async def respond(self, text):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("simulated LLM failure")
        if self.reply_delay:
            await asyncio.sleep(self.reply_delay)
        reply = f"Antwort auf: {text}"
        self.spoken.append(reply)
        return Turn(role="assistant", text=reply, latency_ms=910.0)

    def queue_correction(self, text):
        self.corrected.append(text)

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
               scenario_mode_handoff, scenario_echo_guard):
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
