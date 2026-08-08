"""Exercise the WebSocket plumbing without loading models or opening the mic.

Stubs out the Tutor and the conversation loop, then drives the real server:
event delivery, backlog replay for a late-joining browser, and control
round-trips. `python scripts/test_web.py`
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import server  # noqa: E402
from app.config import load_config  # noqa: E402
from app.events import EventBus  # noqa: E402


class StubSession:
    transcript: list = []
    all_corrections: list = []
    all_vocab: list = []

    def summary(self) -> str:
        return "stub summary"


class StubTutor:
    """Only what create_app / _publish_config / /api/state actually touch."""

    def __init__(self, cfg, bus=None):
        self.cfg = cfg
        self.bus = bus or EventBus()
        self.scenario = types.SimpleNamespace(
            key="baeckerei", title="At the bakery",
            opener="Guten Morgen!", intro="Today we go to the bakery.",
        )
        self.session = StubSession()
        self.levels: list[str] = []
        self.scenarios: list[str] = []
        self.modes: list[str] = []

    @property
    def mode(self):
        return self.cfg.tutor.mode

    def set_mode(self, v):
        self.modes.append(v)
        return False  # skip the opener path, which would need audio

    def set_level(self, v):
        self.levels.append(v)
        self.cfg.tutor.level = v
        return True

    def set_scenario(self, v):
        self.scenarios.append(v)
        return False  # skip the opener path, which would need audio

    async def aclose(self):
        pass


def main() -> int:
    cfg = load_config()
    failures = 0

    def check(label, cond, extra=""):
        nonlocal failures
        failures += not cond
        print(f"  {'PASS' if cond else 'FAIL'}  {label}{'' if cond else '  ' + extra}")

    # Neutralise the parts that need hardware.
    server.Tutor = StubTutor
    orig_run = server.ConversationRunner.run

    async def noop_run(self):
        return None

    server.ConversationRunner.run = noop_run

    try:
        app = server.create_app(cfg)
        with TestClient(app) as client:
            tutor = app.state.tutor
            bus = app.state.bus

            r = client.get("/")
            check("GET / serves the page", r.status_code == 200
                  and "Stammtisch" in r.text, f"status={r.status_code}")

            r = client.get("/api/state")
            check("GET /api/state returns JSON", r.status_code == 200
                  and "level" in r.json())

            # Publish BEFORE connecting: a late browser must still see it.
            bus.publish("user_turn", text="Ich hätte gern ein Brot.")

            with client.websocket_connect("/ws") as ws:
                seen, deadline = [], 25
                cfgs = 0
                while len(seen) < 1 or cfgs < 1:
                    if deadline <= 0:
                        break
                    e = ws.receive_json()
                    deadline -= 1
                    if e["type"] == "user_turn":
                        seen.append(e)
                    elif e["type"] == "config":
                        cfgs += 1

                check("backlog replayed to a late client",
                      any("Brot" in e.get("text", "") for e in seen))
                check("config sent on connect", cfgs >= 1)

                # Live event after connect.
                bus.publish("tutor_turn", text="Gerne!", latency_ms=880)
                live = None
                for _ in range(20):
                    e = ws.receive_json()
                    if e["type"] == "tutor_turn":
                        live = e
                        break
                check("live event delivered", live is not None
                      and live["text"] == "Gerne!")
                check("latency travels with the turn",
                      bool(live) and live.get("latency_ms") == 880)

                # Corrections arriving late must carry their payload intact.
                bus.publish("corrections", utterance="ein Brot",
                            items=[{"original": "ein Brot",
                                    "corrected": "ein Brot, bitte",
                                    "explanation": "politeness"}],
                            vocab=["die Bäckerei"])
                corr = None
                for _ in range(20):
                    e = ws.receive_json()
                    if e["type"] == "corrections":
                        corr = e
                        break
                check("corrections delivered with items+vocab",
                      bool(corr) and corr["items"][0]["corrected"] == "ein Brot, bitte"
                      and corr["vocab"] == ["die Bäckerei"])

                # Browser -> server controls. Do NOT receive_json() here: the
                # server has nothing queued and TestClient's recv blocks.
                ws.send_json({"action": "set_level", "value": "B2"})
                ws.send_json({"action": "set_scenario", "value": "arzttermin"})
                ws.send_json({"action": "set_mode", "value": "practice"})
                # Round-trip a publish to prove both sends were processed.
                bus.publish("ping")
                for _ in range(20):
                    if ws.receive_json()["type"] == "ping":
                        break

            # We replaced run(), so nothing is draining the control queue.
            # Apply them by hand to verify they were enqueued correctly.
            runner = app.state.runner
            queued = []
            while not runner.controls.empty():
                queued.append(runner.controls.get_nowait())
            asyncio.run(_apply_all(runner, queued))

            check("set_level control reached the tutor", "B2" in tutor.levels,
                  f"got {tutor.levels}")
            check("set_scenario control reached the tutor",
                  "arzttermin" in tutor.scenarios, f"got {tutor.scenarios}")
            check("set_mode control reached the tutor",
                  "practice" in tutor.modes, f"got {tutor.modes}")

            # Bus must not wedge when a subscriber stops draining.
            q = bus.subscribe()
            for i in range(600):
                bus.publish("spam", i=i)
            check("bus survives a stalled subscriber", q.qsize() <= 256,
                  f"qsize={q.qsize()}")
            bus.unsubscribe(q)

    finally:
        server.ConversationRunner.run = orig_run

    print(f"\n{'PASS' if not failures else 'FAIL'} — web layer\n")
    return 1 if failures else 0


async def _apply_all(runner, msgs):
    for m in msgs:
        await runner._apply_control(m)


if __name__ == "__main__":
    raise SystemExit(main())
