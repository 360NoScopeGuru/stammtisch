"""Stammtisch — a local German conversation partner.

    python main.py                          # terminal session
    python main.py --web                    # browser UI at :8420
    python main.py --scenario baeckerei --level A2
    python main.py --list-scenarios
    python main.py --list-devices
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from app import scenarios
from app.config import load_config
from app.events import EventBus
from app.session import ConversationRunner
from app.tutor import Tutor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local German conversation practice")
    p.add_argument("--config", default=None)
    p.add_argument("--level", choices=["A1", "A2", "B1", "B2", "C1"])
    p.add_argument("--mode", choices=["mentor", "practice"],
                   help="mentor teaches in English; practice is German immersion")
    p.add_argument("--scenario")
    p.add_argument("--web", action="store_true", help="serve the browser UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8420)
    p.add_argument("--list-scenarios", action="store_true")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


async def print_events(bus: EventBus) -> None:
    """Render the event stream to the terminal."""
    q = bus.subscribe()
    icons = {"user_turn": "🎤", "tutor_turn": "🗣 "}
    while True:
        e = await q.get()
        t = e["type"]
        if t == "mode_changed":
            label = "TEACHING (English)" if e["mode"] == "mentor" else "PRACTICE (German)"
            print(f"\n  ── switched to {label} ──\n")
        elif t in icons:
            tag = f"  ({e['latency_ms']:.0f} ms)" if e.get("latency_ms") else ""
            print(f"  {icons[t]} {e['text']}{tag}")
            if t == "tutor_turn":
                print()
        elif t == "corrections":
            for c in e.get("items", []):
                print(f"      ✗ {c.get('original', '')}")
                print(f"      ✓ {c.get('corrected', '')}")
        elif t in ("error", "fatal"):
            print(f"  ⚠  {e['message']}\n")
        elif t == "session_end":
            print(e["summary"])


async def run_cli(cfg) -> int:
    bus = EventBus()
    tutor = Tutor(cfg, bus=bus)
    runner = ConversationRunner(tutor, bus)
    printer = asyncio.create_task(print_events(bus))

    mode_label = "mentor (English)" if cfg.tutor.mode == "mentor" else "practice (German)"
    print(f"\n  {tutor.scenario.title}  ·  {cfg.tutor.level}  ·  {mode_label}")
    print("  Just start talking. Ctrl+C to end.\n")

    try:
        await runner.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except RuntimeError as e:
        print(f"\n  ⚠  {e}\n")
        return 1
    finally:
        runner.request_stop()
        tutor.speaker.cancel()
        await asyncio.sleep(0.1)  # let the printer flush the summary
        printer.cancel()
        await tutor.aclose()
    return 0


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-14s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.list_scenarios:
        print("\nSzenarien:\n" + scenarios.listing())
        return 0

    if args.list_devices:
        from app.audio_io import list_devices
        print(list_devices())
        return 0

    cfg = load_config(args.config)
    if args.level:
        cfg.tutor.level = args.level
    if args.mode:
        cfg.tutor.mode = args.mode
    if args.scenario:
        if args.scenario not in scenarios.SCENARIOS:
            print(f"Unknown scenario {args.scenario!r}.\n{scenarios.listing()}")
            return 2
        cfg.tutor.scenario = args.scenario

    if args.web:
        # uvicorn creates and owns its own event loop, so this must NOT run
        # inside asyncio.run() — hence main() being sync.
        from app.server import serve
        serve(cfg, host=args.host, port=args.port)
        return 0

    return asyncio.run(run_cli(cfg))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
