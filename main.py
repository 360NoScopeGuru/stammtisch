"""Stammtisch — a local German conversation partner.

    python main.py                          # terminal session
    python main.py --web                    # browser UI at :8420
    python main.py --scenario baeckerei --level A2
    python main.py --chapter 3               # teach chapter 3 of your textbook
    python main.py --list-chapters
    python main.py --progress               # what you have covered so far
    python main.py --homework new           # set written work; then --homework do
    python main.py --doctor                 # why isn't it working?
    python main.py --list-scenarios
    python main.py --list-devices
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from app import curriculum, doctor, homework, progress, scenarios
from app.config import load_config
from app.events import EventBus

# Tutor and ConversationRunner pull in onnxruntime, sounddevice and the model
# stack. The informational commands below (--doctor above all) must work when
# some of that is missing or broken, so they are imported where they are used.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local German conversation practice")
    p.add_argument("--config", default=None)
    p.add_argument("--level", choices=["A1", "A2", "B1", "B2", "C1"])
    p.add_argument("--mode", choices=["mentor", "practice"],
                   help="mentor teaches in English; practice is German immersion")
    p.add_argument("--scenario")
    p.add_argument("--chapter", type=int,
                   help="chapter of the ingested textbook (0 uses --scenario)")
    p.add_argument("--course", help="course file stem in paths.courses_dir")
    p.add_argument("--web", action="store_true", help="serve the browser UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8420)
    p.add_argument("--list-scenarios", action="store_true")
    p.add_argument("--list-chapters", action="store_true",
                   help="show the ingested textbook's chapters")
    p.add_argument("--doctor", action="store_true",
                   help="check every precondition and say what to fix")
    p.add_argument("--homework", nargs="?", const="show",
                   choices=["show", "new", "do", "mark"],
                   help="show / set / answer / mark written work")
    p.add_argument("--progress", action="store_true",
                   help="what you have covered so far, across all sessions")
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
    from app.session import ConversationRunner
    from app.tutor import Tutor

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
    except OSError as e:
        # Almost always the microphone or the audio device.
        print(f"\n  ⚠  audio device problem: {e}\n"
              f"     python main.py --list-devices\n"
              f"     python main.py --doctor\n")
        return 1
    finally:
        runner.request_stop()
        tutor.speaker.cancel()
        await asyncio.sleep(0.1)  # let the printer flush the summary
        printer.cancel()
        await tutor.aclose()
    return 0


async def run_homework(cfg, action: str) -> int:
    """Set, answer or mark written work from the terminal.

    Deliberately usable without the microphone, Whisper or Piper: homework is
    the part you do on a train. Only `new` and `mark` need the LLM at all.
    """
    from app.llm import LlmClient
    from app.tutor import Tutor

    current = homework.latest(cfg.homework_path)

    if action == "show":
        if current is None:
            print("\n  No homework yet. Set some with:  python main.py "
                  "--homework new\n")
            return 0
        print(current.as_text(show_answers=current.is_marked))
        if not current.submitted:
            print("  Answer it with:  python main.py --homework do\n")
        return 0

    if action in ("do", "mark") and current is None:
        print("\n  No homework yet. Set some with:  python main.py "
              "--homework new\n")
        return 0

    # Answering needs nobody but the learner, so do it before waking the LLM.
    if action == "do":
        print(current.as_text())
        print("  Type each answer and press Enter. Leave blank to skip.\n")
        given = []
        for i, ex in enumerate(current.exercises):
            print(f"  {i + 1}. {ex.prompt}")
            given.append(input("     > ").strip())
        current.submit(given)
        homework.save(current, cfg.homework_path)
        print("\n  Saved. Marking...")
        action = "mark"
    elif action == "mark" and not current.submitted:
        print("\n  Nothing handed in yet. Answer it with:  python main.py "
              "--homework do\n")
        return 0

    client = LlmClient(cfg)
    ok, msg = await client.health()
    if not ok:
        print(f"\n  cannot reach the LLM at {cfg.llm.base_url} - {msg}\n"
              f"     For Ollama: ollama serve\n"
              f"     python main.py --doctor\n")
        await client.aclose()
        return 1

    # Build just enough Tutor to reach set_homework/mark_homework, without
    # loading Whisper, Piper or opening the microphone.
    tutor = Tutor.__new__(Tutor)
    tutor.cfg, tutor.llm, tutor.bus = cfg, client, EventBus()
    tutor.course = curriculum.find(cfg.courses_path, cfg.tutor.course)
    tutor.progress = progress.build(cfg.sessions_path)
    cfg.tutor.chapter = tutor.resume_chapter()
    tutor.chapter = tutor._resolve_chapter(cfg.tutor.chapter)
    tutor.scenario = tutor._scenario_for(tutor.chapter)

    try:
        if action == "new":
            print(f"\n  Setting homework on {tutor.scenario.title}...")
            assignment = await tutor.set_homework()
            if assignment is None:
                print("  The model did not produce anything usable. "
                      "Try again.\n")
                return 1
            print(assignment.as_text())
            print("  Answer it with:  python main.py --homework do\n")
            return 0

        marked = await tutor.mark_homework(current)
        print(marked.as_text(show_answers=True))
        return 0
    finally:
        await client.aclose()


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
    if args.course:
        cfg.tutor.course = args.course
    if args.scenario:
        if args.scenario not in scenarios.SCENARIOS:
            print(f"Unknown scenario {args.scenario!r}.\n{scenarios.listing()}")
            return 2
        cfg.tutor.scenario = args.scenario
        # An explicit scenario wins over the book, and must not be undone by
        # resume quietly putting the last chapter back.
        cfg.tutor.chapter = 0
        cfg.tutor.resume = False
    if args.chapter is not None:
        cfg.tutor.chapter = args.chapter
        if args.chapter == 0:
            cfg.tutor.resume = False

    if args.homework:
        return asyncio.run(run_homework(cfg, args.homework))

    if args.doctor:
        checks = doctor.run_all(cfg)
        print(doctor.report(checks))
        return 1 if any(c.bad for c in checks) else 0

    if args.progress:
        print(progress.build(cfg.sessions_path).summary())
        return 0

    if args.list_chapters:
        course = curriculum.find(cfg.courses_path, cfg.tutor.course)
        if not course:
            print(f"\n  No course in {cfg.courses_path}.\n"
                  f"  Ingest one first:\n"
                  f"    python scripts/ingest_textbook.py \"path/to/book.pdf\"\n")
            return 1
        print(f"\n  {course.title}  ({course.level})\n")
        print(course.listing() + "\n")
        return 0

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
