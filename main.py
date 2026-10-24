"""Stammtisch — a local German tutor.

    python main.py                          # type to the tutor in the terminal
    python main.py --web                    # browser UI at :8420 (recommended)
    python main.py --voice                  # talk instead of typing
    python main.py --chapter 3               # teach chapter 3 of your textbook
    python main.py --class "Kapitel 3, S. 44: Akkusativ" --attach board.jpg
    python main.py --handout nomen.pdf      # der/das/die vocabulary from a handout
    python main.py --review                 # flashcards that are due
    python main.py --goals                  # this chapter's goals, and which are done
    python main.py --homework new           # set written work; then --homework do
    python main.py --progress               # what you have covered so far
    python main.py --doctor                 # why isn't it working?
    python main.py --list-chapters | --list-scenarios | --list-devices
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading
from datetime import date
from pathlib import Path

from app import (classlog, curriculum, doctor, documents, handouts, homework,
                 objectives, progress, scenarios, srs)
from app.config import load_config
from app.events import EventBus

# Tutor and ConversationRunner pull in the model stack. The informational
# commands below (--doctor above all) must work when some of that is missing
# or broken, so they are imported where they are used.

CHAT_HELP = """
  Type in English or German and press Enter. Commands:
    /attach <file> [pages]   attach a PDF or photo to your next message
                             e.g. /attach worksheet.pdf 2-3
    /goals                   this chapter's goals
    /check <n>               get a small task for goal n
    /done <n> | /undo <n>    mark goal n done or not done yourself
    /chapter <n>             switch chapter          /next   next chapter
    /mode mentor|practice    teach me / practise
    /voice on|off            talk instead of typing (loads speech recognition)
    /speak on|off            read replies aloud
    /stop                    stop the current reply
    /quit                    end the session (Ctrl+C works too)
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A local German tutor")
    p.add_argument("--config", default=None)
    p.add_argument("--level", choices=["A1", "A2", "B1", "B2", "C1"])
    p.add_argument("--mode", choices=["mentor", "practice"],
                   help="mentor teaches in English; practice is German immersion")
    p.add_argument("--scenario")
    p.add_argument("--chapter", type=int,
                   help="chapter of the ingested textbook (0 uses --scenario)")
    p.add_argument("--course", help="course file stem in paths.courses_dir")
    p.add_argument("--voice", action="store_true",
                   help="talk instead of typing (needs the microphone and Whisper)")
    p.add_argument("--mute", action="store_true",
                   help="do not read replies aloud")
    p.add_argument("--web", action="store_true", help="serve the browser UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8420)
    p.add_argument("--class", dest="class_note", metavar="NOTE",
                   help="log what your class covered (add --attach for photos/pages)")
    p.add_argument("--attach", nargs="+", default=[], metavar="FILE",
                   help="files for --class or --handout")
    p.add_argument("--pages", default="",
                   help="which pages of a PDF to read, e.g. 2-4")
    p.add_argument("--day", help="date of the class, YYYY-MM-DD (default today)")
    p.add_argument("--handout", metavar="FILE",
                   help="read der/das/die vocabulary from a handout")
    p.add_argument("--name", help="name for the handout's deck")
    p.add_argument("--review", action="store_true", help="flashcards that are due")
    p.add_argument("--goals", action="store_true",
                   help="this chapter's goals and which are done")
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


# ---- the conversation in a terminal -------------------------------------------

async def print_events(bus: EventBus) -> None:
    """Render the event stream to the terminal."""
    q = bus.subscribe()
    live = False
    while True:
        e = await q.get()
        t = e["type"]
        if t == "mode_changed":
            label = "TEACHING (English)" if e["mode"] == "mentor" else "PRACTICE (German)"
            print(f"\n  ── switched to {label} ──\n")
        elif t == "user_turn":
            if e.get("typed") is False or e.get("drill"):
                print(f"  🎤 {e['text']}")
        elif t == "tutor_start":
            print("\n  🗣  ", end="", flush=True)
            live = True
        elif t == "tutor_delta":
            if not live:
                print("\n  🗣  ", end="", flush=True)
                live = True
            print(e["text"].replace("\n", "\n      "), end="", flush=True)
        elif t == "tutor_turn":
            tag = (f"  ({e['latency_ms'] / 1000:.1f} s)"
                   if e.get("latency_ms") else "")
            if live:
                print(tag + "\n")
            else:
                text = e["text"].replace("\n", "\n      ")
                print(f"\n  🗣  {text}\n")
            live = False
        elif t == "corrections":
            for c in e.get("items", []):
                print(f"      ✗ {c.get('original', '')}")
                print(f"      ✓ {c.get('corrected', '')}"
                      + (f"  — {c['explanation']}" if c.get("explanation") else ""))
        elif t == "goals":
            if e.get("done"):
                print("  ✓ every goal in this chapter is done — /next moves on\n")
        elif t == "goal_result":
            print(f"  {'✓ goal done' if e['passed'] else '✗ not yet'}: {e['goal']}")
        elif t == "notice":
            print(f"  … {e['message']}")
        elif t in ("error", "fatal"):
            live = False
            print(f"\n  ⚠  {e['message']}\n")
        elif t == "session_end":
            print(e["summary"])


def _read_stdin(loop: asyncio.AbstractEventLoop, q: asyncio.Queue) -> None:
    """Typed lines, read on a thread so the event loop never blocks on input."""
    while True:
        try:
            line = sys.stdin.readline()
        except (OSError, ValueError):
            line = ""
        loop.call_soon_threadsafe(q.put_nowait, line if line else None)
        if not line:
            return


def _show_goals(tutor) -> None:
    goals = tutor.goals()
    if not goals:
        print("  No chapter goals — ingest your textbook and pick a chapter.\n")
        return
    print(f"\n  Goals for {tutor.scenario.title}:")
    for g in goals:
        print(f"    {g.index + 1}. [{'✓' if g.demonstrated else ' '}] {g.text}")
    print()


async def chat_commands(runner, tutor) -> None:
    """Turn typed lines into messages and controls."""
    loop = asyncio.get_running_loop()
    lines: asyncio.Queue = asyncio.Queue()
    threading.Thread(target=_read_stdin, args=(loop, lines), daemon=True).start()
    pending: list[dict] = []

    def goal_arg(arg: str) -> int | None:
        try:
            return int(arg) - 1
        except ValueError:
            print("  Which goal? e.g. /check 2  (see /goals)")
            return None

    while True:
        line = await lines.get()
        if line is None:                      # stdin closed
            runner.request_stop()
            return
        text = line.strip()
        if not text:
            continue
        if not text.startswith("/"):
            runner.submit({"action": "say", "value": text, "attachments": pending})
            pending = []
            continue

        cmd, _, arg = text[1:].partition(" ")
        cmd, arg = cmd.lower(), arg.strip()
        if cmd in ("quit", "exit", "q"):
            runner.request_stop()
            return
        if cmd in ("help", "?"):
            print(CHAT_HELP)
        elif cmd == "attach":
            if not arg:
                print("  /attach <file> [pages]")
                continue
            path, pages = arg, ""
            if not Path(path.strip('"')).exists() and " " in arg:
                path, pages = arg.rsplit(" ", 1)
            try:
                doc = await asyncio.to_thread(documents.load_path, path.strip('"'))
                doc.select(pages)                   # check the range now
            except documents.DocumentError as e:
                print(f"  ⚠  {e}")
                continue
            doc_id = f"cli{len(tutor.documents)}"
            tutor.documents[doc_id] = doc
            pending.append({"id": doc_id, "pages": pages})
            span = f", pages {pages}" if pages else f", {doc.page_count} page(s)"
            print(f"  📎 {doc.name}{span} — now type your question.")
        elif cmd == "goals":
            _show_goals(tutor)
        elif cmd == "check" and (i := goal_arg(arg)) is not None:
            runner.submit({"action": "check_goal", "value": i})
        elif cmd in ("done", "undo") and (i := goal_arg(arg)) is not None:
            runner.submit({"action": "mark_goal", "value": i, "done": cmd == "done"})
        elif cmd == "chapter" and arg.isdigit():
            runner.submit({"action": "set_chapter", "value": int(arg)})
        elif cmd == "next":
            runner.submit({"action": "next_chapter"})
        elif cmd == "mode" and arg in ("mentor", "practice"):
            runner.submit({"action": "set_mode", "value": arg})
        elif cmd == "voice" and arg in ("on", "off"):
            runner.submit({"action": "set_input", "value": "voice" if arg == "on" else "text"})
        elif cmd == "speak" and arg in ("on", "off"):
            runner.submit({"action": "set_speak", "value": arg == "on"})
        elif cmd == "stop":
            runner.interrupt()
        else:
            print("  Unknown command — /help lists them.")


async def run_cli(cfg) -> int:
    from app.session import ConversationRunner
    from app.tutor import Tutor

    bus = EventBus()
    tutor = Tutor(cfg, bus=bus)
    runner = ConversationRunner(tutor, bus)
    printer = asyncio.create_task(print_events(bus))
    commands = asyncio.create_task(chat_commands(runner, tutor))

    mode_label = "mentor (English)" if cfg.tutor.mode == "mentor" else "practice (German)"
    print(f"\n  {tutor.scenario.title}  ·  {cfg.tutor.level}  ·  {mode_label}")
    how = "Talk, or type" if tutor.listening else "Type"
    print(f"  {how} and press Enter. /help for commands, Ctrl+C to end.\n")

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
        commands.cancel()
        tutor.speaker.cancel()
        await asyncio.sleep(0.1)  # let the printer flush the summary
        printer.cancel()
        await tutor.aclose()
    return 0


# ---- study commands that need the model but not the conversation --------------

async def _llm_or_fail(cfg):
    from app.llm import LlmClient

    client = LlmClient(cfg)
    ok, msg = await client.health()
    if not ok:
        print(f"\n  cannot reach the LLM at {cfg.llm.base_url} - {msg}\n"
              f"     For Ollama: ollama serve\n"
              f"     python main.py --doctor\n")
        await client.aclose()
        return None
    return client


def _lite_tutor(cfg, client):
    """Just enough Tutor for the course, chapter and homework methods,
    without loading Whisper or Piper or opening any audio device."""
    from app.tutor import Tutor

    tutor = Tutor.__new__(Tutor)
    tutor.cfg, tutor.llm, tutor.bus = cfg, client, EventBus()
    tutor.course = curriculum.find(cfg.courses_path, cfg.tutor.course)
    tutor.progress = progress.build(cfg.sessions_path, cfg.homework_path)
    cfg.tutor.chapter = tutor.resume_chapter()
    tutor.chapter = tutor._resolve_chapter(cfg.tutor.chapter)
    tutor.scenario = tutor._scenario_for(tutor.chapter)
    return tutor


def _load_attachments(paths: list[str], pages: str):
    out = []
    for path in paths:
        doc = documents.load_path(path)
        out.append((doc, doc.select(pages if doc.kind == documents.PDF else "")))
    return out


async def run_class(cfg, note: str, attach: list[str], pages: str,
                    day: str | None) -> int:
    from app import study

    try:
        attached = _load_attachments(attach, pages)
    except documents.DocumentError as e:
        print(f"\n  ⚠  {e}\n")
        return 1
    client = await _llm_or_fail(cfg)
    if client is None:
        return 1
    try:
        when = date.fromisoformat(day) if day else date.today()
        print("\n  Reading your class notes…")
        course = curriculum.find(cfg.courses_path, cfg.tutor.course)
        entry = await study.log_class(client, course, note, attached, when)
    finally:
        await client.aclose()
    if entry is None:
        print("  The model did not return anything usable. Try again.\n")
        return 1
    classlog.save(entry, cfg.study_path)
    print("\n  Saved:\n  " + entry.describe().replace(". ", ".\n  ") + "\n")
    if any(w.checked is None for w in entry.vocab):
        print("  (read from a photo — the words could not be checked against any "
              "text, so glance over them)\n")
    if entry.dropped:
        print(f"  Left out, not in the source: {', '.join(entry.dropped)}\n")
    return 0


async def run_handout(cfg, path: str, pages: str, name: str | None) -> int:
    from app import study

    try:
        doc = documents.load_path(path)
        chosen = doc.select(pages)
    except documents.DocumentError as e:
        print(f"\n  ⚠  {e}\n")
        return 1
    client = await _llm_or_fail(cfg)
    if client is None:
        return 1
    try:
        deck = await study.read_handout(
            client, doc, name or Path(path).stem, chosen,
            progress=lambda m: print(f"  … {m}"))
    finally:
        await client.aclose()
    if not deck.entries:
        print("\n  No nouns in article columns were found on those pages.\n")
        return 1
    handouts.save_deck(deck, cfg.study_path)
    counts = deck.counts()
    print(f"\n  {counts['confirmed']} confirmed and added to your flashcards; "
          f"{counts['pending']} need you.\n")

    for i, item in enumerate(deck.entries):
        if item.status != handouts.PENDING:
            continue
        readings = ", ".join(f"{k}: {v}" for k, v in item.readings.items())
        answer = input(f"  {item.noun}{' — ' + item.en if item.en else ''}  "
                       f"[{readings}]\n    der / das / die, s to skip, x to remove > "
                       ).strip().lower()
        if answer in handouts.ARTICLES:
            handouts.decide(cfg.study_path, deck.slug, i, handouts.CONFIRMED,
                            de=f"{answer} {item.noun}")
        elif answer == "x":
            handouts.decide(cfg.study_path, deck.slug, i, handouts.REJECTED)
    print()
    return 0


def run_review(cfg) -> int:
    prog = progress.build(cfg.sessions_path, cfg.homework_path)
    done = 0
    while True:
        cards = srs.collect(cfg.study_path, prog)
        states = srs.schedule(cfg.study_path)
        queue = srs.queue(cards, states, date.today())
        if not queue:
            print(f"\n  {'Done for today.' if done else 'Nothing due.'} "
                  f"{len(cards)} card(s) in all.\n")
            return 0
        card = queue[0]
        label = "Fix this sentence" if card.kind == srs.MISTAKE else "In German"
        print(f"\n  {label}:  {card.front}   ({len(queue)} to go)")
        try:
            given = input("  > ")
        except EOFError:
            return 0
        if given.strip().lower() in ("q", "quit", "/quit"):
            return 0
        result = srs.grade(card, given)
        srs.record(cfg.study_path, card, result.q, given)
        print(f"  {'✓' if result.correct else '✗'} {result.note}")
        done += 1


def show_goals(cfg) -> int:
    course = curriculum.find(cfg.courses_path, cfg.tutor.course)
    if not course:
        print("\n  No course ingested — see python main.py --list-chapters.\n")
        return 1
    client = None
    tutor = _lite_tutor(cfg, client)
    if not tutor.chapter:
        print("\n  No chapter selected.\n")
        return 1
    goals = objectives.status(cfg.study_path, course.title, tutor.chapter)
    print(f"\n  {tutor.scenario.title}")
    for g in goals:
        print(f"    [{'✓' if g.demonstrated else ' '}] {g.text}"
              + (f"   ({g.attempts} check(s))" if g.attempts else ""))
    if objectives.chapter_done(goals):
        print("\n  Every goal done — time for the next chapter.")
    print("\n  Check a goal in a session with /check <n>.\n")
    return 0


async def run_homework(cfg, action: str) -> int:
    """Set, answer or mark written work from the terminal.

    Deliberately usable without the microphone, Whisper or Piper: homework is
    the part you do on a train. Only `new` and `mark` need the LLM at all.
    """
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

    client = await _llm_or_fail(cfg)
    if client is None:
        return 1
    tutor = _lite_tutor(cfg, client)

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
        level=logging.DEBUG if args.verbose else logging.WARNING,
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
    if args.voice:
        cfg.tutor.input = "voice"
    if args.mute:
        cfg.tutor.speak = False
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

    if args.class_note is not None or (args.attach and not args.handout):
        return asyncio.run(run_class(cfg, args.class_note or "", args.attach,
                                     args.pages, args.day))

    if args.handout:
        return asyncio.run(run_handout(cfg, args.handout, args.pages, args.name))

    if args.review:
        return run_review(cfg)

    if args.goals:
        return show_goals(cfg)

    if args.doctor:
        checks = doctor.run_all(cfg)
        print(doctor.report(checks))
        return 1 if any(c.bad for c in checks) else 0

    if args.progress:
        print(progress.build(cfg.sessions_path, cfg.homework_path).summary())
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

    # Logging at INFO is useful for a server in a console of its own; in a
    # terminal conversation it would interleave with the tutor's replies.
    if args.web and not args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    if args.web:
        # uvicorn creates and owns its own event loop, so this must NOT run
        # inside asyncio.run() — hence main() being sync.
        from app.server import serve
        serve(cfg, host=args.host, port=args.port)
        return 0

    return asyncio.run(run_cli(cfg))


if __name__ == "__main__":
    # The terminal on Windows defaults to a code page that cannot print «, →
    # or ✓, and a crash on a tutor's arrow is a silly way to lose a session.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
