"""A killed session must still be on disk.

Saving used to run only from `Tutor.aclose()`, i.e. only on a clean shutdown.
Since this app is ended with Ctrl+C or by closing the console, `sessions/` was
empty after every real session — the review and the Anki export had never once
produced a file, and a session's worth of corrections went with it.

The important test here is the third one: it starts a real Python process,
writes turns, and kills it without any cleanup. The first two only check the
mechanics.

    python scripts/test_persistence.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config  # noqa: E402
from app.review import SessionLog  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def _session(tmp: Path) -> SessionLog:
    cfg = load_config()
    cfg.paths.sessions_dir = str(tmp)
    return SessionLog(cfg, level=cfg.tutor.level, scenario=cfg.tutor.scenario)


def test_written_per_turn(tmp: Path) -> dict:
    log = _session(tmp)
    before = list(tmp.glob("*.json"))

    log.add_turn("user", "Guten Morgen.")
    after_one = list(tmp.glob("*.json"))

    log.add_turn("assistant", "Guten Morgen! Wie geht es dir?", 820.0)
    log.add_feedback("Guten Morgen.", [], ["der Morgen"])

    data = json.loads(after_one[0].read_text(encoding="utf-8"))
    return {
        "nothing written before the first turn": not before,
        "file exists after one turn": len(after_one) == 1,
        "turn is in it": data["transcript"][0]["text"] == "Guten Morgen.",
        "anki csv written too": len(list(tmp.glob("*_anki.csv"))) == 1,
        "no .tmp files left behind": not list(tmp.glob("*.tmp")),
    }


def test_scenario_change_keeps_one_file(tmp: Path) -> dict:
    log = _session(tmp)
    log.add_turn("user", "eins")
    log.cfg.tutor.scenario = "baeckerei"      # as a mid-session switch would
    log.cfg.tutor.level = "A2"
    log.add_turn("user", "zwei")

    files = list(tmp.glob("*.json"))
    data = json.loads(files[0].read_text(encoding="utf-8"))
    return {
        "still exactly one session file": len(files) == 1,
        "both turns present": len(data["transcript"]) == 2,
        "live scenario recorded": data["scenario"] == "baeckerei",
        "live level recorded": data["level"] == "A2",
        "original scenario retained": data["started_as"]["scenario"] == "grundlagen",
    }


def test_survives_a_kill(tmp: Path) -> dict:
    """The real case: no aclose(), no atexit, no finally. Just death."""
    script = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(REPO)!r})
        from app.config import load_config
        from app.review import SessionLog
        cfg = load_config()
        cfg.paths.sessions_dir = {str(tmp)!r}
        s = SessionLog(cfg, level="A1", scenario="grundlagen")
        s.add_turn("user", "Ich heisse Vamshee.")
        s.add_turn("assistant", "Freut mich!", 700.0)
        s.add_feedback("Ich heisse Vamshee.", [], ["heissen"])
        print("READY", flush=True)
        time.sleep(60)
    """)
    path = tmp / "_killme.py"
    path.write_text(script, encoding="utf-8")

    proc = subprocess.Popen(
        [sys.executable, str(path)], stdout=subprocess.PIPE, text=True
    )
    assert proc.stdout is not None
    ready = proc.stdout.readline().strip()
    proc.kill()                     # SIGKILL equivalent — no cleanup runs at all
    proc.wait(timeout=10)

    files = [p for p in tmp.glob("*.json")]
    data = json.loads(files[0].read_text(encoding="utf-8")) if files else {}
    return {
        "child reached the kill point": ready == "READY",
        "process really was killed": proc.returncode != 0,
        "session file survived the kill": len(files) == 1,
        "both turns survived": len(data.get("transcript", [])) == 2,
        "vocabulary survived": data.get("vocab") == ["heissen"],
        "json is not truncated": bool(data.get("started")),
    }


def main() -> int:
    fails = 0
    for fn in (test_written_per_turn, test_scenario_change_keeps_one_file,
               test_survives_a_kill):
        print(f"\n{fn.__name__}:")
        with tempfile.TemporaryDirectory() as td:
            results = fn(Path(td))
        for label, ok in results.items():
            fails += not ok
            print(f"  {'PASS' if ok else 'FAIL'}  {label}")

    print(f"\n{'PASS' if not fails else 'FAIL'} — session persistence\n")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
