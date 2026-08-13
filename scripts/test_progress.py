"""Cross-session memory: does the tutor actually remember anything useful?

The test that matters is `test_recurring_mistake_is_found`. Grouping
corrections by their whole sentence finds nothing — a learner does not repeat a
mistake in the same words. It has to group by what *changed* inside them, so
that six der/das slips across six different sentences read as one habit.

The other risk is the opposite: turning noise into a "habit". A one-off is not
a pattern, and neither is a sentence rewritten end to end.

Pure stdlib. `python scripts/test_progress.py`
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import progress  # noqa: E402


def write_session(d: Path, name: str, **kw) -> None:
    payload = {
        "started": kw.get("started", "2026-08-01T10:00:00"),
        "updated": kw.get("updated", "2026-08-01T10:20:00"),
        "level": kw.get("level", "A1"),
        "scenario": kw.get("scenario", "kapitel-1"),
        "transcript": kw.get("transcript", [{"role": "user", "text": "hallo"}]),
        "corrections": kw.get("corrections", []),
        "vocab": kw.get("vocab", []),
    }
    (d / name).write_text(json.dumps(payload, ensure_ascii=False),
                          encoding="utf-8")


def corr(original: str, corrected: str, explanation: str = "") -> dict:
    return {"original": original, "corrected": corrected,
            "explanation": explanation}


def test_empty(d: Path) -> dict:
    p = progress.build(d)
    return {
        "no sessions means empty": p.is_empty,
        "brief is blank, not a stub sentence": p.brief() == "",
        "summary says so plainly": "No sessions yet" in p.summary(),
    }


def test_recurring_mistake_is_found(d: Path) -> dict:
    """Six der/das slips in six different sentences are one habit."""
    write_session(d, "a.json", corrections=[
        corr("Ich sehe der Buch.", "Ich sehe das Buch.", "Buch is neuter"),
        corr("Der Auto ist rot.", "Das Auto ist rot."),
    ])
    write_session(d, "b.json", corrections=[
        corr("Der Mädchen lacht.", "Das Mädchen lacht."),
        corr("Ich habe ein Buch.", "Ich habe ein Buch gelesen."),
    ])
    p = progress.build(d)
    top = p.mistakes[0]
    recurring = p.recurring()
    return {
        "the der/das habit is the top mistake": (top.wrong, top.right) == ("der", "das"),
        "counted across sessions and sentences": top.count == 3,
        "an explanation is kept for context": bool(top.explanation),
        "a one-off is not called recurring": all(
            m.right != "buch gelesen" for m in recurring
        ),
        "recurring is not empty": len(recurring) == 1,
        "the brief names it": "«der» → «das»" in p.brief(),
    }


def test_noise_is_not_a_pattern(d: Path) -> dict:
    write_session(d, "a.json", corrections=[
        corr("Ich gehe zu die Schule heute mit meinem Bruder.",
             "Heute gehe ich mit meinem Bruder in die Schule."),
        corr("", ""),
        corr("Guten Morgen.", "Guten Morgen."),
    ])
    p = progress.build(d)
    return {
        "a wholesale rewrite is not a pattern": p.mistakes == [],
        "nothing recurring to report": p.recurring() == [],
        "brief still mentions the session": "1 previous session" in p.brief(),
    }


def test_vocab_and_chapters(d: Path) -> dict:
    write_session(d, "a.json", scenario="kapitel-1",
                  updated="2026-08-01T10:20:00",
                  vocab=["der Morgen", "heißen"])
    write_session(d, "b.json", scenario="kapitel-3",
                  started="2026-08-05T10:00:00",
                  updated="2026-08-05T10:30:00",
                  vocab=["der Morgen", "bestellen"],
                  transcript=[{"role": "user", "text": "eins"},
                              {"role": "assistant", "text": "zwei"},
                              {"role": "user", "text": "drei"}])
    p = progress.build(d)
    due = [w.text for w in p.due_vocab()]
    return {
        "sessions counted": p.sessions == 2,
        "only user turns counted": p.turns == 3,
        "minutes summed": 49 < p.minutes < 51,
        "vocabulary deduplicated": len(p.vocab) == 3,
        "repeat sightings counted": p.vocab["der morgen"].seen == 2,
        "chapters recorded": dict(p.chapters) == {1: 1, 3: 1},
        "resumes from the latest session, not the first": p.last_chapter == 3,
        "least recently practised comes first": due[0] == "heißen",
        "brief states the chapter": "chapter 3" in p.brief(),
    }


def test_broken_file_is_survived(d: Path) -> dict:
    write_session(d, "good.json", vocab=["danke"])
    (d / "truncated.json").write_text('{"started": "2026-08', encoding="utf-8")
    p = progress.build(d)
    return {
        "the readable session still counts": p.sessions == 1,
        "the broken one is skipped, not fatal": "danke" in p.vocab,
    }


def main() -> int:
    fails = 0
    for fn in (test_empty, test_recurring_mistake_is_found,
               test_noise_is_not_a_pattern, test_vocab_and_chapters,
               test_broken_file_is_survived):
        print(f"\n{fn.__name__}:")
        with tempfile.TemporaryDirectory() as td:
            results = fn(Path(td))
        for label, ok in results.items():
            fails += not ok
            print(f"  {'PASS' if ok else 'FAIL'}  {label}")

    print(f"\n{'PASS' if not fails else 'FAIL'} — progress\n")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
