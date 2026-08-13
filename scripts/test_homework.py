"""Homework: marking that is fair, and model output that cannot be trusted.

Two things are being defended here.

**Unfair marks.** A learner cannot tell a wrong mark from a right one, and one
unfair mark teaches them to distrust every other. Answering a multiple choice
"C) Guten Tag!" when the key is "C", or filling a gap by writing the whole
sentence, is a person answering a question — not a mistake. Both were marked
wrong by the first version of this.

**Model output.** Marks come back renumbered, out of range, or as the wrong
type. An answer silently left unmarked looks exactly like one the marker chose
to ignore, so the mismatch has to be handled rather than dropped.

Pure stdlib. `python scripts/test_homework.py`
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import homework  # noqa: E402
from app.homework import CLOSED, OPEN, Assignment, Exercise  # noqa: E402


def gap(answer="heiße", alts=None):
    return Exercise(kind=CLOSED, prompt="Ich ___ Anna.", answer=answer,
                    alternatives=alts or [])


def build(exercises) -> Assignment:
    return Assignment(id="t", created="", chapter=1, chapter_title="Kapitel 1",
                      level="A1", exercises=exercises)


def main() -> int:
    fails = 0

    def check(label, got, want):
        nonlocal fails
        ok = got == want
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            print(f"        want {want!r}, got {got!r}")

    print("\nmarking a closed exercise — must not be unfair:")
    e = gap()
    check("exact answer", e.check("heiße")[0], True)
    check("case and spacing ignored", e.check("  Heiße  ")[0], True)
    check("umlaut spelling is not the test", e.check("heisse")[0], True)
    check("an alternative counts",
          gap(alts=["heisse"]).check("heisse")[0], True)
    # The two that were marked wrong before, and should not have been.
    check("answering with the whole sentence",
          e.check("Ich heiße Anna")[0], True)
    mc = Exercise(kind=CLOSED, prompt="A) Hallo B) Guten Tag", answer="C")
    check("multiple choice answered as 'C) Guten Tag!'",
          mc.check("C) Guten Tag!")[0], True)
    check("...and as a bare letter", mc.check("c")[0], True)

    print("\nbut it still catches real mistakes:")
    check("the wrong word", e.check("bin")[0], False)
    check("blank is wrong, not unmarkable", e.check("")[0], False)
    check("blank says so", "blank" in e.check("")[1], True)
    check("a near miss is flagged as close",
          "close" in e.check("heiß")[1], True)
    check("the answer is given back", "«heiße»" in e.check("bin")[1], True)
    # Whole-word matching: the answer must not be found inside a longer word.
    ist = Exercise(kind=CLOSED, prompt="___", answer="ist")
    check("an answer inside another word does not count",
          ist.check("Christian")[0], False)

    print("\nwhat cannot be marked mechanically:")
    check("an open exercise returns None",
          Exercise(kind=OPEN, prompt="Write about your weekend").check("x")[0],
          None)
    check("a closed exercise with no answer returns None",
          Exercise(kind=CLOSED, prompt="?", answer="").check("x")[0], None)

    print("\nmark_closed reports what is left for the model:")
    a = build([gap(), Exercise(kind=OPEN, prompt="Write a dialogue."), gap("bin")])
    a.submit(["heiße", "Hallo! Ich heiße Anna.", "wrong"])
    remaining = a.mark_closed()
    check("only the open one is left over", remaining, [1])
    check("closed ones are marked by the rules",
          [x.marked_by for x in sorted(a.answers, key=lambda z: z.index)],
          ["rules", "", "rules"])
    check("score counts only what was judged", a.score, (1, 2))

    print("\nuntrusted model marks:")
    a2 = build([gap(), Exercise(kind=OPEN, prompt="Write.")])
    a2.submit(["heiße", "Ich heiße Anna."])
    open_idx = a2.mark_closed()
    # The real failure: asked to mark exercise 1, the model returned index 0.
    a2.apply_model_marks([{"index": 0, "correct": True, "note": "good"}],
                         expected=open_idx)
    check("a renumbered mark is matched positionally",
          a2.answer_for(1).marked_by, "model")
    check("and does not overwrite the rules' verdict",
          a2.answer_for(0).marked_by, "rules")

    a3 = build([Exercise(kind=OPEN, prompt="Write.")])
    a3.submit(["Etwas."])
    a3.mark_closed()
    a3.apply_model_marks(
        [{"index": 99, "correct": True}, "not a dict", {"index": "x"}],
        expected=[0],
    )
    check("junk marks do not crash it", a3.is_marked, True)
    check("an out-of-range index alone is dropped",
          a3.answer_for(0).correct, None)

    a4 = build([gap()])
    a4.submit(["bin"])
    a4.mark_closed()
    a4.apply_model_marks([{"index": 0, "correct": True}], expected=[])
    check("the model cannot overturn a mechanical mark",
          (a4.answer_for(0).correct, a4.answer_for(0).marked_by), (False, "rules"))

    print("\nparsing what the model generated:")
    raw = {"exercises": [
        {"kind": "closed", "prompt": "Ich ___ Anna.", "answer": "heiße",
         "alternatives": ["heisse"]},
        {"kind": "closed", "prompt": "No answer given."},      # -> open
        {"kind": "open", "prompt": "Write about your day."},
        {"kind": "closed", "answer": "x"},                      # no prompt
        "not a dict",
    ]}
    parsed = homework.parse_generated(raw, chapter=3, chapter_title="Lecker!",
                                      level="A1")
    check("useless entries are dropped", len(parsed.exercises), 3)
    check("a closed exercise with no answer is demoted, not dropped",
          parsed.exercises[1].kind, OPEN)
    check("alternatives survive", parsed.exercises[0].alternatives, ["heisse"])
    check("chapter is recorded", parsed.chapter, 3)
    check("nothing is submitted yet", parsed.submitted, False)

    print("\npersistence:")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        check("nothing there yet", homework.latest(d), None)
        homework.save(parsed, d)
        check("no .tmp left behind", list(d.glob("*.tmp")), [])
        back = homework.latest(d)
        check("round-trips", back.exercises[0].answer, "heiße")
        check("kinds survive", [e.kind for e in back.exercises],
              [CLOSED, OPEN, OPEN])
        check("outstanding lists unanswered work", len(homework.outstanding(d)), 1)
        back.submit(["heiße", "x", "y"])
        homework.save(back, d)
        check("answered work is no longer outstanding",
              homework.outstanding(d), [])
        (d / "broken.json").write_text('{"id": ', encoding="utf-8")
        check("an unreadable file is skipped, not fatal",
              len(homework.load_all(d)), 1)

    print("\nrendering:")
    text = parsed.as_text()
    check("prompts are shown", "Ich ___ Anna." in text, True)
    check("answers are hidden before marking", "heiße" not in text, True)
    check("and shown afterwards", "heiße" in parsed.as_text(show_answers=True),
          True)

    print(f"\n{'PASS' if not fails else 'FAIL'} — homework\n")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
