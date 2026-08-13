"""Repeat-after-me scoring.

The thing to get right is the false negative. Telling a learner their German
was wrong when the recogniser merely spelled it differently is worse than
saying nothing — it teaches them to distrust the app, and they cannot tell the
two cases apart from the inside. So every orthographic variant Whisper
actually produces has to score as correct.

Pure stdlib. `python scripts/test_drills.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.drills import CLOSE, GOOD, compare, normalise  # noqa: E402

TARGET = "Ich hätte gern ein Brötchen"

# (heard, expected verdict, label)
CASES = [
    (TARGET, "good", "said exactly right"),
    # Whisper's spelling is not the learner's mistake.
    ("Ich haette gern ein Broetchen", "good", "ae/oe transliteration"),
    ("ich hatte gern ein Brotchen", "good", "umlauts dropped by the recogniser"),
    ("Ich hätte gern ein Brötchen.", "good", "punctuation and case"),
    # Genuine problems.
    ("Ich hätte gern ein Brot", "close", "one word wrong"),
    ("Ich gern Brötchen", "close", "two words dropped"),
    ("Ich möchte einen Kaffee", "again", "different sentence entirely"),
    ("", "again", "nothing heard at all"),
    # Extra words cost something, but not the phrase.
    ("Ich hätte gern ein Brötchen bitte", "good", "one extra word"),
]


def main() -> int:
    fails = 0

    def check(label, got, want):
        nonlocal fails
        ok = got == want
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            print(f"        want {want!r}, got {got!r}")

    print("\nnormalise:")
    check("ß folds to ss", normalise("heiße"), normalise("heisse"))
    check("umlaut folds to the bare vowel", normalise("Brötchen"),
          normalise("Brotchen"))
    check("case is ignored", normalise("ICH"), "ich")
    check("different words stay different",
          normalise("Brot") == normalise("Brötchen"), False)

    print(f"\ncompare against {TARGET!r}:")
    for heard, want, label in CASES:
        attempt = compare(TARGET, heard)
        got = attempt.verdict
        ok = got == want
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        print(f"          score {attempt.score:.2f} → {got:<6} "
              f"{attempt.feedback()}")
        if not ok:
            print(f"        wanted {want!r}")

    print("\nword-level detail:")
    a = compare(TARGET, "Ich hätte gern ein Brot")
    check("names the word that failed", a.missed, ["Brötchen"])
    check("feedback names it too", "«Brötchen»" in a.feedback(), True)

    b = compare(TARGET, "Ich gern Brötchen")
    check("finds both dropped words", sorted(b.missed), ["ein", "hätte"])

    c = compare(TARGET, TARGET)
    check("a perfect attempt has nothing missed", c.missed, [])
    check("and scores 1.0", c.score, 1.0)
    check("every word marked correct",
          all(r.status == "correct" for r in c.results), True)

    d = compare(TARGET, "")
    check("silence is not silently a pass", d.score, 0.0)
    check("silence gets its own message",
          "did not catch that" in d.feedback(), True)

    print("\nthresholds:")
    check("GOOD is above CLOSE", GOOD > CLOSE, True)
    check("an empty target cannot divide by zero", compare("", "hallo").score, 0.0)

    print(f"\n{'PASS' if not fails else 'FAIL'} — drills\n")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
