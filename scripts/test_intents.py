"""Mode-switch intent detection.

The risk here is asymmetric: missing a switch is a small annoyance, but a false
positive yanks the learner out of a lesson mid-sentence. The negative cases
matter more than the positive ones. `python scripts/test_intents.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.intents import requested_mode  # noqa: E402
from app.prompts import MENTOR, PRACTICE  # noqa: E402

CASES = [
    # (utterance, current_mode, expected)
    ("Can we practise now please?", MENTOR, PRACTICE),
    ("Okay I understand, let's practice", MENTOR, PRACTICE),
    ("I want to practise this", MENTOR, PRACTICE),
    ("Let's try it for real", MENTOR, PRACTICE),
    ("Shall we just talk?", MENTOR, PRACTICE),
    ("lass uns üben", MENTOR, PRACTICE),
    ("Could we do some roleplay", MENTOR, PRACTICE),

    ("I don't understand", PRACTICE, MENTOR),
    ("What does Brötchen mean?", PRACTICE, MENTOR),
    ("Can you explain that again", PRACTICE, MENTOR),
    ("Say it in English please", PRACTICE, MENTOR),
    ("Stop the roleplay", PRACTICE, MENTOR),
    ("ich verstehe das nicht", PRACTICE, MENTOR),

    # --- must NOT switch ---
    ("Hallo, wie geht es dir?", PRACTICE, None),
    ("Ich hätte gerne zwei Brötchen", PRACTICE, None),
    ("That makes sense, thank you", MENTOR, None),
    ("I practise German every day", MENTOR, None),      # not a request
    ("My teacher explained it well", MENTOR, None),     # past tense, not a request
    ("Yes", MENTOR, None),
    ("", MENTOR, None),
    # Already in the target mode — never re-trigger.
    ("Can we practise now?", PRACTICE, None),
    ("I don't understand", MENTOR, None),
]


def main() -> int:
    fails = 0
    for text, mode, want in CASES:
        got = requested_mode(text, mode)
        ok = got == want
        fails += not ok
        arrow = f"{mode} -> {got or '(stay)'}"
        print(f"  {'PASS' if ok else 'FAIL'}  {arrow:<22} {text[:44]!r}")
        if not ok:
            print(f"        expected {want or '(stay)'}")

    print(f"\n{len(CASES) - fails}/{len(CASES)} passed\n")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
