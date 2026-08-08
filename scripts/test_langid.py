"""Language classification for voice routing.

Every case marked REGRESSION is a real string produced by a local model that
was routed to the wrong voice and came out as phonetic mush.
`python scripts/test_langid.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.langid import DE, EN, detect  # noqa: E402
from app.segments import clean_for_speech, split_segments  # noqa: E402

CASES = [
    # (text, tie_break_default, expected, note)
    # --- REGRESSIONS: German that escaped the guillemets (gemma3:4b) ---
    ("Wie geht es Ihnen heute?", EN, DE, "REGRESSION German outside marks"),
    ("Guten Tag!", EN, DE, "REGRESSION short German greeting"),
    ("Ich heiße Anna.", EN, DE, "REGRESSION"),
    # --- REGRESSIONS: English that got inside the guillemets (gemma3:12b) ---
    ("Hello! How are you?", DE, EN, "REGRESSION English inside marks"),
    ("I am happy to talk with you.", DE, EN, "REGRESSION"),
    ("What's your name?", DE, EN, "REGRESSION"),

    # --- ordinary mentor-mode prose ---
    ("Try saying that back to me.", EN, EN, ""),
    ("That was good, well done.", EN, EN, ""),
    ("which means my name is Anna", EN, EN, ""),

    # --- ordinary practice-mode German ---
    ("Was möchtest du sagen?", DE, DE, ""),
    ("Ich hätte gerne zwei Brötchen.", EN, DE, "umlauts decide"),
    ("Wie war dein Wochenende?", EN, DE, ""),

    # --- too short to call: the tie-break must win ---
    ("Ja.", DE, DE, "no evidence -> practice default"),
    ("Anna", EN, EN, "proper noun, no evidence"),
    ("...", DE, DE, "no words at all"),
]


def main() -> int:
    fails = 0

    print("\nclassification:")
    for text, default, want, note in CASES:
        got = detect(text, default)
        ok = got == want
        fails += not ok
        tag = f"  [{note}]" if note else ""
        print(f"  {'PASS' if ok else 'FAIL'}  {got}  {text[:44]!r}{tag}")
        if not ok:
            print(f"        expected {want} (tie-break {default})")

    print("\nstage directions stripped (never spoken):")
    strip_cases = [
        ("(German voice) «Hallo!»", "«Hallo!»"),
        ("(speaking slowly) Guten Tag", "Guten Tag"),
        ("(in English) That is right", "That is right"),
        # Must NOT strip meaningful parentheses.
        ("«Hallo» (hello) is a greeting", "«Hallo» (hello) is a greeting"),
    ]
    for raw, want in strip_cases:
        got = clean_for_speech(raw)
        ok = got == want
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {raw[:44]!r}")
        if not ok:
            print(f"        want {want!r}")
            print(f"        got  {got!r}")

    print("\nend-to-end routing of the strings that broke:")
    routing = [
        # (reply, tie_break, expected [(lang, ...)] languages in order)
        ("«Guten Tag!» Wie geht es Ihnen heute?", DE, [DE]),
        ("Try saying «Ich heiße Anna», which means my name is Anna.", EN,
         [EN, DE, EN]),
        ("«Hallo! Wie geht es dir?» I am happy to talk with you.", DE, [DE, EN]),
        ("(German voice) «Hallo!»", EN, [DE]),
    ]
    for reply, tie, want_langs in routing:
        got = split_segments(reply, tie)
        got_langs = [lang for lang, _ in got]
        ok = got_langs == want_langs
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {got_langs}  {reply[:46]!r}")
        if not ok:
            print(f"        expected {want_langs}")
            print(f"        got      {got}")

    print(f"\n{'PASS' if not fails else 'FAIL'} — langid\n")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
