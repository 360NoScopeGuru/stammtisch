"""Sanity checks for the streaming sentence splitter. Pure stdlib — runs with
no models or GPU. `python scripts/test_sentences.py`"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.sentences import SentenceStreamer  # noqa: E402

CASES: list[tuple[list[str], list[str]]] = [
    # (token chunks fed in, expected sentences out including flush)
    (["Hallo! Wie geht", " es dir?"], ["Hallo!", "Wie geht es dir?"]),
    (["Das kostet ca. 3 Euro."], ["Das kostet ca. 3 Euro."]),
    (["Ich komme am 3. Oktober an."], ["Ich komme am 3. Oktober an."]),
    (["Wir treffen uns, d.h. morgen früh."], ["Wir treffen uns, d.h. morgen früh."]),
    (["Guten Tag", ", Herr Dr. Meier. Bitte", " setzen Sie sich."],
     ["Guten Tag, Herr Dr. Meier.", "Bitte setzen Sie sich."]),
    (['Er sagte: "Komm her!" Dann ging er.'],
     ['Er sagte: "Komm her!"', "Dann ging er."]),
    (["Wirklich?! Das", " glaube ich nicht..."],
     ["Wirklich?!", "Das glaube ich nicht..."]),
    (["Kein Satzende"], ["Kein Satzende"]),
    # streamed one character at a time
    (list("Ja. Nein."), ["Ja.", "Nein."]),
]


def main() -> int:
    failures = 0
    for chunks, expected in CASES:
        s = SentenceStreamer()
        got: list[str] = []
        for c in chunks:
            got.extend(s.feed(c))
        got.extend(s.flush())

        ok = got == expected
        failures += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {''.join(chunks)!r}")
        if not ok:
            print(f"        expected {expected}")
            print(f"        got      {got}")

    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
