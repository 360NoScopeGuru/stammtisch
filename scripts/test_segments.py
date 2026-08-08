"""Language segmentation, mode tokens, and guillemet-aware sentence splitting.

Pure stdlib — no models. `python scripts/test_segments.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.segments import (  # noqa: E402
    DE, EN, extract_mode, split_segments, strip_markers,
)
from app.sentences import SentenceStreamer  # noqa: E402

SEGMENT_CASES = [
    (
        'Try saying «Ich heiße Anna», which means "my name is Anna".',
        EN,
        # The stranded comma rides along with the phrase it followed, so the
        # synthesiser pauses in the right place.
        [(EN, "Try saying"), (DE, "Ich heiße Anna,"),
         (EN, 'which means "my name is Anna".')],
    ),
    (
        "«Guten Morgen!» is how you greet someone before noon.",
        EN,
        [(DE, "Guten Morgen!"), (EN, "is how you greet someone before noon.")],
    ),
    ("«Wie geht es dir?»", DE, [(DE, "Wie geht es dir?")]),
    ("That is exactly right.", EN, [(EN, "That is exactly right.")]),
    # Same language back to back: merged into one call, commas preserved, so it
    # is spoken as a list rather than three disconnected words.
    ("«eins», «zwei», «drei»", EN, [(DE, "eins, zwei, drei")]),
    # Real capture from practice mode: the model glossed itself in brackets
    # despite being told never to. Without splitting on the bracket the whole
    # line, English included, would go to the German voice.
    (
        "«Guten Tag! Wie heißt du? (What is your name?)»", DE,
        [(DE, "Guten Tag! Wie heißt du?"), (EN, "What is your name?")],
    ),
    # A German aside in brackets is still German — the bracket only breaks
    # ties, it does not decide. The trailing full stop rides along, as ever.
    (
        "That costs «zwei Euro» (zwei Euro fünfzig).", EN,
        [(EN, "That costs"), (DE, "zwei Euro zwei Euro fünfzig.")],
    ),
]

TOKEN_CASES = [
    ("Los geht's! [[PRACTICE]]", "Los geht's!", "practice"),
    ("Let us practise. [[ PRACTICE ]]", "Let us practise.", "practice"),
    ("Back to basics. **[[MENTOR]]**", "Back to basics.", "mentor"),
    ("[[PRACTICE]]", "", "practice"),
    ("No token at all.", "No token at all.", None),
]

# The splitter must not cut inside a German quote, or half the phrase goes to
# the wrong voice.
SENTENCE_CASES = [
    (["Say «Ich heiße Anna. Und du?» after me."],
     ["Say «Ich heiße Anna. Und du?» after me."]),
    (["«Guten Tag!» means hello. Try it."],
     ["«Guten Tag!» means hello.", "Try it."]),
    (["First this. Then «bitte schön». Done."],
     ["First this.", "Then «bitte schön».", "Done."]),
    # Practice mode: every sentence is wrapped, and they must still stream one
    # at a time rather than all arriving at flush.
    (["«Wie geht es dir?» «Und was machst du heute?»"],
     ["«Wie geht es dir?»", "«Und was machst du heute?»"]),
]


def test_practice_streams_incrementally() -> bool:
    """In practice mode the first sentence must emit before the reply ends,
    otherwise sentence-level streaming buys nothing."""
    s = SentenceStreamer()
    early = s.feed("«Guten Tag!» «Wie geht es")
    return early == ["«Guten Tag!»"]


def main() -> int:
    fails = 0

    def check(label, got, want):
        nonlocal fails
        ok = got == want
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            print(f"        want {want}")
            print(f"        got  {got}")

    print("\nsplit_segments:")
    for text, default, want in SEGMENT_CASES:
        check(text[:52], split_segments(text, default), want)

    print("\nextract_mode:")
    for text, want_clean, want_mode in TOKEN_CASES:
        check(text[:52], extract_mode(text), (want_clean, want_mode))

    print("\nstrip_markers:")
    check("removes guillemets",
          strip_markers("Say «Guten Tag» now."), "Say Guten Tag now.")

    print("\nsentence splitting inside guillemets:")
    for chunks, want in SENTENCE_CASES:
        s = SentenceStreamer()
        got = []
        for c in chunks:
            got.extend(s.feed(c))
        got.extend(s.flush())
        check(chunks[0][:52], got, want)

    print("\nstreaming behaviour:")
    check("practice mode emits mid-stream, not only at flush",
          test_practice_streams_incrementally(), True)

    print(f"\n{'PASS' if not fails else 'FAIL'} — segments\n")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
