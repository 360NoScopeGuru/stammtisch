"""Textbook parsing, and the alignment bug that nearly shipped.

The first version of `parse_syllabus` produced exactly the right number of
blocks with every grammar list shifted one chapter along — so chapter 3, about
food, was labelled with chapter 2's articles and plurals. Counting the blocks
did not catch it. Nothing would have caught it except reading the output, and
by then it would have been teaching the wrong grammar in the wrong order.

So the fixture below is the real page layout in miniature, including the
wrapped bullet that made the naive parser hard to write, and the checks are on
alignment rather than on counts.

Pure stdlib. `python scripts/test_curriculum.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import curriculum  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest_textbook import (  # noqa: E402
    parse_listing, parse_syllabus, validate,
)

# Page one of the contents spread. Topic keywords are indented; chapter titles
# shout; section titles do not. Chapter 1 has no page number printed.
LISTING = """INHALT2
THEMENKAPITEL
1 LOS GEHT'S
1 Freut mich!
2 Haben Sie bitte ein wenig Geduld!
3 Fragen und Antworten
  Kennenlernen      Länder
  Zahlen      Alphabet
26 2 DEUTSCHE SPRACHE, SCHWERE SPRACHE?
1 Die Nomengruppe
2 Im ganzen Satz, bitte!
  Kursraum      Lernen
43 3 LECKER!
1 Was darf es sein?
2 Das esse ich
  Essen      Einkaufen
SEITE
INHALT
4
"""

# Page two. Note the wrapped bullet in the third block: its continuation line
# is indented, and that indentation is the only thing distinguishing it from
# the start of a new chapter's can-do statements.
SYLLABUS = """INHALT 3
SPRACHHANDLUNGEN GRAMMATIK
sich begrüßen und verabschieden | sich und andere vorstellen |
zählen | buchstabieren | Fragen stellen
• Personalpronomen
• Verbkonjugation Präsens
• Verbpositionen
nach Gegenständen fragen |
zustimmen und widersprechen
• Artikel
• Plural
• Negation
höfliche Bitten formulieren | über Vorlieben sprechen
• Verben mit Akkusativobjekt
• Stellung von Personalpronomen im
 Dativ und Akkusativ
Impressum und Quellenverzeichnis ab Seite 203
"""


def main() -> int:
    fails = 0

    def check(label, got, want):
        nonlocal fails
        ok = got == want
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            print(f"        want {want!r}")
            print(f"        got  {got!r}")

    print("\nparse_listing:")
    chapters = parse_listing(LISTING)
    check("finds every chapter", [c.number for c in chapters], [1, 2, 3])
    check("chapter title", chapters[0].title, "LOS GEHT'S")
    check("title with punctuation survives",
          chapters[1].title, "DEUTSCHE SPRACHE, SCHWERE SPRACHE?")
    check("section titles", [u.title for u in chapters[0].units],
          ["Freut mich!", "Haben Sie bitte ein wenig Geduld!",
           "Fragen und Antworten"])
    check("indented topic keywords", chapters[0].topics,
          ["Kennenlernen", "Länder", "Zahlen", "Alphabet"])
    check("book page when printed", chapters[1].book_page, 26)
    check("no page for chapter one", chapters[0].book_page, None)
    check("trailing junk is not a chapter",
          [c.title for c in chapters].count("INHALT"), 0)

    print("\nparse_syllabus:")
    blocks = parse_syllabus(SYLLABUS)
    check("one block per chapter", len(blocks), 3)

    can_do_1, grammar_1 = blocks[0]
    can_do_2, grammar_2 = blocks[1]
    can_do_3, grammar_3 = blocks[2]

    check("can-do statements split on the pipe", can_do_1,
          ["sich begrüßen und verabschieden", "sich und andere vorstellen",
           "zählen", "buchstabieren", "Fragen stellen"])
    check("chapter 1 grammar", grammar_1,
          ["Personalpronomen", "Verbkonjugation Präsens", "Verbpositionen"])

    # The alignment bug: these two used to hold chapter 1's and chapter 2's
    # grammar respectively, while the can-do lists ran a chapter ahead.
    check("chapter 2 grammar is not chapter 1's", grammar_2,
          ["Artikel", "Plural", "Negation"])
    check("chapter 2 can-do does not absorb chapter 1's", can_do_2,
          ["nach Gegenständen fragen", "zustimmen und widersprechen"])
    check("chapter 3 can-do stays its own", can_do_3,
          ["höfliche Bitten formulieren", "über Vorlieben sprechen"])
    check("a wrapped bullet is one grammar point, not two", grammar_3,
          ["Verben mit Akkusativobjekt",
           "Stellung von Personalpronomen im Dativ und Akkusativ"])

    print("\nvalidate:")
    check("a clean parse has no problems",
          validate(chapters, blocks, SYLLABUS, min_chapters=1), [])
    check("a chapter/block mismatch is caught",
          bool(validate(chapters, blocks[:2], SYLLABUS, min_chapters=1)), True)
    # Simulate the shift: statements stitched across two chapters are not
    # contiguous text on the page, which is what makes the check work.
    shifted = [(["Fragen stellen nach Gegenständen fragen"], grammar_1),
               (can_do_2, grammar_2), (can_do_3, grammar_3)]
    check("misalignment is caught even with the right block count",
          bool(validate(chapters, shifted, SYLLABUS, min_chapters=1)), True)

    print("\nChapter.describe:")
    chapters[0].can_do, chapters[0].grammar = can_do_1, grammar_1
    text = chapters[0].describe()
    check("names the chapter", "LOS GEHT'S" in text, True)
    check("states the objectives", "buchstabieren" in text, True)
    check("states the grammar", "Personalpronomen" in text, True)

    print("\nCourse:")
    course = curriculum.Course(title="t", level="A1", chapters=chapters)
    check("get by number", course.get(2).title,
          "DEUTSCHE SPRACHE, SCHWERE SPRACHE?")
    check("first chapter", course.first().number, 1)
    check("next after", course.next_after(1).number, 2)
    check("next after the last is None", course.next_after(3), None)

    print(f"\n{'PASS' if not fails else 'FAIL'} — curriculum\n")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
