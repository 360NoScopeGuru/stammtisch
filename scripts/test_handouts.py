"""Reading der/das/die columns from a handout, and refusing to guess.

The failure this guards against is the one that blocked the feature: a row
with an empty middle cell, read by text extraction, shifts its right-hand noun
one column left and silently teaches the wrong article. The positional reader
must put it under the heading it actually sits under, and the merge must only
confirm an article that independent readings agree on.

`python scripts/test_handouts.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import handouts  # noqa: E402
from pdfmaker import make_pdf  # noqa: E402

failures = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global failures
    failures += not cond
    print(f"  {'PASS' if cond else 'FAIL'}  {label}"
          + (f"\n        {detail}" if detail and not cond else ""))


def column_page() -> bytes:
    # Three columns, as the class handouts lay them out. Row 2 has no «das»
    # entry: the row that used to come out as «das Lampe».
    D, N, F = 80, 260, 440
    rows = [
        [(D, "Tisch, -e"), (N, "Buch, -¨er"), (F, "Tür, -en")],
        [(D, "Stuhl, -¨e"), (F, "Lampe, -n")],
        [(N, "Fenster, -"), (F, "Tasche, -n")],
        [(D, "Kugelschreiber, -")],
    ]
    items = [(30, 800, "Nomen im Klassenzimmer"),
             (D, 760, "der"), (N, 760, "das"), (F, 760, "die")]
    y = 735
    for row in rows:
        items += [(x, y, text) for x, text in row]
        y -= 22
    return make_pdf([items])


print("\nposition:")
words = handouts.words_from_pdf_page(column_page(), 1)
read = dict(handouts.read_by_position(words))
by_noun = {handouts.clean_noun(k): v for k, v in read.items()}
check("every noun is found", set(by_noun) == {
    "Tisch", "Buch", "Tür", "Stuhl", "Lampe", "Fenster", "Tasche",
    "Kugelschreiber"}, f"got {sorted(by_noun)}")
check("a noun after an empty middle cell keeps its own column",
      by_noun.get("Lampe") == "die", f"Lampe -> {by_noun.get('Lampe')}")
check("a noun with an empty first cell keeps its own column",
      by_noun.get("Fenster") == "das", f"Fenster -> {by_noun.get('Fenster')}")
check("every article right", by_noun == {
    "Tisch": "der", "Buch": "das", "Tür": "die", "Stuhl": "der",
    "Lampe": "die", "Fenster": "das", "Tasche": "die",
    "Kugelschreiber": "der"}, str(by_noun))
check("the title above the headings is not a noun",
      not any("Nomen" in k for k in read))

prose = make_pdf([[(40, 760, "Die Katze ist das Tier der Nachbarin."),
                   (40, 740, "Sie heißt Mia.")]])
check("prose with all three articles is not mistaken for a table",
      handouts.read_by_position(handouts.words_from_pdf_page(prose, 1)) == [])

gendered = make_pdf([[(80, 760, "Maskulin"), (260, 760, "Neutrum"),
                      (440, 760, "Feminin"), (80, 735, "Hund"),
                      (440, 735, "Katze")]])
check("gender names work as headings",
      dict(handouts.read_by_position(handouts.words_from_pdf_page(gendered, 1)))
      == {"Hund": "der", "Katze": "die"})

print("\nnoun keys:")
check("article, plural and case do not matter",
      handouts.noun_key("der Tisch, -e") == handouts.noun_key("Tisch (-e)")
      == handouts.noun_key("tisch"))
check("umlauts fold", handouts.noun_key("Tür") == handouts.noun_key("Tuer")
      or handouts.noun_key("Tür") == "tur")

print("\nmerging:")
items = handouts.combine(
    position=[("Tisch, -e", "der"), ("Lampe, -n", "die"), ("Buch", "das")],
    vision=[("Tisch", "der"), ("Lampe", "das"), ("Heft", "das")],
    lexicon={"tisch": ("der", "table"), "lampe": ("die", "lamp"),
             "buch": ("das", "book"), "heft": ("das", "exercise book")},
)
got = {i.noun: i for i in items}
check("three readings agreeing confirm",
      got["Tisch"].status == "confirmed" and got["Tisch"].de == "der Tisch")
check("any disagreement leaves it pending",
      got["Lampe"].status == "pending", str(got["Lampe"].readings))
check("a pending entry shows what the page says, not the majority",
      got["Lampe"].de == "die Lampe")
check("two agreeing readings are enough", got["Buch"].status == "confirmed")
check("a photo (vision + lexicon) can confirm", got["Heft"].status == "confirmed")
check("the meaning is carried", got["Tisch"].en == "table")

lonely = handouts.combine(position=[("Tafel", "die")], vision=[], lexicon={})
check("a single reading is never enough", lonely[0].status == "pending")

garbage = handouts.parse_vision({"nouns": [
    {"noun": "Tisch", "article": "der"}, {"noun": "Stuhl", "article": "den"},
    {"noun": "", "article": "die"}, "nonsense", {"article": "das"}]})
check("malformed model output is dropped, not guessed at",
      garbage == [("Tisch", "der")], str(garbage))

print("\ndecks:")
import tempfile  # noqa: E402
study = Path(tempfile.mkdtemp())
deck = handouts.new_deck("Nomen 1", "blatt.pdf", items + [
    handouts.Item(de="der Tisch", status="confirmed")])
check("a noun on two pages is kept once",
      sum(1 for e in deck.entries if e.noun == "Tisch") == 1)
handouts.save_deck(deck, study)
loaded = handouts.find_deck(study, deck.slug)
check("a deck round-trips", loaded and len(loaded.entries) == len(deck.entries))
lampe = [e.noun for e in loaded.entries].index("Lampe")
item = handouts.decide(study, deck.slug, lampe, handouts.CONFIRMED, de="die Lampe")
check("the learner's decision confirms it",
      item and handouts.find_deck(study, deck.slug).entries[lampe].status
      == "confirmed")
check("and is recorded as theirs", item.readings.get("learner") == "die")
check("a bad status is refused",
      handouts.decide(study, deck.slug, 0, "maybe") is None)

print(f"\n{'PASS' if not failures else 'FAIL'} — handouts\n")
sys.exit(1 if failures else 0)
