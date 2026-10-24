"""Spaced repetition: the schedule, the queue, and fair marking.

`python scripts/test_srs.py`
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import classlog, handouts, srs  # noqa: E402
from app.progress import Mistake, Progress, Word  # noqa: E402

failures = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global failures
    failures += not cond
    print(f"  {'PASS' if cond else 'FAIL'}  {label}"
          + (f"\n        {detail}" if detail and not cond else ""))


D0 = date(2026, 10, 1)

print("\nSM-2:")
s = srs.State()
intervals = []
for q in (5, 5, 5, 5):
    srs.apply(s, q, D0)
    intervals.append(s.interval)
check("intervals grow 1, 6, then multiply", intervals[:2] == [1, 6]
      and intervals[2] > 6 and intervals[3] > intervals[2], str(intervals))
before = s.ef
srs.apply(s, 1, D0)
check("a lapse collapses the interval to a day", s.interval == 1 and s.reps == 0)
check("and is counted", s.lapses == 1)
check("and lowers the ease", s.ef < before)
for _ in range(20):
    srs.apply(s, 0, D0)
check("ease never drops below 1.3", s.ef == 1.3)
hard = srs.apply(srs.State(), 3, D0)
easy = srs.apply(srs.State(), 5, D0)
check("a hard pass eases less than an easy one", hard.ef < easy.ef)

print("\nmarking vocabulary:")
card = srs.Card(id="x", kind=srs.VOCAB, front="bread roll", back="das Brötchen",
                source="class")


def mark(given):
    r = srs.grade(card, given)
    return r.correct, r.q, r.note


check("exact", mark("das Brötchen")[:2] == (True, 5))
check("case and punctuation forgiven", mark("Das brötchen.")[:2] == (True, 5))
check("oe for the umlaut is not a mistake", mark("das Broetchen")[:2] == (True, 4))
check("nor is a plain o", mark("das Brotchen")[:2] == (True, 4))
check("the umlaut spelling is shown", "«das Brötchen»" in mark("das Brotchen")[2])
ok, q, note = mark("der Brötchen")
check("a wrong article is WRONG", ok is False and q < 3 and "article" in note, note)
ok, q, note = mark("Brötchen")
check("no article: right word, but told to learn the article",
      ok is True and q == 3 and "article" in note, note)
check("a one-letter slip on a long word is close", mark("das Brötchn")[0] is True)
check("a different word is wrong", mark("das Brot")[0] is False)
check("blank is wrong and shows the answer", mark("")[:2] == (False, 0)
      and "Brötchen" in mark("")[2])
short = srs.Card(id="y", kind=srs.VOCAB, front="yes", back="ja", source="s")
check("short words get no slack", srs.grade(short, "je").correct is False)

print("\nmarking a mistake card:")
mc = srs.Card(id="m", kind=srs.MISTAKE, front="Ich sehe der Buch.",
              back="Ich sehe das Buch.", source="mistakes",
              wrong="der", right="das", note="Buch is neuter.")
check("the corrected sentence", srs.grade(mc, "Ich sehe das Buch").correct)
check("fixed in other words", srs.grade(mc, "ich sehe das buch heute").correct)
r = srs.grade(mc, "Ich sehe der Buch")
check("the mistake left in is wrong, with the reason",
      not r.correct and "neuter" in r.note, r.note)
check("both forms at once is not a fix",
      not srs.grade(mc, "der das Buch").correct)

print("\ncollecting cards:")
study = Path(tempfile.mkdtemp())
progress = Progress(
    sessions=2,
    vocab={"danke": Word("danke", meaning="thank you"),
           "bitte": Word("bitte"),                     # no meaning: no card
           "brötchen": Word("das Brötchen", meaning="roll")},
    mistakes=[Mistake("der", "das", 3, example="Ich sehe der Buch.",
                      corrected="Ich sehe das Buch.", explanation="neuter"),
              Mistake("ist", "bin", 1, example="Ich ist.", corrected="Ich bin.")],
)
classlog.save(classlog.Entry(id="c1", day="2026-10-01", created="x", vocab=[
    classlog.Word(de="das Brötchen", en="bread roll", checked=True),
    classlog.Word(de="Brot", en="bread", checked=True, article="unsure")]), study)
deck = handouts.new_deck("Nomen", "h.pdf", [
    handouts.Item(de="der Tisch", en="table", status=handouts.CONFIRMED),
    handouts.Item(de="das Lampe", en="lamp", status=handouts.PENDING)])
handouts.save_deck(deck, study)

cards = srs.collect(study, progress)
backs = {c.back: c for c in cards}
check("session words with a meaning", "danke" in backs)
check("a word with no meaning makes no card", "bitte" not in backs)
check("confirmed handout words", "der Tisch" in backs)
check("a pending handout word is NOT learned",
      "das Lampe" not in backs and "die Lampe" not in backs)
check("class vocabulary, once, with the class's meaning",
      sum(1 for c in cards if c.back == "das Brötchen") == 1
      and backs["das Brötchen"].front == "bread roll")
check("a noun whose article is unsure is not drilled", "Brot" not in backs)
check("a recurring mistake makes a card", "Ich sehe das Buch." in backs)
check("a one-off mistake does not", "Ich bin." not in backs)

print("\nthe queue:")
today = date(2026, 10, 20)
states = srs.schedule(study)
q = srs.queue(cards, states, today)
check("everything is new at first", len(q) == len(cards))
check("mistakes come first", q[0].kind == srs.MISTAKE)

# Review the Tisch card today, the danke card well in the past.
log = study / srs.LOG_NAME
with log.open("w", encoding="utf-8") as fh:
    fh.write(json.dumps({"at": "2026-10-20T10:00:00", "card": backs["der Tisch"].id,
                         "q": 5}) + "\n")
    fh.write(json.dumps({"at": "2026-10-01T10:00:00", "card": backs["danke"].id,
                         "q": 5}) + "\n")
    fh.write("{ broken line from a kill\n")
states = srs.schedule(study)
check("a damaged log line is skipped, not fatal", len(states) == 2)
q = srs.queue(cards, states, today)
ids = [c.id for c in q]
check("a card just reviewed is not due", backs["der Tisch"].id not in ids)
check("an overdue card is due, ahead of new ones",
      ids[0] == backs["danke"].id, str([c.back for c in q]))
check("tomorrow it is due", backs["der Tisch"].id in
      [c.id for c in srs.queue(cards, states, today + timedelta(days=1))])

many = [srs.Card(id=str(i), kind=srs.VOCAB, front=str(i), back=str(i),
                 source="s") for i in range(30)]
check("new cards are rationed per day",
      len(srs.queue(many, {}, today)) == srs.NEW_PER_DAY)
st = {str(i): srs.State(reviews=1, first=today, due=today + timedelta(days=1))
      for i in range(4)}
check("cards started today use up the ration",
      len(srs.queue(many, st, today)) == srs.NEW_PER_DAY - 4)

brief = srs.brief(study, progress, today)
check("the tutor hears which words are due", "«danke»" in brief, brief)
check("and nothing when nothing is due",
      srs.brief(Path(tempfile.mkdtemp()), Progress(), today) == "")

srs.record(study, backs["der Tisch"], 2, "die Tisch")
check("a review is appended to the log",
      srs.read_log(study)[-1]["given"] == "die Tisch")

print(f"\n{'PASS' if not failures else 'FAIL'} — spaced repetition\n")
sys.exit(1 if failures else 0)
