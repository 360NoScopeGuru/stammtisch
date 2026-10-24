"""Class notes: what the learner typed beats what the model read, and words
that appear nowhere in the source are not taught.

`python scripts/test_classlog.py`
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import classlog, objectives, prompts  # noqa: E402

failures = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global failures
    failures += not cond
    print(f"  {'PASS' if cond else 'FAIL'}  {label}"
          + (f"\n        {detail}" if detail and not cond else ""))


print("\nreading the note:")
check("Kapitel", classlog.stated_chapter("Kapitel 3 heute") == 3)
check("chapter / ch.", classlog.stated_chapter("did ch. 4 today") == 4
      and classlog.stated_chapter("Chapter 12") == 12)
check("no chapter", classlog.stated_chapter("Akkusativ, Seite 44") is None)
check("pages", classlog.stated_pages("pages 44-47") == "44-47")
check("Seite / S.", classlog.stated_pages("Seite 44") == "44"
      and classlog.stated_pages("S. 44–45") == "44-45")
check("several ranges", classlog.stated_pages("pp. 44-45 and 48") == "44-45,48",
      classlog.stated_pages("pp. 44-45 and 48"))
check("a number that is not a page",
      classlog.stated_pages("we learnt 3 verbs") == "")

print("\nchecking the model's reading:")
raw = {"chapter": 4, "pages": "12", "summary": "Food.",
       "topics": ["Essen", "Essen", ""], "grammar": ["Akkusativ"],
       "vocab": [{"de": "das Brötchen", "en": "bread roll"},
                 {"de": "der Zauberstab", "en": "magic wand"},
                 "Kaffee", {"de": "", "en": "x"}, 42],
       "phrases": ["Ich hätte gern ..."]}
source = ("Im Café. Ich hätte gern ein Brötchen und einen Kaffee, bitte. "
          "Was darf es sein? Ein Brötchen mit Käse. Das macht 3,50 Euro. "
          "Möchten Sie noch etwas? Nein, danke. Auf Wiedersehen!")
e = classlog.parse_extracted(raw, note="Kapitel 3, S. 44", source_text=source,
                             sources=["[attached blatt.pdf, page 1 of 1]"],
                             day=date(2026, 10, 21), chapters={1, 2, 3, 4})
check("the chapter the learner typed wins", e.chapter == 3)
check("the pages the learner typed win", e.pages == "44")
words = [w.de for w in e.vocab]
check("words in the source are kept, and marked checked",
      "das Brötchen" in words and "Kaffee" in words
      and all(w.checked for w in e.vocab), str(words))
check("a word in no source text is dropped", "der Zauberstab" not in words)
check("and the drop is recorded, not silent", e.dropped == ["der Zauberstab"])
check("junk vocab entries are ignored", len(e.vocab) == 2)
check("duplicate and empty topics go", e.topics == ["Essen"])

e2 = classlog.parse_extracted({"chapter": 9}, note="", source_text="",
                              sources=[], day=date(2026, 10, 21), chapters={1, 2, 3})
check("a chapter not in the book is dropped", e2.chapter is None)
e3 = classlog.parse_extracted({"chapter": "2", "pages": "see board"}, note="",
                              source_text="", sources=[], day=date(2026, 10, 21),
                              chapters={1, 2, 3})
check("a chapter given as text still counts", e3.chapter == 2)
check("pages that are not page numbers are dropped", e3.pages == "")

photo = classlog.parse_extracted(
    {"vocab": [{"de": "die Tafel", "en": "board"}]}, note="photo of the board",
    source_text="", sources=["[attached image: IMG_0014.JPG]"],
    day=date(2026, 10, 21), chapters=set())
check("words from a photo cannot be checked, and say so",
      photo.vocab and photo.vocab[0].checked is None)

print("\narticles get a second opinion:")
# The real failure: the model gave «der Brot» for a page that never prints
# «das Brot» next to the noun. «Brötchen» is printed with its article.
src = ("Was darf es sein? Ich hätte gern das Brötchen mit Käse und ein Brot. "
       "Wie viel kostet das? Zwei Euro fünfzig, bitte. Danke schön!") * 2
art = classlog.parse_extracted(
    {"vocab": [{"de": "das Brötchen", "en": "roll"}, {"de": "der Brot", "en": "bread"},
               {"de": "die Käse", "en": "cheese"}, {"de": "bitte", "en": "please"}]},
    note="", source_text=src, sources=[], day=date(2026, 10, 22), chapters=set())
todo = classlog.articles_to_check(art, src)
check("an article printed with the noun needs no second opinion",
      art.vocab[0].article == "page" and "Brötchen" not in todo)
check("unprinted articles are sent to the lexicon", sorted(todo) == ["Brot", "Käse"],
      str(todo))
classlog.apply_lexicon(art, {"brot": ("das", "bread"), "kase": ("der", "cheese")})
by = {w.en: w for w in art.vocab}
check("a disagreement removes the article rather than guess",
      by["bread"].de == "Brot" and by["bread"].article == "unsure")
check("so does a second disagreement", by["cheese"].de == "Käse")
check("a word that is not a noun is left alone",
      by["please"].de == "bitte" and by["please"].article == "")
agree = classlog.parse_extracted(
    {"vocab": [{"de": "das Brot", "en": "bread"}]}, note="", source_text=src,
    sources=[], day=date(2026, 10, 22), chapters=set())
classlog.articles_to_check(agree, src)
classlog.apply_lexicon(agree, {"brot": ("das", "bread")})
check("agreement keeps it", agree.vocab[0].de == "das Brot"
      and agree.vocab[0].article == "agreed")

print("\nstoring and briefing:")
study = Path(tempfile.mkdtemp())
classlog.save(e, study)
old = classlog.parse_extracted({"summary": "Greetings."}, note="Kapitel 1",
                               source_text="", sources=[],
                               day=date(2026, 9, 1), chapters={1, 2, 3})
old.id = "older"
classlog.save(old, study)
entries = classlog.load_all(study)
check("entries come back oldest class first",
      [x.day for x in entries] == ["2026-09-01", "2026-10-21"])
brief = classlog.brief(study, date(2026, 10, 23))
check("the brief carries the recent class", "Kapitel" not in brief
      and "chapter 3" in brief and "«das Brötchen» (bread roll)" in brief, brief)
check("a class weeks ago no longer steers", "Greetings" not in brief)
check("the latest chapter is the newest class's",
      classlog.latest_chapter(study) == 3)
check("deleting works", classlog.delete(study, "older")
      and len(classlog.load_all(study)) == 1)
check("a path in an id is refused", not classlog.delete(study, "../x"))
check("the brief reaches the tutor's prompt",
      "practise this first" in prompts.system_prompt(
          "mentor", "A1", "x", class_notes=brief))

print("\ngoal prechecks:")
check("an English answer fails without asking the model",
      objectives.precheck("I would say that I want a coffee please")[0] is False)
check("a short German answer with no function words is NOT failed",
      objectives.precheck("Einen Tee.") is None)
check("an empty answer fails", objectives.precheck("  ")[0] is False)
check("a German answer goes on to be judged",
      objectives.precheck("Ich hätte gern einen Kaffee, bitte.") is None)
check("a verdict needs 'passed'", objectives.parse_verdict({"note": "x"}) is None)
check("'passed' as a string is read",
      objectives.parse_verdict({"passed": "false"})[0] is False
      and objectives.parse_verdict({"passed": "true"})[0] is True)
check("no answer from the model is not a fail",
      objectives.parse_verdict(None) is None)

print(f"\n{'PASS' if not failures else 'FAIL'} — class notes and goals\n")
sys.exit(1 if failures else 0)
