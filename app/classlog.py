"""What the learner's class actually covered.

The textbook says what a chapter contains; it does not say where the class is
inside it. A course moves by lesson, not by chapter, and a tutor that teaches
the whole of chapter 3 to someone whose class has only reached its first two
pages is teaching ahead of the teacher and around the gap.

So the learner logs each class: a line of text ("Akkusativ, pages 44-47"),
a photo of the board, the handout, the page of the book. The model turns that
into a structured entry — chapter, pages, topics, grammar, vocabulary — and the
most recent entries go into the tutor's prompt, ahead of the chapter as a
whole.

Model output is treated as untrusted, as everywhere else here:

- A chapter or page range the learner *typed* wins over whatever the model
  read. "Kapitel 3" in the note is a fact; the model's guess is not.
- The chapter must exist in the course, or it is dropped.
- Vocabulary is checked against the text the entry was made from, wherever
  there is text to check against. A word that appears nowhere in the note or
  in the PDF's text layer was invented, and invented German taught as the
  class's own vocabulary is the worst failure available here. It is dropped.
  A photo has no text layer, so its words cannot be checked; they are kept and
  marked unchecked, and the UI says so.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from .homework import normalise

log = logging.getLogger(__name__)

# How long a class stays "recent" enough to steer the tutor. A course meets a
# few times a week; three weeks old is no longer where the class is.
RECENT_DAYS = 21
BRIEF_ENTRIES = 2
MAX_VOCAB = 40


@dataclass
class Word:
    de: str
    en: str = ""
    # True: found in the source text. None: there was no text to check
    # against (a photo). Words that failed the check are not kept at all.
    checked: bool | None = None
    # A noun's article, separately: "page" (printed with the noun in the
    # source), "agreed" (the reading and an independent lexicon check
    # agree), or "unsure" (they disagreed, so the article was removed rather
    # than taught wrong). "" for words that are not nouns with an article.
    article: str = ""


@dataclass
class Entry:
    id: str
    day: str                          # the class's date, ISO
    created: str
    chapter: int | None = None
    pages: str = ""
    summary: str = ""
    topics: list[str] = field(default_factory=list)
    grammar: list[str] = field(default_factory=list)
    vocab: list[Word] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    note: str = ""                    # what the learner typed, verbatim
    sources: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)   # failed the check

    def describe(self) -> str:
        """For the system prompt."""
        where = []
        if self.chapter:
            where.append(f"chapter {self.chapter}")
        if self.pages:
            single = self.pages.isdigit()
            where.append(f"page{'' if single else 's'} {self.pages}")
        head = f"Class on {self.day}" + (f" ({', '.join(where)})" if where else "")
        parts = [head + (f": {self.summary}" if self.summary else ".")]
        if self.topics:
            parts.append("Topics: " + ", ".join(self.topics) + ".")
        if self.grammar:
            parts.append("Grammar: " + ", ".join(self.grammar) + ".")
        if self.vocab:
            parts.append("Vocabulary: " + ", ".join(
                f"«{w.de}»" + (f" ({w.en})" if w.en else "")
                for w in self.vocab[:20]) + ".")
        if self.phrases:
            parts.append("Phrases: " + ", ".join(f"«{p}»" for p in self.phrases[:8])
                         + ".")
        return " ".join(parts)


# ---- reading what the learner typed ----------------------------------------

_CHAPTER = re.compile(r"\b(?:kapitel|chapter|ch\.?|kap\.?|lektion|unit)\s*(\d{1,2})\b",
                      re.IGNORECASE)
_PAGES = re.compile(
    r"\b(?:pages?|pp?\.?|seiten?|s\.)\s*(\d{1,3}(?:\s*[-–]\s*\d{1,3})?"
    r"(?:\s*(?:,|and|und|&)\s*\d{1,3}(?:\s*[-–]\s*\d{1,3})?)*)",
    re.IGNORECASE)


def stated_chapter(note: str) -> int | None:
    m = _CHAPTER.search(note or "")
    return int(m.group(1)) if m else None


def stated_pages(note: str) -> str:
    m = _PAGES.search(note or "")
    if not m:
        return ""
    raw = re.sub(r"\s*(?:and|und|&)\s*", ",", m.group(1), flags=re.IGNORECASE)
    return re.sub(r"\s+", "", raw.replace("–", "-"))


# ---- turning the model's reading into an entry ------------------------------

def _strings(items, limit: int, max_len: int = 80) -> list[str]:
    out, seen = [], set()
    for item in items or []:
        if not isinstance(item, str):
            continue
        text = item.strip()
        key = text.lower()
        if text and len(text) <= max_len and key not in seen:
            seen.add(key)
            out.append(text)
    return out[:limit]


_ARTICLE = re.compile(r"^(?:der|die|das|ein|eine)\s+", re.IGNORECASE)


def _found_in(word: str, folded_source: str) -> bool:
    """Is `word` (article aside) present in the source text, as whole words?"""
    bare = normalise(_ARTICLE.sub("", word))
    if not bare:
        return False
    return f" {bare} " in folded_source


_DEFINITE = re.compile(r"^(der|die|das)\s+(.+)$", re.IGNORECASE)


def articles_to_check(entry: Entry, source_text: str) -> list[str]:
    """Nouns whose article is not printed with them in the source.

    Marks the ones that are as "page", and returns the bare nouns of the
    rest for an independent lexicon check. Asked to give an article "only if
    it is shown", the model still wrote «der Brot» for a page that says
    «das Brot» nowhere near it — so the article is never taken on its word.
    """
    folded = " " + normalise(f"{entry.note}\n{source_text}") + " "
    out = []
    for w in entry.vocab:
        m = _DEFINITE.match(w.de)
        if not m:
            continue
        if f" {normalise(w.de)} " in folded:
            w.article = "page"
        else:
            out.append(m.group(2))
    return out


def apply_lexicon(entry: Entry, lexicon: dict[str, tuple[str, str]]) -> None:
    """Keep an unprinted article only where the lexicon agrees with it.

    `lexicon` is handouts.parse_lexicon output: {noun_key: (article, en)}.
    Disagreement removes the article instead of picking one: a noun without
    its article is incomplete, a noun with the wrong one is a mistake the
    learner will practise until it sticks.
    """
    from .handouts import noun_key

    for w in entry.vocab:
        m = _DEFINITE.match(w.de)
        if not m or w.article == "page":
            continue
        said = lexicon.get(noun_key(m.group(2)), ("", ""))[0]
        if said == m.group(1).lower():
            w.article = "agreed"
        else:
            w.de, w.article = m.group(2), "unsure"


def parse_extracted(raw: dict, *, note: str, source_text: str,
                    sources: list[str], day: date, chapters: set[int]
                    ) -> Entry:
    """Build an Entry from the model's JSON, checking everything checkable."""
    raw = raw if isinstance(raw, dict) else {}

    chapter = stated_chapter(note)
    if chapter is None:
        try:
            chapter = int(raw.get("chapter"))
        except (TypeError, ValueError):
            chapter = None
    if chapter is not None and chapters and chapter not in chapters:
        log.info("class note names chapter %s, which is not in the course", chapter)
        chapter = None

    pages = stated_pages(note) or str(raw.get("pages") or "").strip()[:20]
    if pages and not re.fullmatch(r"[\d,\-–\s]+", pages):
        pages = ""

    # Words can be checked only against text that exists. Below a few
    # sentences of it, "not found" says more about the source than the word.
    folded = " " + normalise(f"{note}\n{source_text}") + " "
    can_check = len(folded) > 120

    vocab: list[Word] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for item in raw.get("vocab") or []:
        if isinstance(item, str):
            de, en = item.strip(), ""
        elif isinstance(item, dict):
            de = str(item.get("de", "")).strip()
            en = str(item.get("en", "")).strip()
        else:
            continue
        if not de or len(de) > 60 or de.lower() in seen:
            continue
        seen.add(de.lower())
        if can_check:
            if not _found_in(de, folded):
                dropped.append(de)
                continue
            vocab.append(Word(de=de, en=en[:80], checked=True))
        else:
            vocab.append(Word(de=de, en=en[:80], checked=None))
    if dropped:
        log.info("class note: dropped %d word(s) not in the source: %s",
                 len(dropped), ", ".join(dropped[:8]))

    now = datetime.now()
    return Entry(
        id=now.strftime("%Y-%m-%d_%H%M%S"),
        day=day.isoformat(),
        created=now.isoformat(timespec="seconds"),
        chapter=chapter,
        pages=pages,
        summary=str(raw.get("summary") or "").strip()[:400],
        topics=_strings(raw.get("topics"), 8),
        grammar=_strings(raw.get("grammar"), 8),
        vocab=vocab[:MAX_VOCAB],
        phrases=_strings(raw.get("phrases"), 10, max_len=120),
        note=note.strip()[:2000],
        sources=sources,
        dropped=dropped,
    )


# ---- persistence -----------------------------------------------------------

def _dir(study_path: Path) -> Path:
    return Path(study_path) / "classlog"


def save(entry: Entry, study_path: Path) -> Path:
    d = _dir(study_path)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{entry.id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(entry), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def _from_dict(d: dict) -> Entry:
    return Entry(
        id=str(d.get("id", "")),
        day=str(d.get("day", "")),
        created=str(d.get("created", "")),
        chapter=d.get("chapter"),
        pages=str(d.get("pages", "")),
        summary=str(d.get("summary", "")),
        topics=[str(x) for x in d.get("topics", [])],
        grammar=[str(x) for x in d.get("grammar", [])],
        vocab=[Word(de=str(w.get("de", "")), en=str(w.get("en", "")),
                    checked=w.get("checked"), article=str(w.get("article", "")))
               for w in d.get("vocab", []) if isinstance(w, dict)],
        phrases=[str(x) for x in d.get("phrases", [])],
        note=str(d.get("note", "")),
        sources=[str(x) for x in d.get("sources", [])],
        dropped=[str(x) for x in d.get("dropped", [])],
    )


def load_all(study_path: Path) -> list[Entry]:
    """Every entry, oldest class first."""
    d = _dir(study_path)
    if not d.is_dir():
        return []
    out = []
    for path in d.glob("*.json"):
        try:
            out.append(_from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            log.warning("skipping unreadable class note %s", path.name)
    return sorted(out, key=lambda e: (e.day, e.created))


def delete(study_path: Path, entry_id: str) -> bool:
    if not re.fullmatch(r"[\w\-]+", entry_id or ""):
        return False
    path = _dir(study_path) / f"{entry_id}.json"
    if not path.is_file():
        return False
    path.unlink()
    return True


def recent(study_path: Path, today: date) -> list[Entry]:
    cutoff = (today - timedelta(days=RECENT_DAYS)).isoformat()
    return [e for e in load_all(study_path) if e.day >= cutoff]


def brief(study_path: Path, today: date) -> str:
    """The latest classes, newest first, for the system prompt."""
    entries = recent(study_path, today)[-BRIEF_ENTRIES:]
    return "\n".join(e.describe() for e in reversed(entries))


def latest_chapter(study_path: Path) -> int | None:
    for e in reversed(load_all(study_path)):
        if e.chapter:
            return e.chapter
    return None
