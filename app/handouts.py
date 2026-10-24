"""Vocabulary from class handouts, with the article checked twice.

The handouts list nouns in «der» / «das» / «die» columns, which makes them
better A1 material than the textbook: it is what the class is actually using,
and the article — the hard part of every German noun — is given by the layout.

That is also why reading them is dangerous. A noun put in the wrong column
teaches the wrong article, and the learner has no way to know. pypdf's text
extraction loses the columns entirely on these docx-export PDFs, so a row with
an empty middle cell shifts its right-hand noun one column left: «die Lampe»
becomes «das Lampe», silently.

So no single reading is trusted. Each noun gets up to three independent
opinions on its article:

  position  where the word sits on the page, from pypdfium2's per-character
            boxes — real coordinates, which survive the docx export
  vision    the model reading the page image
  lexicon   the model asked, separately and without the page, what the
            article of this noun is

An entry is *confirmed* only when at least two opinions exist and every
opinion agrees. Anything else is *pending*: shown to the learner with the
readings side by side, and kept out of the flashcards until they decide. A
pending list is a chore; a wrong article learned is a mistake that takes
months to unlearn.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .homework import normalise

log = logging.getLogger(__name__)

CONFIRMED, PENDING, REJECTED = "confirmed", "pending", "rejected"
ARTICLES = ("der", "das", "die")

# Column headings: the article itself, or the grammatical gender it stands for.
_HEADINGS = {
    "der": "der", "das": "das", "die": "die",
    "maskulin": "der", "maskulinum": "der", "m": "der",
    "neutrum": "das", "neutral": "das", "n": "das",
    "feminin": "die", "femininum": "die", "f": "die",
}


@dataclass
class Item:
    de: str                          # "der Tisch"
    en: str = ""
    status: str = PENDING
    readings: dict[str, str] = field(default_factory=dict)   # source -> article
    page: int | None = None

    @property
    def noun(self) -> str:
        parts = self.de.split(" ", 1)
        return parts[1] if len(parts) == 2 and parts[0].lower() in ARTICLES else self.de


@dataclass
class Deck:
    name: str
    source: str
    created: str
    entries: list[Item] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return re.sub(r"[^\w\-]+", "-", self.name.lower()).strip("-") or "deck"

    def counts(self) -> dict[str, int]:
        out = {CONFIRMED: 0, PENDING: 0, REJECTED: 0}
        for e in self.entries:
            out[e.status] = out.get(e.status, 0) + 1
        return out


# ---- reading 1: position -----------------------------------------------------

@dataclass
class _Word:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


def words_from_pdf_page(pdf_bytes: bytes | str, page_number: int) -> list[_Word]:
    """Every word on the page with its box, in PDF points (y grows upward)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        tp = pdf[page_number - 1].get_textpage()
        words: list[_Word] = []
        cur: list[tuple[str, tuple]] = []

        def flush() -> None:
            if cur:
                text = "".join(c for c, _ in cur)
                boxes = [b for _, b in cur]
                words.append(_Word(text=text,
                                   x0=min(b[0] for b in boxes),
                                   x1=max(b[2] for b in boxes),
                                   bottom=min(b[1] for b in boxes),
                                   top=max(b[3] for b in boxes)))
                cur.clear()

        prev_box = None
        for i in range(tp.count_chars()):
            ch = tp.get_text_range(i, 1)
            if not ch or ch.isspace():
                flush()
                prev_box = None
                continue
            box = tp.get_charbox(i)
            # A big horizontal jump with no space character between is still
            # two words — table cells are often laid out exactly like that.
            if prev_box is not None and box[0] - prev_box[2] > 3 * max(
                    1.0, prev_box[2] - prev_box[0]):
                flush()
            cur.append((ch, box))
            prev_box = box
        flush()
        return words
    finally:
        pdf.close()


def _lines(words: list[_Word]) -> list[list[_Word]]:
    """Group words into visual lines, top of the page first."""
    lines: list[list[_Word]] = []
    for w in sorted(words, key=lambda w: -w.top):
        h = max(1.0, w.top - w.bottom)
        for line in lines:
            ref = line[0]
            if abs(((ref.top + ref.bottom) / 2) - ((w.top + w.bottom) / 2)) < h * 0.5:
                line.append(w)
                break
        else:
            lines.append([w])
    return [sorted(line, key=lambda w: w.x0) for line in lines]


def read_by_position(words: list[_Word]) -> list[tuple[str, str]]:
    """[(cell text, article)] from a page laid out in article columns.

    Finds the heading row — a line holding all three articles, or their
    gender names — and assigns every word below it to the nearest heading by
    horizontal position. Returns [] when there is no such row: a page that is
    not a column table is not guessed at.
    """
    lines = _lines(words)
    heading_idx, centres = None, {}
    for i, line in enumerate(lines):
        # A heading row is short. «Die Katze ist das Tier der Nachbarin»
        # holds all three articles too, and is not one.
        if len(line) > 6:
            continue
        found = {}
        for w in line:
            key = _HEADINGS.get(w.text.strip(":.").lower())
            if key and key not in found:
                found[key] = w.cx
        if len(found) == 3:
            heading_idx, centres = i, found
            break
    if heading_idx is None:
        return []

    order = sorted(centres.items(), key=lambda kv: kv[1])
    # Column boundaries halfway between heading centres.
    bounds = [(order[k][1] + order[k + 1][1]) / 2 for k in range(2)]

    def column(x: float) -> str:
        if x < bounds[0]:
            return order[0][0]
        if x < bounds[1]:
            return order[1][0]
        return order[2][0]

    out: list[tuple[str, str]] = []
    for line in lines[heading_idx + 1:]:
        cells: dict[str, list[str]] = {}
        for w in line:
            cells.setdefault(column(w.cx), []).append(w.text)
        for article, parts in cells.items():
            text = " ".join(parts).strip()
            # A heading repeated further down, or a stray article.
            if not text or text.lower() in _HEADINGS:
                continue
            out.append((text, article))
    return out


# ---- normalising nouns across readings ---------------------------------------

_PLURAL_TAIL = re.compile(r"\s*[,;(/].*$")


def noun_key(text: str) -> str:
    """'der Tisch, -e' / 'Tisch (-e)' / 'tisch' -> 'tisch'."""
    text = _PLURAL_TAIL.sub("", text.strip())
    parts = text.split()
    if parts and parts[0].lower() in ARTICLES:
        parts = parts[1:]
    return normalise(" ".join(parts))


def clean_noun(text: str) -> str:
    """The noun as it should be shown: plural note and article removed."""
    text = _PLURAL_TAIL.sub("", text.strip())
    parts = text.split()
    if parts and parts[0].lower() in ARTICLES:
        parts = parts[1:]
    return " ".join(parts)


def parse_vision(raw: dict | None) -> list[tuple[str, str]]:
    """The model's reading of the page: [(noun, article)]."""
    out = []
    for item in (raw or {}).get("nouns") or []:
        if not isinstance(item, dict):
            continue
        noun = clean_noun(str(item.get("noun", "")))
        article = str(item.get("article", "")).strip().lower()
        if noun and article in ARTICLES and len(noun) <= 40:
            out.append((noun, article))
    return out


def parse_lexicon(raw: dict | None) -> dict[str, tuple[str, str]]:
    """{noun_key: (article, english)} from the model's own knowledge."""
    out = {}
    for item in (raw or {}).get("nouns") or []:
        if not isinstance(item, dict):
            continue
        noun = clean_noun(str(item.get("noun", "")))
        article = str(item.get("article", "")).strip().lower()
        if noun and article in ARTICLES:
            out[noun_key(noun)] = (article, str(item.get("en", "")).strip()[:80])
    return out


def combine(position: list[tuple[str, str]], vision: list[tuple[str, str]],
            lexicon: dict[str, tuple[str, str]], page: int | None = None
            ) -> list[Item]:
    """Merge the readings, confirming only where at least two agree and none
    disagrees."""
    order: list[str] = []
    shown: dict[str, str] = {}
    readings: dict[str, dict[str, str]] = {}

    def add(source: str, noun_text: str, article: str) -> None:
        key = noun_key(noun_text)
        if not key or not re.search(r"[a-zß]", key):
            return
        if key not in readings:
            order.append(key)
            readings[key] = {}
            shown[key] = clean_noun(noun_text)
        readings[key].setdefault(source, article)

    for text, article in position:
        add("position", text, article)
    for noun, article in vision:
        add("vision", noun, article)
    for key in list(order):
        if key in lexicon:
            readings[key]["lexicon"] = lexicon[key][0]

    items = []
    for key in order:
        r = readings[key]
        votes = set(r.values())
        agreed = len(r) >= 2 and len(votes) == 1
        # Shown with the majority article, or the page's own reading when
        # there is no majority: the learner is deciding, so show them what
        # the handout says.
        article = (next(iter(votes)) if agreed
                   else r.get("position") or r.get("vision") or r.get("lexicon"))
        en = lexicon.get(key, ("", ""))[1]
        items.append(Item(de=f"{article} {shown[key]}", en=en,
                          status=CONFIRMED if agreed else PENDING,
                          readings=dict(r), page=page))
    return items


# ---- persistence -------------------------------------------------------------

def _dir(study_path: Path) -> Path:
    return Path(study_path) / "decks"


def save_deck(deck: Deck, study_path: Path) -> Path:
    d = _dir(study_path)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{deck.slug}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(deck), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_decks(study_path: Path) -> list[Deck]:
    d = _dir(study_path)
    if not d.is_dir():
        return []
    out = []
    for path in sorted(d.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            out.append(Deck(
                name=str(raw.get("name", path.stem)),
                source=str(raw.get("source", "")),
                created=str(raw.get("created", "")),
                entries=[Item(de=str(e.get("de", "")), en=str(e.get("en", "")),
                              status=str(e.get("status", PENDING)),
                              readings=dict(e.get("readings") or {}),
                              page=e.get("page"))
                         for e in raw.get("entries", []) if isinstance(e, dict)],
            ))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            log.warning("skipping unreadable deck %s", path.name)
    return out


def find_deck(study_path: Path, slug: str) -> Deck | None:
    return next((d for d in load_decks(study_path) if d.slug == slug), None)


def decide(study_path: Path, slug: str, index: int, status: str,
           de: str | None = None, en: str | None = None) -> Item | None:
    """The learner's verdict on one entry, optionally correcting it."""
    deck = find_deck(study_path, slug)
    if deck is None or not 0 <= index < len(deck.entries) or status not in (
            CONFIRMED, PENDING, REJECTED):
        return None
    item = deck.entries[index]
    if de is not None and de.strip():
        item.de = de.strip()[:80]
    if en is not None:
        item.en = en.strip()[:80]
    item.status = status
    if status == CONFIRMED:
        item.readings["learner"] = item.de.split(" ", 1)[0].lower()
    save_deck(deck, study_path)
    return item


def new_deck(name: str, source: str, items: list[Item]) -> Deck:
    # One noun once, even when it appears on several pages.
    seen, unique = set(), []
    for it in items:
        key = noun_key(it.de)
        if key in seen:
            continue
        seen.add(key)
        unique.append(it)
    return Deck(name=name, source=source,
                created=datetime.now().isoformat(timespec="seconds"),
                entries=unique)
