"""Read a course book's contents pages with the vision model.

The text parser in scripts/ingest_textbook.py reads the A1 and A2 volumes of
the learner's course from their text layer. The B1 and B2 volumes are scans:
not one character of text on any page. A course that stops being followable
the day the class moves up a level is not much of a course companion, so this
reads the contents pages as images instead.

It is held to the same rule as everything else here that comes from a model:
nothing is written unless it can be checked. A scan offers nothing to check
a reading against, so each part is read twice, by methods that fail
differently, and the two readings must agree chapter by chapter. Where the
PDF does have text, every can-do statement and grammar item must also be
found in it.

What was measured, on the A1 and A2 books, whose text layers give the right
answer, and on the B1 and B2 scans:

- One question per page beats one question for both. Asked for the whole
  syllabus from both contents pages at once, the model shifted the can-do or
  grammar lists by a chapter.
- The chapter list reads near-perfectly, whole or in halves.
- The syllabus page is the hard part. Read whole, it is usually right but
  can drift a row. Cut into top and bottom halves it is badly wrong: the cut
  goes through rows (A2 grammar agreement with the text: 0.13–0.54). Its
  columns read separately lose the grouping into chapters altogether.
- What does work as the second reading is to cut the page into one strip per
  chapter, using the page's own geometry: the dotted rules between rows are
  the only ink that crosses the blank gutter between the columns. A strip
  holds exactly one chapter, so nothing in it can be attributed to another.
  Each strip is shown with the page's column headings above it, since on the
  B2 page both columns are bullet lists and only the heading says which is
  grammar.

End-of-chapter grammar summaries are not read: on a scan, finding them means
looking at every page of the book.
"""

from __future__ import annotations

import difflib
import io
import logging
import re
from typing import TYPE_CHECKING

from . import curriculum, documents
from .homework import normalise

if TYPE_CHECKING:  # the deciding half is testable without an HTTP client
    from .llm import LlmClient

log = logging.getLogger(__name__)

# How far into the book to look for the contents pages.
SCAN_PAGES = 14
# Two readings of the same title or list must be at least this alike.
AGREE = 0.8


class ReadError(RuntimeError):
    pass


CLASSIFY = """\
Look at this page from a German course book. Which kind of page is it?

- "chapters": a table of contents listing the chapters, with their numbers, \
titles and page numbers (often headed INHALT, KAPITEL, THEMEN)
- "syllabus": a page listing, per chapter, what the learner will be able to \
do and the grammar (often headed SPRACHHANDLUNGEN and GRAMMATIK)
- "other": anything else

Reply with JSON only: {"kind": "chapters"}"""

CHAPTERS = """\
This is the table-of-contents page of a German course book. List every \
chapter on it, in order:
- "number": the chapter number
- "title": the chapter title exactly as printed
- "book_page": the page number printed for the chapter, or null
- "units": the numbered sections of the chapter, [{"number": 1, "title": \
"..."}], or [] if none are printed
- "topics": the topic keywords (Themen) printed for the chapter
- "can_do": the can-do statements (SPRACHHANDLUNGEN), only if this page \
prints them, separated as on the page by «|»; otherwise []

Copy text exactly as printed, with umlauts. Do not add anything that is not \
on the page.

Reply with JSON only: {"chapters": [...]}"""

SYLLABUS = """\
This page of a German course book's table of contents is laid out in {n} \
rows, one per chapter, top to bottom. {layout}

Return exactly {n} rows, in order from the top. For each row give the \
can-do statements and the grammar bullets that sit in that row. Copy text \
exactly as printed, with umlauts. A statement that wraps onto a second line \
is still one statement.

Reply with JSON only: {{"rows": [{{"can_do": ["..."], "grammar": ["..."]}}]}}"""

ROW = """\
This image shows the column headings of a German course book's \
table-of-contents page, and below them ONE row of that table: one chapter. \
{layout}

Give the can-do statements and the grammar points in that row. Copy text \
exactly as printed, with umlauts. A statement that wraps onto a second line \
is still one statement.

Reply with JSON only: {{"can_do": ["..."], "grammar": ["..."]}}"""

COLUMN_ROW = """\
This image shows the heading of one column of a German course book's \
table-of-contents page, «{heading}», and below it that column's entry for ONE \
chapter. List the {what} in it, exactly as printed, with umlauts. {how}

Reply with JSON only: {{"items": ["..."]}}"""

HEADING = """\
This is the top of one column of a table on a German course book page. What \
is the column's heading, exactly as printed? If there is none, give "".

Reply with JSON only: {"heading": "..."}"""

# Which heading means which field. Anything else (STRATEGIETRAINING in B2,
# THEMEN, SEITE) is not part of the syllabus and is never read as one.
FIELDS = {
    "SPRACHHANDLUNG": ("can_do", "can-do statements",
                       "They are separated by «|»; a statement that wraps onto "
                       "the next line is still one statement."),
    "GRAMMATIK": ("grammar", "grammar points",
                  "Each bullet point is one item; a bullet that wraps onto the "
                  "next line is still one item."),
}


SPELLING = """\
These lines were read off a scanned page of a German course book by a \
machine, which sometimes misreads a letter: «Modalverbalalternativen» for \
«Modalverbalternativen», «GROBER» for «GROßER». Chapter titles can be puns \
or wordplay; leave those alone.

List only the lines that contain a word which is not real German, or is \
very likely a misreading, with the correction you think was printed. If all \
are fine, return an empty list.

Reply with JSON only: {"suspect": [{"line": "...", "probably": "..."}]}"""


async def spelling_suspects(llm: LlmClient, chapters: list[curriculum.Chapter]
                            ) -> list[str]:
    """Lines a text-only check thinks were misread, for the learner to check.

    Voting between readings of the same image cannot catch a slip they all
    share, and on the B1 and B2 scans some were shared: «Zeugaussage»,
    «Modalverbalalternativen», «GROBER». A second opinion that never sees the
    pixels does not share their misleading shapes. It only flags — a
    "correction" from a model that cannot see the page is a guess, and the
    page is right there for the learner to look at.
    """
    lines = []
    for c in chapters:
        lines += [f"chapter {c.number}: {c.title}"]
        lines += [f"chapter {c.number}: {x}" for x in c.can_do + c.grammar]
    raw = await llm.complete_json(
        [{"role": "system", "content": SPELLING},
         {"role": "user", "content": "\n".join(lines)}],
        temperature=0.0, max_tokens=1500)
    out = []
    for s in (raw or {}).get("suspect") or []:
        if isinstance(s, dict) and s.get("line") and s.get("probably"):
            line, fix = str(s["line"]).strip(), str(s["probably"]).strip()
            fix = re.sub(r"^chapter \d+:\s*", "", fix)
            if (line in lines and normalise(line.split(": ", 1)[-1])
                    != normalise(fix)):
                out.append(f"{line}  (perhaps «{fix}»)")
    return out


def _layout(fields: dict[str, str]) -> str:
    """Tell the model which columns this page has, and which it has not.

    Told only that the columns "can include" can-do statements, the model
    filled the can-do field on B2's page — which has none — from its
    strategy column, and on one row invented two sentences outright.
    """
    left, right = fields.get("left", ""), fields.get("right", "")
    names = {"can_do": "SPRACHHANDLUNGEN (the can-do statements)",
             "grammar": "GRAMMATIK (the grammar points)"}
    parts = [f"The left column is {names[left]}." if left else
             "The left column is not part of the syllabus: ignore it.",
             f"The right column is {names[right]}." if right else
             "The right column is not part of the syllabus: ignore it."]
    for field in ("can_do", "grammar"):
        if field not in (left, right):
            parts.append(f"This page has no {names[field].split(' ')[0]} "
                         f"column, so \"{field}\" is always [].")
    return " ".join(parts)


def _ask(text: str, pages: list[documents.Page], tile: bool) -> list[dict]:
    return [{"role": "user", "content": [{"type": "text", "text": text},
                                         *documents.image_parts(pages, tile=tile)]}]


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalise(a), normalise(b)).ratio()


def _list_sim(a: list[str], b: list[str]) -> float:
    """How alike two lists are, item by best-matching item, both ways."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    def one_way(x, y):
        return sum(max(_sim(i, j) for j in y) for i in x) / len(x)

    return min(one_way(a, b), one_way(b, a))


def _strings(items) -> list[str]:
    out = []
    for item in items or []:
        if isinstance(item, str):
            # Soft hyphens are typesetting, not text: the A1 book's layer has
            # «trenn\xadbare», which no reading of the page should copy.
            item = item.replace("\xad", "")
            # "Uhrzeiten erfragen | sich verabreden" is how the book prints
            # several statements on one line.
            out += [p.strip(" •·-") for p in item.split("|") if p.strip(" •·-")]
    return out


# ---- the page's geometry ----------------------------------------------------

def _gutter(img) -> int:
    """The x position of the blank strip between the two columns.

    Found from the pixels rather than assumed at the middle: on these pages
    the left column runs past the centre, and a cut at half-width slices its
    text in two.
    """
    gray = img.convert("L")
    w, h = gray.size
    top, bottom = int(h * 0.12), int(h * 0.92)     # skip headings and footer
    px = gray.load()
    xs = list(range(int(w * 0.35), int(w * 0.75), 2))
    ink = [sum(1 for y in range(top, bottom, 3) if px[x, y] < 150) for x in xs]
    # The widest blank band, not the single emptiest x: on the B1 scan the
    # thin strip between a column of bullet dots and the grammar text is as
    # empty as the real gutter, just narrower. A little ink is allowed for
    # the rules that cross the gutter, and for scan grain.
    quiet = [i <= min(ink) + 6 for i in ink]
    best, run_start = (0, len(xs) // 2, len(xs) // 2), None
    for i, q in enumerate([*quiet, False]):
        if q and run_start is None:
            run_start = i
        elif not q and run_start is not None:
            if i - run_start > best[0]:
                best = (i - run_start, run_start, i - 1)
            run_start = None
    return xs[(best[1] + best[2]) // 2]


def _separators(img, gutter: int) -> list[int]:
    """y positions of the rules between rows: ink that crosses the gutter.

    Text never crosses the blank strip between the two columns; the dotted
    lines between chapter rows run straight through it. That holds on the
    born-digital pages and on the scans alike, where grain and show-through
    make the gaps between lines too noisy to measure.
    """
    gray = img.convert("L")
    w, h = gray.size
    px = gray.load()
    band = range(max(0, gutter - 12), min(w, gutter + 12))
    hit = [sum(1 for x in band if px[x, y] < 170) >= 2 for y in range(h)]
    edge = int(h * 0.04)        # a scan's own border is not a rule
    out, y = [], 0
    while y < h:
        if hit[y]:
            start = y
            while y < h and hit[y]:
                y += 1
            mid = (start + y) // 2
            # A rule is thin, and runs across both columns, not just the
            # gutter — that leaves out a heading banner or a picture. The
            # scanned B1 page is skewed by about 11 px across its width, so
            # the rule is looked for a little above and below.
            if edge < mid < h - edge and y - start <= 8 and all(
                    any(px[x, yy] < 170 for x in range(int(w * f) - 30, int(w * f) + 30)
                        for yy in range(max(0, mid - 14), min(h, mid + 15)))
                    for f in (0.2, 0.85)):
                out.append(mid)
        else:
            y += 1
    return out


def row_strips(img, n: int) -> list[tuple[int, int]] | None:
    """(top, bottom) of each of the `n` chapter rows, or None if the page
    does not divide into exactly `n` rows — a refusal, not a guess."""
    h = img.size[1]
    seps = _separators(img, _gutter(img))
    if len(seps) != n - 1 or n < 2:
        log.info("found %d row rule(s), wanted %d", len(seps), n - 1)
        return None
    gaps = sorted(b - a for a, b in zip(seps, seps[1:]))
    step = gaps[len(gaps) // 2] if gaps else h // (n + 1)
    # The first and last rows have a rule on one side only. The first is
    # given a little more than a typical row above its rule — not more, or
    # it swallows the column headings, which the heading band needs; the last
    # gets more room below, where at most a footer can intrude.
    bounds = [max(0, seps[0] - int(step * 1.15)), *seps,
              min(h, seps[-1] + int(step * 1.6))]
    return list(zip(bounds, bounds[1:]))


def _part(image) -> dict:
    image.thumbnail((documents.SEND_SIDE, documents.SEND_SIDE))
    return {"type": "image_url", "image_url": {
        "url": documents._data_url(documents._encode(image))}}


def _stack(top, bottom):
    from PIL import Image

    w = max(top.size[0], bottom.size[0])
    sheet = Image.new("RGB", (w, top.size[1] + bottom.size[1] + 8), "white")
    sheet.paste(top, (0, 0))
    sheet.paste(bottom, (0, top.size[1] + 8))
    return sheet


class Rows:
    """A syllabus page cut into one strip per chapter row."""

    def __init__(self, page: documents.Page, n: int) -> None:
        from PIL import Image

        self.img = Image.open(io.BytesIO(page.image)).convert("RGB")
        self.strips = row_strips(self.img, n)
        self.gutter = _gutter(self.img)

    @property
    def ok(self) -> bool:
        return self.strips is not None

    def _heading(self, left: int, right: int):
        h = self.img.size[1]
        return self.img.crop((left, int(h * 0.04), right, self.strips[0][0]))

    def whole(self, i: int) -> list[dict]:
        """Row `i` across the full width, under the page's headings."""
        w = self.img.size[0]
        top, bottom = self.strips[i]
        return [_part(_stack(self._heading(0, w),
                             self.img.crop((0, top, w, bottom))))]

    def _side(self, side: str) -> tuple[int, int]:
        return (0, self.gutter) if side == "left" else (self.gutter, self.img.size[0])

    def column(self, i: int, side: str) -> list[dict]:
        """One column of row `i`, under that column's heading. Safe for one
        row, where there is no grouping into chapters left to lose."""
        a, b = self._side(side)
        top, bottom = self.strips[i]
        return [_part(_stack(self._heading(a, b), self.img.crop((a, top, b, bottom))))]

    async def fields(self, llm: LlmClient) -> dict[str, tuple[str, str]]:
        """{side: (field, heading)} for the columns that are syllabus."""
        out = {}
        for side in ("left", "right"):
            raw = await llm.complete_json(
                [{"role": "user", "content": [
                    {"type": "text", "text": HEADING},
                    _part(self._heading(*self._side(side)))]}],
                temperature=0.0, max_tokens=60)
            heading = str((raw or {}).get("heading", "")).strip()
            for key, (field, _, _) in FIELDS.items():
                if key in heading.upper():
                    out[side] = (field, heading)
        return out


# ---- reading ------------------------------------------------------------------

async def find_contents(llm: LlmClient, doc: documents.Document,
                        progress=None) -> tuple[int, int]:
    """(chapters page, syllabus page), 1-based, by looking at each page.

    One page's label is not trusted on its own. On both the A1 and B1 books
    the model called the cover a contents page, and the lesson pages after
    the syllabus "chapters" too. What did hold on both was the pair: the
    first chapters page directly followed by a syllabus page is the contents
    spread.
    """
    kinds: dict[int, str] = {}
    for n in range(1, min(SCAN_PAGES, doc.page_count) + 1):
        if progress:
            progress(f"Looking for the contents: page {n}")
        raw = await llm.complete_json(_ask(CLASSIFY, [doc.page(n)], tile=False),
                                      temperature=0.0, max_tokens=50)
        kinds[n] = str((raw or {}).get("kind", "")).lower()
        if kinds.get(n - 1) == "chapters" and kinds[n] == "syllabus":
            return n - 1, n
    raise ReadError(f"no contents spread (a chapter list followed by a syllabus "
                    f"page) found in the first {SCAN_PAGES} pages — give the "
                    f"pages with --contents, e.g. --contents 4-5")


def _parse_chapters(raw: dict | None) -> list[curriculum.Chapter]:
    out = []
    for c in (raw or {}).get("chapters") or []:
        if not isinstance(c, dict):
            continue
        try:
            number = int(c.get("number"))
        except (TypeError, ValueError):
            continue
        try:
            page = int(c["book_page"]) if c.get("book_page") is not None else None
        except (TypeError, ValueError):
            page = None
        units = []
        for i, u in enumerate(c.get("units") or [], 1):
            if isinstance(u, dict) and str(u.get("title", "")).strip():
                try:
                    un = int(u.get("number", i))
                except (TypeError, ValueError):
                    un = i
                units.append(curriculum.Unit(un, str(u["title"]).strip()))
            elif isinstance(u, str) and u.strip():
                units.append(curriculum.Unit(i, re.sub(r"^\d+\s+", "", u.strip())))
        out.append(curriculum.Chapter(
            number=number, title=str(c.get("title", "")).strip(),
            book_page=page, units=units,
            topics=_strings(c.get("topics")),
            can_do=_strings(c.get("can_do")),
        ))
    return out


def _parse_rows(raw: dict | None) -> list[tuple[list[str], list[str]]]:
    return [(_strings(r.get("can_do")), _strings(r.get("grammar")))
            for r in (raw or {}).get("rows") or [] if isinstance(r, dict)]


async def read_row(llm: LlmClient, images: list[dict], layout: str
                   ) -> tuple[list[str], list[str]]:
    """A full-width row: both fields at once."""
    raw = await llm.complete_json(
        [{"role": "user", "content": [
            {"type": "text", "text": ROW.format(layout=layout)}, *images]}],
        temperature=0.0, max_tokens=1000)
    raw = raw if isinstance(raw, dict) else {}
    return _strings(raw.get("can_do")), _strings(raw.get("grammar"))


async def read_row_columns(llm: LlmClient, rows: Rows, i: int,
                           fields: dict[str, tuple[str, str]]
                           ) -> tuple[list[str], list[str]]:
    """Row `i`, each syllabus column on its own, asked only for its field."""
    got = {"can_do": [], "grammar": []}
    for side, (field, heading) in fields.items():
        what, how = next((w, h) for f, w, h in FIELDS.values() if f == field)
        raw = await llm.complete_json(
            [{"role": "user", "content": [
                {"type": "text", "text": COLUMN_ROW.format(
                    heading=heading, what=what, how=how)},
                *rows.column(i, side)]}],
            temperature=0.0, max_tokens=800)
        got[field] = _strings((raw or {}).get("items"))
    return got["can_do"], got["grammar"]


def _text_sim(a: list[str], b: list[str]) -> float:
    """How alike two lists read as running text.

    Where one reading splits a long bullet at its line break and the other
    does not — «temporale Nebensätze mit bis, seit(dem), | bevor, während …»
    — the items differ but the text is the same. A row shifted onto the
    wrong chapter differs in its text, so this still catches that.
    """
    if not a and not b:
        return 1.0
    return _sim(" ".join(a), " ".join(b))


def _row_agrees(x, y) -> bool:
    return _text_sim(x[0], y[0]) >= AGREE and _text_sim(x[1], y[1]) >= AGREE


def _closest(item: str, items: list[str]) -> str | None:
    best = max(items, key=lambda j: _sim(item, j), default=None)
    return best if best is not None and _sim(item, best) >= AGREE else None


def _vote(primary, second, third) -> tuple[tuple[list[str], list[str]], list[str]]:
    """Each item of the primary reading, as at least two readings spell it.

    A scan has no text to check spelling against: «Kinderinnerungen» for
    «Kindheitserinnerungen» is a perfectly plausible reading of one blurred
    line. Three readings that slip independently rarely slip the same way.
    """
    out, unsure = [], []
    for field, (mine, b, c) in enumerate(zip(primary, second, third)):
        kept = []
        for item in mine:
            # Backed as it stands if another reading has the same words, even
            # split differently at a line break or run into its neighbour.
            key = normalise(item)
            if key and any(key in normalise(" ".join(other)) for other in (b, c)):
                kept.append(item)
                continue
            m2, m3 = _closest(item, b), _closest(item, c)
            spellings = [s for s in (item, m2, m3) if s]
            votes = {}
            for s in spellings:
                votes.setdefault(normalise(s), []).append(s)
            best = max(votes.values(), key=len)
            if len(best) >= 2:
                kept.append(best[0])
            else:
                kept.append(item)
                unsure.append(item)
        out.append(kept)
    return (out[0], out[1]), unsure


def compare_chapters(a: list[curriculum.Chapter],
                     b: list[curriculum.Chapter]) -> list[str]:
    """Where two readings of the chapter list disagree."""
    if [c.number for c in a] != [c.number for c in b]:
        return [f"the chapter numbers differ between readings: "
                f"{[c.number for c in a]} vs {[c.number for c in b]}"]
    problems = []
    for x, y in zip(a, b):
        if _sim(x.title, y.title) < AGREE:
            problems.append(f"chapter {x.number}: title {x.title!r} vs {y.title!r}")
        if _list_sim([u.title for u in x.units], [u.title for u in y.units]) < AGREE:
            problems.append(f"chapter {x.number}: the sections differ")
        if _list_sim(x.can_do, y.can_do) < AGREE:
            problems.append(f"chapter {x.number}: the can-do statements differ")
    return problems


def compare_rows(a, b, numbers: list[int]) -> list[str]:
    """Where two readings of the syllabus disagree, row by row."""
    if len(a) != len(b):
        return [f"one reading found {len(a)} syllabus rows, the other {len(b)}"]
    problems = []
    for n, (x, y) in zip(numbers, zip(a, b)):
        if _text_sim(x[0], y[0]) < AGREE:
            problems.append(f"chapter {n}: the can-do statements differ")
        if _text_sim(x[1], y[1]) < AGREE:
            problems.append(f"chapter {n}: the grammar differs")
    return problems


def check_against_text(chapters: list[curriculum.Chapter], text: str) -> list[str]:
    """Every item must be in the page's text layer, where there is one."""
    if len(text.strip()) < 200:
        return []
    flat = " " + normalise(text.replace("\xad", "")) + " "
    problems = []
    for c in chapters:
        for item in c.can_do + c.grammar:
            if normalise(item) and normalise(item) not in flat:
                problems.append(f"chapter {c.number}: {item!r} is not on the page")
    return problems


async def read_contents(llm: LlmClient, doc: documents.Document,
                        chapters_page: int, syllabus_page: int,
                        progress=None
                        ) -> tuple[list[curriculum.Chapter], list[str]]:
    """Chapters with their can-do statements and grammar, checked, and the
    items whose spelling no two readings agreed on. Raises ReadError with
    every disagreement if the readings cannot be trusted."""
    def say(msg):
        if progress:
            progress(msg)

    readings = []
    for tile in (True, False):
        say(f"Reading the chapter list (page {chapters_page}, "
            f"{'in halves' if tile else 'whole'})")
        readings.append(_parse_chapters(await llm.complete_json(
            _ask(CHAPTERS, [doc.page(chapters_page)], tile),
            temperature=0.0, max_tokens=4000)))
    chapters, other = readings
    if not chapters:
        raise ReadError("no chapters could be read from the contents page")
    problems = compare_chapters(chapters, other)

    n = len(chapters)
    page = doc.page(syllabus_page)
    rows = Rows(page, n)
    if not rows.ok:
        raise ReadError(f"page {syllabus_page} does not divide into {n} rows by "
                        f"its rules, so it cannot be read one chapter at a time")

    say(f"Reading the column headings (page {syllabus_page})")
    fields = await rows.fields(llm)
    if not fields:
        raise ReadError(f"page {syllabus_page} has neither a SPRACHHANDLUNGEN nor "
                        f"a GRAMMATIK column")
    layout = _layout({side: f for side, (f, _) in fields.items()})

    # Each row is read twice on its own — across the full width, and column
    # by column — and the whole page once more. Counted item by item against
    # the A1 and A2 text layers, the column reading got 125 of 128 items
    # exactly right and the full-width strip 122, and they slip on different
    # items, which is what makes a vote between them worth anything.
    say(f"Reading the syllabus (page {syllabus_page}), one chapter row at a time")
    columns = [await read_row_columns(llm, rows, i, fields) for i in range(n)]
    strips = [await read_row(llm, rows.whole(i), layout) for i in range(n)]
    say("Checking it against a reading of the whole page")
    whole = _parse_rows(await llm.complete_json(
        _ask(SYLLABUS.format(n=n, layout=layout), [page], tile=False),
        temperature=0.0, max_tokens=4000))
    if len(whole) != n:
        whole = (whole + [([], [])] * n)[:n]

    # A row is kept when the column reading agrees with at least one other
    # on what the row holds; its items are then spelt as two readings agree.
    # An item no two readings spell alike is kept as the column reading has
    # it and named, for the learner to check against the book.
    uncertain: list[str] = []
    agreed: list[tuple[list[str], list[str]]] = []
    for i, chapter in enumerate(chapters):
        if not (_row_agrees(columns[i], strips[i])
                or _row_agrees(columns[i], whole[i])):
            problems += compare_rows([columns[i]], [strips[i]], [chapter.number])
            agreed.append(columns[i])
            continue
        row, unsure = _vote(columns[i], strips[i], whole[i])
        agreed.append(row)
        uncertain += [f"chapter {chapter.number}: {u!r}" for u in unsure]
    strips = agreed

    # Can-do statements come from whichever page prints them: the syllabus
    # page in A1-B1, the chapter list in B2.
    syllabus_has_can_do = any(f == "can_do" for f, _ in fields.values())
    for chapter, (can_do, grammar) in zip(chapters, strips):
        if syllabus_has_can_do:
            chapter.can_do = can_do
        chapter.grammar = grammar

    problems += check_against_text(chapters, page.text)
    if problems:
        raise ReadError("the readings of the contents pages do not agree:\n  - "
                        + "\n  - ".join(problems))

    # With a text layer, spelling was already checked against it exactly.
    if len(page.text.strip()) < 200:
        say("Checking the spelling")
        uncertain += await spelling_suspects(llm, chapters)
    return chapters, uncertain
