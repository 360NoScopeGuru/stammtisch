"""Turn a German textbook PDF into the course the tutor teaches from.

    python scripts/ingest_textbook.py "path/to/Akademie Deutsch A1+ Band 1.pdf"
    python scripts/ingest_textbook.py "path/to/B1 book.pdf" --level B1
    python scripts/ingest_textbook.py book.pdf --vision --contents 4-5
    python scripts/ingest_textbook.py --show          # print what was built

Why bother: the tutor without a syllabus teaches whatever the model thinks of
next, which for a beginner means a random walk through German. The learner
already has an order — their course book — and following it means what the app
teaches lines up with what their class is doing.

This targets the *Akademie Deutsch* layout, which prints a two-page contents
spread carrying everything a syllabus needs:

    page 1   chapter numbers, titles, section titles, topic keywords
    page 2   SPRACHHANDLUNGEN (can-do statements) and GRAMMATIK, per chapter

and then a "ÜBERSICHT GRAMMATIK UND REDEMITTEL" reference page at the end of
each chapter.

The parse is heuristic — it is a PDF — so it is checked structurally before
anything is written. A silently half-parsed syllabus would be worse than none,
because it would look like a course while quietly skipping half the grammar.
Every failure here is loud and refuses to write.

A book with no usable text layer — the B1 and B2 volumes are scans — is read
from images of its contents pages instead, by app/bookreader.py, and held to
the same standard: two independent readings that must agree, or nothing is
written. `--vision` forces that path; it needs the LLM server running.

Output goes to `paths.courses_dir` (default `./courses`), which is gitignored:
this content is copyrighted and the repository is public.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import curriculum  # noqa: E402
from app.config import load_config  # noqa: E402

BULLET = "•"
UEBERSICHT = "ÜBERSICHT GRAMMATIK UND REDEMITTEL"

# "26 2 DEUTSCHE SPRACHE, SCHWERE SPRACHE?" — optional book page, chapter
# number, then an all-caps title. Chapter one has no page number printed.
# Later volumes carry on the numbering — the A2 book opens on chapter 10 — so
# the chapter number can have two digits.
CHAPTER_LINE = re.compile(
    r"^(?:(\d{1,3})\s+)?([1-9]\d?)\s+([^a-zäöü]{3,}?)\s*$"
)
# "1 Freut mich!" — same shape, but the title is not shouting.
UNIT_LINE = re.compile(r"^([1-9])\s+(\S.*[a-zäöü].*)$")

SKIP = re.compile(r"^(INHALT|THEMENKAPITEL|SEITE|h\d+h\s|\d+\s*$)", re.IGNORECASE)


class IngestError(RuntimeError):
    pass


def page_texts(pdf: Path) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise IngestError(
            "pypdf is not installed — pip install pypdf"
        ) from None
    reader = PdfReader(str(pdf))
    return [(p.extract_text() or "") for p in reader.pages]


# --- the contents spread -------------------------------------------------

def find_contents(pages: list[str]) -> tuple[int, int]:
    """Locate the two contents pages: the chapter list, then the syllabus."""
    listing = syllabus = -1
    for i, text in enumerate(pages[:12]):
        if listing < 0 and "THEMENKAPITEL" in text:
            listing = i
        if syllabus < 0 and "SPRACHHANDLUNGEN" in text:
            syllabus = i
    if listing < 0 or syllabus < 0:
        raise IngestError(
            "could not find the contents pages (looked for THEMENKAPITEL and "
            "SPRACHHANDLUNGEN in the first 12 pages). This script only "
            "understands the Akademie Deutsch layout."
        )
    return listing, syllabus


def parse_listing(text: str) -> list[curriculum.Chapter]:
    """Chapter numbers, titles, section titles and topic keywords."""
    chapters: list[curriculum.Chapter] = []
    for raw in text.split("\n"):
        line = raw.rstrip()
        if not line.strip() or SKIP.match(line.strip()):
            continue

        # Topic keywords are indented and separated by runs of spaces.
        if raw.startswith(" ") and chapters:
            chapters[-1].topics.extend(
                t.strip() for t in re.split(r"\s{2,}", line.strip()) if t.strip()
            )
            continue

        if m := CHAPTER_LINE.match(line.strip()):
            page, number, title = m.groups()
            chapters.append(curriculum.Chapter(
                number=int(number),
                title=title.strip(),
                book_page=int(page) if page else None,
            ))
            continue

        if (m := UNIT_LINE.match(line.strip())) and chapters:
            number, title = m.groups()
            chapters[-1].units.append(
                curriculum.Unit(int(number), title.strip())
            )
    return chapters


def parse_syllabus(text: str) -> list[tuple[list[str], list[str]]]:
    """Alternating can-do blocks and grammar bullet lists, in chapter order.

    A bullet that wraps onto a second line is indented, which is the only
    thing distinguishing it from the next chapter's can-do statements.
    """
    blocks: list[tuple[list[str], list[str]]] = []
    pending: list[str] = []          # can-do lines for the chapter in progress
    grammar: list[str] = []

    for raw in text.split("\n"):
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("SPRACHHANDLUNGEN"):
            continue
        if SKIP.match(stripped) or stripped.startswith("Impressum"):
            continue

        if stripped.startswith(BULLET):
            grammar.append(stripped.lstrip(BULLET).strip())
        elif raw.startswith(" ") and grammar:
            # Indentation is the only thing separating a wrapped bullet from
            # the next chapter's opening line.
            grammar[-1] = f"{grammar[-1]} {stripped}".strip()
        else:
            # A plain line after a run of bullets means the previous chapter
            # is finished. Closing the block here rather than at the next
            # bullet is the whole difference between the grammar lining up
            # with its chapter and being shifted one along.
            if grammar:
                blocks.append((_split_can_do(pending), grammar))
                pending, grammar = [], []
            pending.append(stripped)

    if pending or grammar:
        blocks.append((_split_can_do(pending), grammar))
    return blocks


def _split_can_do(lines: list[str]) -> list[str]:
    joined = " ".join(lines)
    return [p.strip() for p in joined.split("|") if p.strip()]


# --- chapter reference pages ---------------------------------------------

def collect_references(pages: list[str], chapters: list[curriculum.Chapter]) -> int:
    """Attach each chapter's end-of-chapter grammar summary."""
    found = 0
    for text in pages:
        if UEBERSICHT not in text:
            continue
        # The chapter number opens the heading's line: "6 ÜBERSICHT…", or
        # "10ÜBERSICHT…" in the A2 book. Anchored to the line start, because
        # the running header just above runs page and chapter together
        # ("KAPITEL   6126") and an unanchored match reads that as 26.
        m = re.search(r"(?:^|\n)(\d{1,2})\s*" + re.escape(UEBERSICHT), text)
        if not m:
            continue
        number = int(m.group(1))
        chapter = next((c for c in chapters if c.number == number), None)
        if chapter is None:
            continue
        body = text.split(UEBERSICHT, 1)[1].strip()
        # Running headers ("KAPITEL 6126") are noise in a reference section.
        body = re.sub(r"^KAPITEL\s+\d+.*$", "", body, flags=re.MULTILINE)
        chapter.reference = (chapter.reference + "\n" + body).strip()
        found += 1
    return found


# --- validation ----------------------------------------------------------

def validate(chapters: list[curriculum.Chapter],
             blocks: list[tuple[list[str], list[str]]],
             syllabus_text: str = "", min_chapters: int = 5) -> list[str]:
    """Everything that would make this course quietly wrong."""
    problems = []

    # The block count matching the chapter count is not enough on its own: an
    # earlier version of this parser produced exactly nine blocks with every
    # grammar list shifted one chapter along, because it closed a block at the
    # wrong moment. Each can-do statement has to be text that genuinely appears
    # in the page, which a statement stitched together across two chapters does
    # not.
    if syllabus_text:
        flat = re.sub(r"\s+", " ", syllabus_text)
        for i, (can_do, _) in enumerate(blocks, start=1):
            for phrase in can_do:
                if re.sub(r"\s+", " ", phrase) not in flat:
                    problems.append(
                        f"block {i}: {phrase!r} is not contiguous text in the "
                        f"contents page — the blocks are misaligned"
                    )
                    break

    if len(chapters) < min_chapters:
        problems.append(f"only {len(chapters)} chapters parsed — expected the "
                        f"whole book")

    numbers = [c.number for c in chapters]
    if numbers != sorted(numbers) or len(set(numbers)) != len(numbers):
        problems.append(f"chapter numbers are not a clean sequence: {numbers}")

    if len(blocks) != len(chapters):
        problems.append(
            f"{len(chapters)} chapters in the contents list but {len(blocks)} "
            f"syllabus blocks — they cannot be matched up reliably"
        )

    for c in chapters:
        if not c.title:
            problems.append(f"chapter {c.number} has no title")
        # The B2 volume prints no sections in its contents at all; a book
        # that lists them for some chapters and not others is misread.
        if not c.units and any(x.units for x in chapters):
            problems.append(f"chapter {c.number} ({c.title}) has no sections")

    return problems


def build(pdf: Path, level: str) -> dict:
    pages = page_texts(pdf)
    listing_idx, syllabus_idx = find_contents(pages)

    chapters = parse_listing(pages[listing_idx])
    blocks = parse_syllabus(pages[syllabus_idx])

    if problems := validate(chapters, blocks, pages[syllabus_idx]):
        raise IngestError(
            "the contents pages did not parse cleanly:\n  - "
            + "\n  - ".join(problems)
        )

    for chapter, (can_do, grammar) in zip(chapters, blocks):
        chapter.can_do = can_do
        chapter.grammar = grammar

    refs = collect_references(pages, chapters)

    return {
        "title": pdf.stem,
        "level": level,
        "source": pdf.name,
        "ingested_at": datetime.now().isoformat(timespec="seconds"),
        "pages": len(pages),
        "references_found": refs,
        "chapters": [
            {
                "number": c.number,
                "title": c.title,
                "book_page": c.book_page,
                "units": [{"number": u.number, "title": u.title} for u in c.units],
                "topics": c.topics,
                "can_do": c.can_do,
                "grammar": c.grammar,
                "reference": c.reference,
            }
            for c in chapters
        ],
    }


async def build_vision(pdf: Path, level: str, contents: str = "") -> dict:
    """The contents pages read as images. See app/bookreader.py."""
    from app import bookreader, documents
    from app.llm import LlmClient

    cfg = load_config()
    llm = LlmClient(cfg)
    ok, msg = await llm.health()
    if not ok:
        await llm.aclose()
        raise IngestError(f"reading a scanned book needs the LLM: {msg}")

    def say(m):
        print(f"  … {m}", flush=True)

    try:
        doc = documents.load_path(pdf)
        if contents:
            numbers = documents.parse_pages(contents, doc.page_count)
            chapters_page = numbers[0]
            syllabus_page = numbers[1] if len(numbers) > 1 else numbers[0] + 1
        else:
            chapters_page, syllabus_page = await bookreader.find_contents(
                llm, doc, progress=say)
        chapters, uncertain = await bookreader.read_contents(
            llm, doc, chapters_page, syllabus_page, progress=say)
    except (bookreader.ReadError, documents.DocumentError) as e:
        raise IngestError(str(e)) from e
    finally:
        await llm.aclose()

    problems = validate(chapters, [(c.can_do, c.grammar) for c in chapters])
    # A chapter the readings agree is empty is still not a chapter of this
    # book: every volume gives each one its goals and its grammar.
    problems += [f"chapter {c.number} has no {what}" for c in chapters
                 for what, items in (("can-do statements", c.can_do),
                                     ("grammar", c.grammar)) if not items]
    if problems:
        raise IngestError("the contents did not hold together:\n  - "
                          + "\n  - ".join(problems))

    # A text layer, if there is one, still supplies the chapter summaries.
    refs = 0
    try:
        pages = page_texts(pdf)
        if sum(len(t) for t in pages) > 1000:
            refs = collect_references(pages, chapters)
    except Exception:
        pages = []

    return {
        "title": pdf.stem,
        "level": level,
        "source": pdf.name,
        "read_by": "vision",
        # Spelling no two readings agreed on: printed after ingesting, for
        # the learner to compare with the book.
        "uncertain": uncertain,
        "contents_pages": [chapters_page, syllabus_page],
        "ingested_at": datetime.now().isoformat(timespec="seconds"),
        "pages": doc.page_count,
        "references_found": refs,
        "chapters": [
            {
                "number": c.number,
                "title": c.title,
                "book_page": c.book_page,
                "units": [{"number": u.number, "title": u.title} for u in c.units],
                "topics": c.topics,
                "can_do": c.can_do,
                "grammar": c.grammar,
                "reference": c.reference,
            }
            for c in chapters
        ],
    }


def show(course: curriculum.Course) -> None:
    print(f"\n  {course.title}   ({course.level}, "
          f"{len(course.chapters)} chapters)\n")
    for c in course.chapters:
        print(f"  {c.number}. {c.title}")
        for u in c.units:
            print(f"       {u.number}. {u.title}")
        if c.topics:
            print(f"     topics : {', '.join(c.topics)}")
        if c.can_do:
            print(f"     can-do : {c.can_do[0]}"
                  + (f" (+{len(c.can_do) - 1} more)" if len(c.can_do) > 1 else ""))
        if c.grammar:
            print(f"     grammar: {', '.join(c.grammar)}")
        print(f"     reference: {len(c.reference)} chars")
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", nargs="?", help="the textbook PDF")
    ap.add_argument("--level", default="A1")
    ap.add_argument("--name", default="", help="output file stem")
    ap.add_argument("--show", action="store_true",
                    help="print the course that is already ingested")
    ap.add_argument("--vision", action="store_true",
                    help="read the contents pages as images (scanned books)")
    ap.add_argument("--contents", default="",
                    help="the contents pages, e.g. 4-5, if they cannot be found")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = cfg.courses_path

    if args.show or not args.pdf:
        course = curriculum.find(out_dir, args.name)
        if not course:
            print(f"  no course in {out_dir} — run with a PDF path first")
            return 1
        show(course)
        return 0

    pdf = Path(args.pdf)
    if not pdf.is_file():
        print(f"  no such file: {pdf}")
        return 1

    data = None
    if not args.vision:
        try:
            data = build(pdf, args.level)
        except IngestError as e:
            print(f"\n  the text layer did not parse: {str(e).splitlines()[0]}")
            print("  reading the contents pages as images instead…\n")
    if data is None:
        try:
            data = asyncio.run(build_vision(pdf, args.level, args.contents))
        except IngestError as e:
            print(f"\n  could not ingest {pdf.name}:\n  {e}\n")
            return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or re.sub(r"[^a-z0-9]+", "-", pdf.stem.lower()).strip("-")
    out = out_dir / f"{stem}.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    course = curriculum.load(out)
    show(course)
    if data.get("uncertain"):
        print("  These may be misread: the readings disagreed on them, or they do")
        print("  not look like German. Check them against the book, and fix them")
        print("  in the file below if they are wrong:")
        for item in data["uncertain"]:
            print(f"    {item}")
        print()
    missing = [c.number for c in course.chapters if not c.reference]
    print(f"  written to {out}")
    if missing and data.get("read_by") == "vision":
        print("  note: a scanned book's end-of-chapter grammar summaries are not "
              "read")
    elif missing:
        print(f"  note: no reference section found for chapter(s) {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
