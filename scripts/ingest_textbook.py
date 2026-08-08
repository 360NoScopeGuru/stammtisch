"""Turn a German textbook PDF into the course the tutor teaches from.

    python scripts/ingest_textbook.py "path/to/Akademie Deutsch A1+ Band 1.pdf"
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

Output goes to `paths.courses_dir` (default `./courses`), which is gitignored:
this content is copyrighted and the repository is public.
"""

from __future__ import annotations

import argparse
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
CHAPTER_LINE = re.compile(
    r"^(?:(\d{1,3})\s+)?([1-9])\s+([^a-zäöü]{3,}?)\s*$"
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
        m = re.search(r"(\d)\s*" + re.escape(UEBERSICHT), text)
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
        if not c.units:
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

    try:
        data = build(pdf, args.level)
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
    missing = [c.number for c in course.chapters if not c.reference]
    print(f"  written to {out}")
    if missing:
        print(f"  note: no reference section found for chapter(s) {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
