"""The course the learner is actually taking.

Stammtisch's scenarios ("at the bakery", "office small talk") are fine for
conversation practice but they are not a syllabus: nothing says what comes
first, nothing says when something has been learned, and a beginner ends up
being taught whatever the model happens to think of. A real course has an
order, and the learner already has one — their textbook.

A `Course` is that book turned into an ordered list of chapters, each with the
things a syllabus actually specifies: what you will be able to *do* afterwards
(the CEFR can-do statements), which grammar it introduces, and the reference
material the book itself prints at the end of the chapter.

Courses are built by `scripts/ingest_textbook.py` and loaded from
`paths.courses_dir` as JSON. They are deliberately not committed: they are
derived from copyrighted textbooks, and this repository is public.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# The chapter reference section is a flattened grammar table and can run to a
# couple of thousand characters. Injected whole it would crowd out the actual
# conversation, so it is trimmed to something a tutor can hold in mind.
REFERENCE_BUDGET = 900


@dataclass
class Unit:
    number: int
    title: str


@dataclass
class Chapter:
    number: int
    title: str
    units: list[Unit] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    can_do: list[str] = field(default_factory=list)
    grammar: list[str] = field(default_factory=list)
    reference: str = ""
    book_page: int | None = None

    @property
    def key(self) -> str:
        return f"kapitel-{self.number}"

    def describe(self) -> str:
        """The chapter as a scenario description the tutor can teach from."""
        parts = [f"Chapter {self.number} of the learner's textbook: {self.title}."]
        if self.topics:
            parts.append("Topics: " + ", ".join(self.topics) + ".")
        if self.units:
            parts.append("Sections: "
                         + "; ".join(u.title for u in self.units) + ".")
        if self.can_do:
            parts.append(
                "By the end the learner should be able to: "
                + "; ".join(self.can_do) + "."
            )
        if self.grammar:
            parts.append("Grammar introduced here: " + ", ".join(self.grammar) + ".")
        return " ".join(parts)

    def reference_extract(self, budget: int = REFERENCE_BUDGET) -> str:
        if len(self.reference) <= budget:
            return self.reference
        return self.reference[:budget].rsplit("\n", 1)[0]


@dataclass
class Course:
    title: str
    level: str
    chapters: list[Chapter] = field(default_factory=list)
    source: str = ""

    def get(self, number: int) -> Chapter | None:
        return next((c for c in self.chapters if c.number == number), None)

    def first(self) -> Chapter | None:
        return self.chapters[0] if self.chapters else None

    def next_after(self, number: int) -> Chapter | None:
        return self.get(number + 1)

    def listing(self) -> str:
        return "\n".join(
            f"  {c.number:>2}. {c.title:<38} "
            f"{', '.join(g for g in c.grammar[:3])}"
            for c in self.chapters
        )


def title_case(text: str) -> str:
    """Chapter titles are printed in caps; this is spoken aloud.

    `str.title()` is not usable here — it treats an apostrophe as a word
    boundary and turns «LOS GEHT’S» into "Los Geht’S".
    """
    return " ".join(w[:1].upper() + w[1:].lower() for w in text.split(" "))


def _join_de(items: list[str]) -> str:
    """German list: 'Essen, Einkaufen und Lebensmittel'."""
    items = [i for i in items if i.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " und " + items[-1]


def _join_en(items: list[str]) -> str:
    items = [i for i in items if i.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def as_scenario(chapter: Chapter, level: str = "A1") -> "Scenario":
    """A textbook chapter dressed as a scenario.

    Everything downstream — the system prompt, the openers, the session log —
    already speaks `Scenario`, so a chapter that becomes one needs no special
    handling anywhere else.

    The two openers are built from the chapter's own words rather than asked
    for from the model. Openers are spoken before the learner has said
    anything, so a model that wanders here sets the wrong tone for the whole
    session, and the topic keywords are already German nouns.
    """
    from .scenarios import Scenario

    topics_de = _join_de(chapter.topics)
    # Left capitalised even inside English prose: they are German nouns, and
    # "it covers essen and einkaufen" is wrong in both languages.
    topics_en = _join_en(chapter.topics)

    title = title_case(chapter.title)
    intro = f"Today we are doing chapter {chapter.number} of your textbook"
    if title:
        # Chapter titles often end in their own punctuation ("Lecker!"), and
        # this line is spoken aloud, so a stray "Lecker!." matters.
        intro += f", {title}" + ("" if title[-1] in "!?." else ".") + " "
    else:
        intro += ". "
    if topics_en:
        intro += f"It covers {topics_en}. "
    if chapter.can_do:
        intro += (f"By the end you should be able to "
                  f"{_join_en(chapter.can_do[:3])}. ")
    intro += "Let us start with something small. Ready?"

    opener = "Hallo! "
    if topics_de:
        opener += f"Heute geht es um {topics_de}. "
    opener += "Fangen wir an!"

    return Scenario(
        key=chapter.key,
        title=f"Kapitel {chapter.number}: {title}",
        description=chapter.describe(),
        opener=opener,
        intro=intro,
        min_level=level,
    )


def _chapter_from_dict(d: dict) -> Chapter:
    return Chapter(
        number=int(d["number"]),
        title=str(d.get("title", "")),
        units=[Unit(int(u["number"]), str(u["title"]))
               for u in d.get("units", [])],
        topics=[str(t) for t in d.get("topics", [])],
        can_do=[str(t) for t in d.get("can_do", [])],
        grammar=[str(t) for t in d.get("grammar", [])],
        reference=str(d.get("reference", "")),
        book_page=d.get("book_page"),
    )


def load(path: Path) -> Course:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return Course(
        title=str(raw.get("title", path.stem)),
        level=str(raw.get("level", "A1")),
        source=str(raw.get("source", "")),
        chapters=[_chapter_from_dict(c) for c in raw.get("chapters", [])],
    )


def find(courses_dir: Path, name: str = "") -> Course | None:
    """Load a course by file stem, or the only one present if unnamed."""
    courses_dir = Path(courses_dir)
    if not courses_dir.is_dir():
        return None

    files = sorted(courses_dir.glob("*.json"))
    if not files:
        return None

    if name:
        for f in files:
            if f.stem == name:
                return load(f)
        log.warning("course %r not found in %s", name, courses_dir)
        return None

    if len(files) > 1:
        log.info("several courses present, using %s", files[0].name)
    return load(files[0])
