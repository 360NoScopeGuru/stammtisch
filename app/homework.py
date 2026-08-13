"""Assign written work, then mark it.

Talking is not studying. A tutor that only ever holds a conversation leaves
nothing to do between sessions and never makes the learner produce German
under their own steam, which is where the gaps actually show.

Two kinds of exercise, and the distinction is the whole design:

  **closed**  — a gap to fill, a verb to conjugate, an article to choose.
                There is one right answer and it is known when the exercise is
                written, so marking is a string comparison. No model involved.

  **open**    — translate this, write three sentences about your weekend.
                No answer key can exist, so a model has to judge it.

Everything that *can* be closed should be, because model marking is the weak
link: it is inconsistent between runs, it invents faults to look useful, and a
learner cannot tell a wrong mark from a right one. The workbook that ships with
the textbook has no answer key either — the `Lösungen` in it are the word
"solutions" inside exercise instructions — so there is no shortcut here.

Marks are advisory. This is practice, not an exam, and nothing here should
imply otherwise.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

CLOSED, OPEN = "closed", "open"

# A closed answer is marked right if it matches after this much forgiveness.
# Case, spacing, punctuation and umlaut spelling are not what is being tested.
_FOLD = str.maketrans({"ß": "ss", "ä": "a", "ö": "o", "ü": "u"})


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text.strip().lower()).translate(_FOLD)
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if not unicodedata.combining(c)
    )
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains(haystack: list[str], needle: list[str]) -> bool:
    """Does `needle` appear as a run of whole words in `haystack`?

    Whole words matter: the answer «ist» must not be found inside «Christian».
    """
    if not needle or len(needle) > len(haystack):
        return False
    return any(haystack[i:i + len(needle)] == needle
               for i in range(len(haystack) - len(needle) + 1))


@dataclass
class Exercise:
    kind: str                       # CLOSED | OPEN
    prompt: str
    answer: str = ""                # the expected answer, closed only
    alternatives: list[str] = field(default_factory=list)
    hint: str = ""

    def check(self, given: str) -> tuple[bool | None, str]:
        """Mark a closed exercise. Returns (correct, note).

        `None` means "not mechanically checkable" — an open exercise, or a
        closed one whose author forgot the answer. Never guess.
        """
        if self.kind != CLOSED or not self.answer.strip():
            return None, ""
        if not given.strip():
            return False, "left blank"

        want = [normalise(w) for w in (self.answer, *self.alternatives) if w]
        got = normalise(given)

        if got in want:
            return True, ""

        # Answering a gap-fill with the whole sentence, or a multiple choice
        # with "C) Guten Tag!" when the key is "C", is not a mistake — it is a
        # person answering a question. Marking those wrong is the failure that
        # matters here: the learner cannot tell an unfair mark from a real one,
        # and one unfair mark teaches them to distrust every other.
        tokens = got.split()
        for w in want:
            expected = w.split()
            if _contains(tokens, expected):
                return True, ""

        # Some of the right words, but not the answer.
        if any(w in got or got in w for w in want):
            return False, f"close — the answer is «{self.answer}»"
        return False, f"the answer is «{self.answer}»"


@dataclass
class Answer:
    index: int
    given: str = ""
    correct: bool | None = None
    note: str = ""
    marked_by: str = ""             # "rules" | "model" | ""


@dataclass
class Assignment:
    id: str
    created: str
    chapter: int | None
    chapter_title: str
    level: str
    exercises: list[Exercise] = field(default_factory=list)
    answers: list[Answer] = field(default_factory=list)
    marked: str = ""                # timestamp, empty until marked
    comment: str = ""               # one line of overall feedback

    # ---- state ---------------------------------------------------------

    @property
    def submitted(self) -> bool:
        return any(a.given.strip() for a in self.answers)

    @property
    def is_marked(self) -> bool:
        return bool(self.marked)

    @property
    def score(self) -> tuple[int, int]:
        """(right, markable). Unmarkable answers are excluded, not counted wrong."""
        judged = [a for a in self.answers if a.correct is not None]
        return sum(1 for a in judged if a.correct), len(judged)

    def answer_for(self, index: int) -> Answer:
        for a in self.answers:
            if a.index == index:
                return a
        a = Answer(index=index)
        self.answers.append(a)
        return a

    def submit(self, given: list[str]) -> None:
        for i, text in enumerate(given):
            self.answer_for(i).given = text

    # ---- marking -------------------------------------------------------

    def mark_closed(self) -> list[int]:
        """Mark everything that can be marked without a model.

        Returns the indices still needing judgement, so the caller knows
        exactly how little to send to the LLM.
        """
        remaining = []
        for i, ex in enumerate(self.exercises):
            ans = self.answer_for(i)
            correct, note = ex.check(ans.given)
            if correct is None:
                remaining.append(i)
                continue
            ans.correct, ans.note, ans.marked_by = correct, note, "rules"
        return remaining

    def apply_model_marks(self, marks: list[dict],
                          expected: list[int] | None = None) -> None:
        """Fold in the model's verdicts for the open exercises.

        Defensive on purpose: this is untrusted output. A verdict for an
        exercise the rules already marked is dropped rather than allowed to
        overwrite a certain answer with a guessed one.

        Models renumber. Asked to mark exercise 4 it will happily return index
        0 or 5, and an answer silently left unmarked looks exactly like an
        answer the marker ignored. So when the indices do not line up but the
        *count* does, they are matched positionally instead.
        """
        marks = [m for m in marks if isinstance(m, dict)]

        def as_index(m: dict) -> int | None:
            try:
                i = int(m.get("index", -1))
            except (TypeError, ValueError):
                return None
            return i if 0 <= i < len(self.exercises) else None

        pairs: list[tuple[int, dict]] = []
        if expected is not None and len(marks) == len(expected) and any(
            as_index(m) not in expected for m in marks
        ):
            pairs = list(zip(expected, marks))
        else:
            pairs = [(i, m) for m in marks if (i := as_index(m)) is not None]

        for i, m in pairs:
            if self.exercises[i].kind == CLOSED:
                continue          # the rules already decided, and they are right
            ans = self.answer_for(i)
            ans.correct = bool(m.get("correct"))
            ans.note = str(m.get("note", ""))[:400]
            ans.marked_by = "model"
        self.marked = datetime.now().isoformat(timespec="seconds")

    # ---- rendering -----------------------------------------------------

    def as_text(self, show_answers: bool = False) -> str:
        lines = [
            "",
            "─" * 58,
            f"  Hausaufgabe — {self.chapter_title or self.level}",
            "─" * 58,
        ]
        for i, ex in enumerate(self.exercises):
            lines.append(f"  {i + 1}. {ex.prompt}")
            if ex.hint:
                lines.append(f"     ({ex.hint})")
            ans = next((a for a in self.answers if a.index == i), None)
            if ans and ans.given:
                lines.append(f"     → {ans.given}")
            if ans and ans.correct is not None:
                mark = "✓" if ans.correct else "✗"
                lines.append(f"     {mark} {ans.note}".rstrip())
            elif show_answers and ex.answer:
                lines.append(f"     = {ex.answer}")
            lines.append("")

        if self.is_marked:
            right, total = self.score
            lines.append(f"  {right} of {total} marked correct")
            if self.comment:
                lines.append(f"  {self.comment}")
        lines.append("─" * 58)
        return "\n".join(lines)


# --- persistence ---------------------------------------------------------

def _to_dict(a: Assignment) -> dict:
    return asdict(a)


def _from_dict(d: dict) -> Assignment:
    return Assignment(
        id=str(d.get("id", "")),
        created=str(d.get("created", "")),
        chapter=d.get("chapter"),
        chapter_title=str(d.get("chapter_title", "")),
        level=str(d.get("level", "A1")),
        exercises=[
            Exercise(
                kind=str(e.get("kind", OPEN)),
                prompt=str(e.get("prompt", "")),
                answer=str(e.get("answer", "")),
                alternatives=[str(x) for x in e.get("alternatives", [])],
                hint=str(e.get("hint", "")),
            )
            for e in d.get("exercises", [])
        ],
        answers=[
            Answer(
                index=int(a.get("index", 0)),
                given=str(a.get("given", "")),
                correct=a.get("correct"),
                note=str(a.get("note", "")),
                marked_by=str(a.get("marked_by", "")),
            )
            for a in d.get("answers", [])
        ],
        marked=str(d.get("marked", "")),
        comment=str(d.get("comment", "")),
    )


def save(assignment: Assignment, directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{assignment.id}.json"
    # Atomic, like the session log: homework is written whenever an answer
    # changes, so a kill mid-write is a live possibility.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_to_dict(assignment), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_all(directory: Path) -> list[Assignment]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("*.json")):
        try:
            out.append(_from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            log.warning("skipping unreadable homework %s", path.name)
    return out


def latest(directory: Path) -> Assignment | None:
    items = load_all(directory)
    return items[-1] if items else None


def outstanding(directory: Path) -> list[Assignment]:
    """Assigned but not yet handed in."""
    return [a for a in load_all(directory) if not a.submitted]


def parse_generated(raw: dict, chapter: int | None, chapter_title: str,
                    level: str) -> Assignment:
    """Turn the model's JSON into an Assignment, discarding anything malformed.

    The model is asked for a specific shape and does not always produce it. An
    exercise with no prompt is useless; a closed exercise with no answer is
    worse than useless, because it looks markable and is not — so it is
    demoted to open rather than dropped.
    """
    exercises: list[Exercise] = []
    for item in raw.get("exercises", []) or []:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            continue
        kind = CLOSED if str(item.get("kind", "")).lower() == CLOSED else OPEN
        answer = str(item.get("answer", "")).strip()
        if kind == CLOSED and not answer:
            kind = OPEN
        alts = [str(x).strip() for x in (item.get("alternatives") or [])
                if str(x).strip()]
        exercises.append(Exercise(
            kind=kind, prompt=prompt, answer=answer,
            alternatives=alts, hint=str(item.get("hint", "")).strip(),
        ))

    now = datetime.now()
    return Assignment(
        id=now.strftime("%Y-%m-%d_%H%M%S"),
        created=now.isoformat(timespec="seconds"),
        chapter=chapter,
        chapter_title=chapter_title,
        level=level,
        exercises=exercises,
    )
