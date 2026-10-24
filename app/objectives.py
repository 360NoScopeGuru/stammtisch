"""Chapter goals, and whether the learner has shown they can do them.

Each chapter of the course book opens with its can-do statements — «Bestellungen
aufgeben», «über Vorlieben sprechen». They are the book's own definition of
what the chapter is for, which makes them the right test of whether it is
done. Without one, moving on is decided by feel, and feel is optimistic.

A goal is *demonstrated* by a check: the tutor sets a small task that needs
the goal ("You are in a café. Order a coffee and a piece of cake."), the
learner does it in German, and it is judged. Every check is appended to a log
and the status is rebuilt from the log, like progress is rebuilt from the
sessions — nothing is stored that could drift from what actually happened.

The judging is done by the model, which this project otherwise avoids for
anything that matters. Two things keep it honest:

- An answer with no German in it fails without being sent anywhere. That is
  the commonest wrong pass: a learner explaining in English what they *would*
  say.
- The learner can mark a goal done, or undone, by hand. The mark is advice,
  the learner is in charge, and a manual mark is recorded as one.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import langid

log = logging.getLogger(__name__)

LOG_NAME = "objectives.jsonl"
MANUAL, CHECK = "manual", "check"


@dataclass
class GoalStatus:
    index: int
    text: str
    demonstrated: bool = False
    attempts: int = 0
    last_note: str = ""
    last_at: str = ""
    by: str = ""


def _log_path(study_path: Path) -> Path:
    return Path(study_path) / LOG_NAME


def _read(study_path: Path) -> list[dict]:
    path = _log_path(study_path)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            # An append cut short by a kill. Everything before it is intact,
            # which is the point of an append-only log.
            log.warning("skipping a damaged line in %s", LOG_NAME)
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def record(study_path: Path, course: str, chapter: int, goal: str,
           passed: bool, note: str = "", by: str = CHECK,
           answer: str = "") -> dict:
    entry = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "course": course,
        "chapter": chapter,
        "goal": goal,
        "passed": bool(passed),
        "by": by,
        "note": note[:400],
        "answer": answer[:600],
    }
    path = _log_path(study_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def status(study_path: Path, course: str, chapter) -> list[GoalStatus]:
    """Every goal of `chapter`, with what the log says about it.

    The latest verdict wins, so a manual "not yet" undoes an earlier pass —
    the learner knows better than one judged answer whether it has stuck.
    """
    goals = [GoalStatus(index=i, text=t) for i, t in enumerate(chapter.can_do)]
    by_text = {g.text: g for g in goals}
    for item in _read(study_path):
        # Matched on chapter and goal, not on the course title: the title of
        # a course joined from several volumes changes each time one is
        # added ("A1–A2", then "A1–B1"), and a goal demonstrated last month
        # must not vanish because the next book was ingested.
        if item.get("chapter") != chapter.number:
            continue
        g = by_text.get(item.get("goal"))
        if g is None:
            continue
        if item.get("by") == CHECK:
            g.attempts += 1
        g.demonstrated = bool(item.get("passed"))
        g.last_note = str(item.get("note", ""))
        g.last_at = str(item.get("at", ""))
        g.by = str(item.get("by", ""))
    return goals


def chapter_done(goals: list[GoalStatus]) -> bool:
    return bool(goals) and all(g.demonstrated for g in goals)


def precheck(answer: str) -> tuple[bool, str] | None:
    """Fail an answer that cannot pass, without asking the model.

    Returns (False, reason) for a sure fail, or None to go on to judging.
    """
    text = (answer or "").strip()
    if len(text) < 3:
        return False, "There was no answer to judge."
    # Only a clear English answer fails here. «Einen Tee» contains none of
    # the function words langid counts, and failing it would be exactly the
    # unfair mark that teaches a learner to distrust every other one.
    de, en = langid.score(text)
    if en >= 2 and en > de * 2:
        return False, ("That was in English. Say or write it in German — that "
                       "is what the goal is about.")
    return None


def parse_verdict(raw: dict | None) -> tuple[bool, str] | None:
    """The model's {"passed", "note"}, or None if it did not give one.

    A missing verdict is not a fail: nothing is recorded, and the learner is
    told the check did not work rather than told they failed it.
    """
    if not isinstance(raw, dict) or "passed" not in raw:
        return None
    passed = raw.get("passed")
    if isinstance(passed, str):
        passed = passed.strip().lower() in ("true", "yes", "1")
    return bool(passed), str(raw.get("note", "")).strip()[:400]
