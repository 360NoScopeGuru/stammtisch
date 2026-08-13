"""What the learner has already done, across every past session.

Without this the tutor is amnesiac. Every launch it re-introduces «Guten
Morgen», never notices that der/das has been wrong four sessions running, and
never picks up where the last lesson stopped. That is the single biggest thing
separating a tutor from a chatbot that happens to speak German.

Progress is **derived from the session files, not stored alongside them**. The
sessions are the record; anything kept in parallel would drift the first time
one was deleted or hand-edited. Rebuilding costs a few milliseconds for a few
hundred small JSON files, and it is self-healing.

The interesting part is `recurring_mistakes`. Grouping corrections by their
full sentence is useless — a learner rarely makes the same mistake in the same
words twice. Grouping by the *words that changed* turns twenty scattered
corrections into "der → das, six times", which is a lesson.
"""

from __future__ import annotations

import difflib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .corrections import tokens

log = logging.getLogger(__name__)

# A mistake has to happen more than once before it is worth the tutor's
# attention. Below this it is noise, or a transcription artefact.
RECURRING_THRESHOLD = 2

# How much of the learner's history to put in the system prompt. This competes
# directly with the conversation for the model's attention, so it is small on
# purpose.
BRIEF_MISTAKES = 4
BRIEF_VOCAB = 25


@dataclass
class Mistake:
    wrong: str
    right: str
    count: int
    example: str = ""
    explanation: str = ""

    @property
    def label(self) -> str:
        return f"«{self.wrong}» → «{self.right}»"


@dataclass
class Word:
    text: str
    seen: int = 0
    first: str = ""
    last: str = ""


@dataclass
class Progress:
    sessions: int = 0
    turns: int = 0
    minutes: float = 0.0
    last_seen: str = ""
    chapters: Counter = field(default_factory=Counter)
    last_chapter: int | None = None
    last_level: str = ""
    vocab: dict[str, Word] = field(default_factory=dict)
    mistakes: list[Mistake] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.sessions == 0

    def recurring(self, limit: int = BRIEF_MISTAKES) -> list[Mistake]:
        return [m for m in self.mistakes
                if m.count >= RECURRING_THRESHOLD][:limit]

    def due_vocab(self, limit: int = BRIEF_VOCAB) -> list[Word]:
        """Least recently practised first — the ones worth bringing back."""
        return sorted(self.vocab.values(), key=lambda w: (w.last, w.seen))[:limit]

    def brief(self) -> str:
        """A compact history for the system prompt, or "" on a first session.

        Written as facts rather than instructions to recite. A tutor told
        "they keep confusing der and das" should work that into a sentence,
        not read the learner their own error log.
        """
        if self.is_empty:
            return ""

        lines = [f"You have taught this learner before: {self.sessions} "
                 f"previous session(s), {self.turns} turns in total."]
        if self.last_chapter:
            lines.append(f"Last time you were on chapter {self.last_chapter}.")

        if recurring := self.recurring():
            lines.append(
                "Mistakes they make repeatedly: "
                + "; ".join(f"{m.label} ({m.count}x)" for m in recurring)
                + "."
            )

        if known := self.due_vocab():
            lines.append(
                "German they have met already, which you can use freely "
                "without re-teaching: "
                + ", ".join(w.text for w in known) + "."
            )

        lines.append(
            "Use this quietly. Work a recurring mistake back in when the "
            "conversation gives you a natural opening, and do not read this "
            "history out or open the session by summarising it."
        )
        return "\n".join(lines)

    def summary(self) -> str:
        """Human-readable, for `main.py --progress`."""
        if self.is_empty:
            return "\n  No sessions yet.\n"

        out = [
            "",
            "─" * 58,
            "  Fortschritt",
            "─" * 58,
            f"  {self.sessions} sessions · {self.turns} turns · "
            f"{self.minutes:.0f} min total",
        ]
        if self.last_seen:
            out.append(f"  Last session: {self.last_seen}"
                       + (f" (chapter {self.last_chapter})"
                          if self.last_chapter else ""))
        if self.chapters:
            worked = ", ".join(f"{n} ({c}x)"
                               for n, c in sorted(self.chapters.items()))
            out.append(f"  Chapters worked on: {worked}")

        if recurring := self.recurring(limit=10):
            out += ["", "  Keeps getting wrong:"]
            for m in recurring:
                out.append(f"    {m.label}   ({m.count}x)")
                if m.example:
                    out.append(f"      e.g. {m.example}")
        elif self.mistakes:
            out += ["", "  No mistake repeated yet — nothing has stuck as a habit."]

        out.append(f"\n  Vocabulary met: {len(self.vocab)} words")
        if due := self.due_vocab(12):
            out.append("  Least recently practised: "
                       + ", ".join(w.text for w in due))
        out.append("─" * 58)
        return "\n".join(out)


def _signature(original: str, corrected: str) -> tuple[str, str] | None:
    """The words that actually changed, as a (wrong, right) pair.

    A learner almost never repeats a mistake in the same sentence, so grouping
    whole corrections finds nothing. Grouping what changed inside them finds
    the habit: "der → das" across six different sentences is one lesson.
    """
    a, b = tokens(original), tokens(corrected)
    if not a or not b:
        return None

    wrong, right = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag == "equal":
            continue
        wrong.extend(a[i1:i2])
        right.extend(b[j1:j2])

    if not wrong and not right:
        return None
    # A whole sentence rewritten is not a pattern, it is a different sentence.
    if len(wrong) > 4 or len(right) > 4:
        return None
    return (" ".join(wrong).lower(), " ".join(right).lower())


def _load_sessions(sessions_dir: Path) -> list[dict]:
    out = []
    for path in sorted(Path(sessions_dir).glob("*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            # A session killed mid-write, from before writes were atomic.
            log.warning("skipping unreadable session %s", path.name)
    return out


def build(sessions_dir: Path) -> Progress:
    p = Progress()
    sessions = _load_sessions(sessions_dir)
    if not sessions:
        return p

    pairs: Counter = Counter()
    examples: dict[tuple[str, str], tuple[str, str]] = {}

    for data in sessions:
        transcript = data.get("transcript") or []
        if not transcript:
            continue

        p.sessions += 1
        p.turns += sum(1 for t in transcript if t.get("role") == "user")

        started, updated = data.get("started", ""), data.get("updated", "")
        if started and updated:
            try:
                p.minutes += (datetime.fromisoformat(updated)
                              - datetime.fromisoformat(started)).total_seconds() / 60
            except ValueError:
                pass

        when = (updated or started or "")[:10]
        if when >= p.last_seen:
            p.last_seen = when
            p.last_level = data.get("level", "") or p.last_level
            scenario = str(data.get("scenario", ""))
            if scenario.startswith("kapitel-"):
                try:
                    p.last_chapter = int(scenario.split("-", 1)[1])
                    p.chapters[p.last_chapter] += 1
                except ValueError:
                    pass

        for word in data.get("vocab") or []:
            key = str(word).strip().lower()
            if not key:
                continue
            entry = p.vocab.get(key) or Word(text=str(word).strip(), first=when)
            entry.seen += 1
            entry.last = max(entry.last, when)
            p.vocab[key] = entry

        for c in data.get("corrections") or []:
            sig = _signature(str(c.get("original", "")), str(c.get("corrected", "")))
            if not sig:
                continue
            pairs[sig] += 1
            examples.setdefault(
                sig, (str(c.get("original", "")), str(c.get("explanation", "")))
            )

    p.mistakes = [
        Mistake(wrong=wrong, right=right, count=count,
                example=examples.get((wrong, right), ("", ""))[0],
                explanation=examples.get((wrong, right), ("", ""))[1])
        for (wrong, right), count in pairs.most_common()
    ]
    return p
