"""Detect mode-switch requests from what the learner said.

The model is *asked* to emit [[PRACTICE]] / [[MENTOR]] when the learner wants to
change gear, but it does not do so reliably — gemma3:12b will happily answer
"can we practise now?" by carrying on with the lesson. Since the learner's
intent is usually stated in plain words, we also check the transcript directly.

This runs on text we already have, so it costs nothing and cannot add latency.
Deliberately narrow: a missed switch is a minor annoyance, a false switch
yanks the learner out of a lesson mid-sentence.
"""

from __future__ import annotations

import re

from .prompts import MENTOR, PRACTICE

_WANT = r"(?:can we|could we|let'?s|lets|i want to|i'?d like to|shall we|"\
        r"how about we|ready to|time to)"

PRACTICE_PATTERNS = [
    rf"{_WANT}\s+(?:\w+\s+){{0,2}}(?:practi[sc]e|roleplay|role play)",
    rf"{_WANT}\s+(?:actually\s+)?(?:try|do)\s+it",
    rf"{_WANT}\s+(?:just\s+)?(?:talk|chat|speak)",
    r"\bstart (?:the )?(?:practi[sc]e|roleplay|conversation)\b",
    r"\b(?:lass uns|lass) (?:mal )?(?:üben|sprechen|reden)\b",
    r"\bich (?:möchte|will) (?:jetzt )?üben\b",
]

MENTOR_PATTERNS = [
    r"\b(?:i )?(?:don'?t|do not|can'?t) understand\b",
    r"\bwhat does (?:that|this|it|.{1,30}) mean\b",
    r"\bexplain (?:that|this|it|again|please)\b",
    r"\bin english\b",
    r"\b(?:stop|pause|quit) (?:the )?(?:practi[sc]e|roleplay)\b",
    rf"{_WANT}\s+(?:go )?back to (?:the )?(?:lesson|teaching|english)",
    r"\bteach me\b",
    r"\bich verstehe (?:das )?nicht\b",
]

# "Let me try saying that." Narrower than the mode patterns, because a false
# positive here interrupts the lesson to drill a phrase nobody asked about.
DRILL_PATTERNS = [
    r"\b(?:let me|can i|i want to|i'?d like to)\s+(?:just\s+)?"
    r"(?:try|say|repeat|practi[sc]e)\s+(?:that|it|this|again)\b",
    r"\b(?:say|repeat)\s+(?:that|it)\s+again\b",
    r"\b(?:one|once)\s+more\s+time\b",
    r"\bhow do i say (?:that|it)\b",
    r"\blet me repeat\b",
    r"\bdrill (?:that|it|me)\b",
    r"\bnoch (?:ein)?mal\b",
]

_PRACTICE_RE = [re.compile(p, re.IGNORECASE) for p in PRACTICE_PATTERNS]
_MENTOR_RE = [re.compile(p, re.IGNORECASE) for p in MENTOR_PATTERNS]
_DRILL_RE = [re.compile(p, re.IGNORECASE) for p in DRILL_PATTERNS]


def wants_drill(text: str) -> bool:
    """Did the learner ask to say the last phrase back?"""
    return any(r.search(text) for r in _DRILL_RE)


def requested_mode(text: str, current_mode: str) -> str | None:
    """Return the mode the learner is asking for, or None.

    Only ever returns a mode different from `current_mode`.
    """
    if not text:
        return None

    if current_mode != PRACTICE and any(r.search(text) for r in _PRACTICE_RE):
        return PRACTICE
    if current_mode != MENTOR and any(r.search(text) for r in _MENTOR_RE):
        return MENTOR
    return None
