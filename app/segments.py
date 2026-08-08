"""Split mixed English/German text into per-language segments for TTS.

In mentor mode the model writes English prose with German wrapped in
guillemets:

    Try saying «Ich heiße Anna», which means "my name is Anna".

Feeding that whole string to the German voice mangles the English, and the
English voice mangles the German. So we split on the guillemets and send each
piece to the right voice.
"""

from __future__ import annotations

import re

from .langid import DE, EN, detect
from .prompts import MENTOR, PRACTICE, TOKEN_MENTOR, TOKEN_PRACTICE

_QUOTED = re.compile(r"«([^»]*)»")
_PARENS = re.compile(r"\(([^)]*)\)")


def _split_parens(kind: str, text: str) -> list[tuple[str, str]]:
    """Pull bracketed asides out of `text`, keeping everything in order."""
    out: list[tuple[str, str]] = []
    pos = 0
    for m in _PARENS.finditer(text):
        out.append((kind, text[pos : m.start()]))
        out.append(("paren", m.group(1)))
        pos = m.end()
    out.append((kind, text[pos:]))
    return out
_TOKENS = {TOKEN_PRACTICE: PRACTICE, TOKEN_MENTOR: MENTOR}
# Models sometimes wrap the token in punctuation or markdown.
_TOKEN_RE = re.compile(
    r"[\s\*_`\[\]()]*\[\[\s*(PRACTICE|MENTOR)\s*\]\][\s\*_`\[\]().]*",
    re.IGNORECASE,
)


def extract_mode(text: str) -> tuple[str, str | None]:
    """Pull any mode-switch token out of `text`.

    Returns the cleaned text and the requested mode (or None).
    """
    found: str | None = None

    def _take(m: re.Match) -> str:
        nonlocal found
        found = m.group(1).lower()
        return " "

    cleaned = _TOKEN_RE.sub(_take, text)
    return cleaned.strip(), found


def strip_markers(text: str) -> str:
    """Remove guillemets, leaving readable plain text."""
    return _QUOTED.sub(lambda m: m.group(1), text).strip()


# Stage directions the model sometimes emits ("(German voice)", "(in English)").
# They are instructions to itself, not speech, and must never be read aloud.
_STAGE_DIRECTION = re.compile(
    r"\((?:[^)]*\b(?:voice|speaking|accent|in german|in english|pause|slowly)\b"
    r"[^)]*)\)",
    re.IGNORECASE,
)


def clean_for_speech(text: str) -> str:
    return _STAGE_DIRECTION.sub(" ", text).strip()


def split_segments(text: str, default_lang: str = EN) -> list[tuple[str, str]]:
    """Split into [(lang, text), ...], preserving order.

    Guillemets are a hint, not the decision. Each fragment is classified by its
    own words, because models mark languages unreliably in both directions —
    see app/langid.py. The guillemets only break ties on fragments too short to
    classify ("Hallo!"), where being inside marks means German and outside
    means whatever the mode implies.
    """
    text = clean_for_speech(text)
    out: list[tuple[str, str]] = []

    # Guillemets and brackets both mark a change of voice. Brackets matter
    # because models gloss themselves in them even when told not to —
    # gemma3:4b produced «Guten Tag! Wie heißt du? (What is your name?)» in
    # practice mode, and without a split here the whole thing, English gloss
    # included, goes to the German voice and comes out as noise.
    # Brackets are split in a second pass rather than in one combined regex,
    # because the gloss usually sits *inside* the guillemets and a single
    # alternation would let the guillemet match swallow it whole.
    pieces: list[tuple[str, str]] = []   # (kind, text)
    pos = 0
    for m in _QUOTED.finditer(text):
        pieces += _split_parens("plain", text[pos : m.start()])
        pieces += _split_parens("quoted", m.group(1))
        pos = m.end()
    pieces += _split_parens("plain", text[pos:])

    # An aside in brackets is a gloss, so English breaks its ties; inside
    # guillemets German does; elsewhere the mode decides.
    tie_break = {"quoted": DE, "paren": EN, "plain": default_lang}

    def carry_punctuation(punct: str) -> None:
        """Attach orphaned punctuation to the previous segment.

        Quotes leave punctuation stranded between them ("«eins», «zwei»").
        Dropping it turns a list into a run-on; keeping it as its own segment
        would have the synthesiser read a bare comma. Appending preserves the
        pause where it belongs.
        """
        punct = punct.strip()
        if punct and out:
            out[-1] = (out[-1][0], f"{out[-1][1]}{punct}")

    for kind, chunk in pieces:
        chunk = chunk.strip()
        if not chunk:
            continue

        if not re.search(r"[^\W_]", chunk, re.UNICODE):
            carry_punctuation(chunk)
            continue

        if lead := re.match(r"^[,;:.!?…\s]+", chunk):
            carry_punctuation(lead.group())
            chunk = chunk[lead.end():].strip()
            if not chunk:
                continue

        out.append((detect(chunk, tie_break[kind]), chunk))

    return _merge_adjacent(out)


def _merge_adjacent(segs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Join neighbouring fragments in the same language.

    Fewer, longer TTS calls sound more natural than many short ones — Piper
    resets prosody at every call, so splitting a sentence makes it choppy.
    """
    merged: list[tuple[str, str]] = []
    for lang, chunk in segs:
        if merged and merged[-1][0] == lang:
            merged[-1] = (lang, f"{merged[-1][1]} {chunk}")
        else:
            merged.append((lang, chunk))
    return merged
