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

from .prompts import MENTOR, PRACTICE, TOKEN_MENTOR, TOKEN_PRACTICE

DE, EN = "de", "en"

_QUOTED = re.compile(r"«([^»]*)»")
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


def split_segments(text: str, default_lang: str = EN) -> list[tuple[str, str]]:
    """Split into [(lang, text), ...], preserving order.

    Text inside guillemets is German; everything else is `default_lang`.
    Segments that contain no speakable characters are dropped.
    """
    out: list[tuple[str, str]] = []
    pos = 0

    def add(lang: str, chunk: str) -> None:
        # A quote usually leaves orphaned punctuation behind it ("«…», which
        # means…"). Speaking a leading comma is noise, so trim it.
        chunk = chunk.strip().lstrip(",;:. ").strip()
        # Skip fragments that are only punctuation or whitespace.
        if chunk and re.search(r"[^\W_]", chunk, re.UNICODE):
            out.append((lang, chunk))

    for m in _QUOTED.finditer(text):
        add(default_lang, text[pos : m.start()])
        add(DE, m.group(1))
        pos = m.end()

    add(default_lang, text[pos:])
    return out
