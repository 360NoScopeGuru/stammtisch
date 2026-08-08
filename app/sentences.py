"""Incremental sentence splitting for the LLM token stream.

This is the piece that buys us low latency: rather than waiting for the whole
reply, we hand each finished sentence to TTS the moment it closes. The tutor
starts talking while the model is still writing.

The hard part is not splitting on abbreviations — "Das kostet ca. 3 Euro" must
not break after "ca.".
"""

from __future__ import annotations

import re

# German abbreviations that end in a period but do not end a sentence.
ABBREVIATIONS = {
    "z.b", "d.h", "u.a", "u.s.w", "usw", "bzw", "ca", "evtl", "ggf", "inkl",
    "exkl", "vgl", "bspw", "sog", "etc", "nr", "abb", "bzgl", "einschl",
    "dr", "prof", "hr", "fr", "st", "jhd", "jh", "mio", "mrd", "tel",
    "z", "b", "d", "h", "u", "s", "o", "a", "m",  # single letters from split abbrevs
}

_TERMINATORS = ".!?…"
# A sentence ends at a terminator followed by whitespace or end-of-string.
_BOUNDARY = re.compile(rf"[{re.escape(_TERMINATORS)}]+[\"'»«)\]]*(?=\s|$)")
_TRAILING_WORD = re.compile(r"([\wäöüÄÖÜß.]+)\.$")


def _is_false_boundary(text: str) -> bool:
    """True if `text` ends at an abbreviation or a bare digit, not a sentence."""
    stripped = text.rstrip()
    if not stripped.endswith("."):
        return False  # ! and ? are never abbreviations

    m = _TRAILING_WORD.search(stripped)
    if not m:
        return False
    word = m.group(1).rstrip(".").lower()

    if word in ABBREVIATIONS:
        return True
    # Ordinals and dates: "am 3. Oktober", "im 19. Jahrhundert"
    if word.isdigit():
        return True
    return False


class SentenceStreamer:
    """Feed tokens in, get complete sentences out.

    >>> s = SentenceStreamer()
    >>> s.feed("Hallo! Wie geht")
    ['Hallo!']
    >>> s.feed(" es dir?")
    ['Wie geht es dir?']
    >>> s.flush()
    []
    """

    def __init__(self, min_chars: int = 2) -> None:
        self._buf = ""
        self._min_chars = min_chars

    def feed(self, chunk: str) -> list[str]:
        self._buf += chunk
        out: list[str] = []

        while True:
            match = None
            for m in _BOUNDARY.finditer(self._buf):
                candidate = self._buf[: m.end()]
                if _is_false_boundary(candidate):
                    continue
                match = m
                break

            if match is None:
                break

            sentence = self._buf[: match.end()].strip()
            self._buf = self._buf[match.end():].lstrip()
            if len(sentence) >= self._min_chars:
                out.append(sentence)

        return out

    def flush(self) -> list[str]:
        """Emit whatever is left. Call once the token stream ends."""
        rest = self._buf.strip()
        self._buf = ""
        return [rest] if len(rest) >= self._min_chars else []
