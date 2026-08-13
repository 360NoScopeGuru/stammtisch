"""Repeat-after-me: say a phrase back and find out which words landed.

**This is not pronunciation scoring, and it must not be sold as one.** Whisper
returns text, not phonemes. It cannot tell you your vowel was too far forward.
Real scoring needs forced alignment and a goodness-of-pronunciation model, and
that means torch, which this project deliberately does not have.

What it *can* tell you is whether a German speech recogniser, listening to you
in German, heard the words you were aiming at. That is a weaker claim and a
genuinely useful one: if Whisper hears «Brötchen» when you say it, you said
something close enough to be understood. If it hears «Brot chen» or nothing at
all, you did not. For a beginner that is most of the value, and it is honest.

The failure mode to design against is a false negative — telling someone their
German was wrong when the recogniser simply mangled it. So orthographic
variants that no listener would count as errors are normalised away before
comparison: «heisse» and «heiße» are the same word, and so are «Brotchen» and
«Brötchen».
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field

# Above this the attempt counts as correct.
GOOD = 0.85
# Below GOOD but above this, it was close — worth one more go rather than a
# re-explanation.
CLOSE = 0.55

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

# Whisper spells German inconsistently, and a learner is not wrong because the
# recogniser chose "ss" over "ß".
_FOLD = str.maketrans({"ß": "ss", "ä": "a", "ö": "o", "ü": "u"})


def normalise(word: str) -> str:
    """Fold a word to the form used for comparison."""
    word = unicodedata.normalize("NFC", word.lower()).translate(_FOLD)
    # Strip any remaining combining marks, so "ó" from a bad decode matches "o".
    return "".join(
        c for c in unicodedata.normalize("NFD", word)
        if not unicodedata.combining(c)
    )


def words(text: str) -> list[str]:
    return _WORD.findall(text)


@dataclass
class WordResult:
    target: str            # what they were asked to say ("" if extra)
    heard: str             # what came back ("" if missing)
    status: str            # "correct" | "close" | "wrong" | "missing" | "extra"

    @property
    def ok(self) -> bool:
        return self.status in ("correct", "close")


@dataclass
class Attempt:
    target: str
    heard: str
    score: float
    results: list[WordResult] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.score >= GOOD:
            return "good"
        if self.score >= CLOSE:
            return "close"
        return "again"

    @property
    def missed(self) -> list[str]:
        """The target words that did not come through."""
        return [r.target for r in self.results if r.target and not r.ok]

    def feedback(self) -> str:
        """One English sentence, spoken back to the learner.

        Deliberately concrete. "Not quite, try again" teaches nothing; naming
        the word that failed tells them where to aim.
        """
        if not words(self.heard):
            return "I did not catch that at all. Have another go, a bit louder."

        if self.verdict == "good":
            return "That was good."

        missed = self.missed
        if not missed:
            # Score dragged down by extra words rather than wrong ones.
            return ("Nearly — you added a little extra. Try saying just the "
                    "phrase on its own.")

        if len(missed) == 1:
            return (f"Close. The word «{missed[0]}» did not quite come "
                    f"through — try that one again.")
        listed = ", ".join(f"«{w}»" for w in missed[:3])
        if self.verdict == "close":
            return f"Close. Watch {listed} — say it once more."
        return f"Not quite yet. The tricky parts are {listed}. Listen again."


def _word_status(target: str, heard: str) -> str:
    if normalise(target) == normalise(heard):
        return "correct"
    # A near miss at the character level is the recogniser hearing you
    # imperfectly rather than you saying the wrong word.
    ratio = difflib.SequenceMatcher(
        None, normalise(target), normalise(heard)
    ).ratio()
    return "close" if ratio >= 0.8 else "wrong"


def compare(target: str, heard: str) -> Attempt:
    """Line up what was said against what was asked for, word by word."""
    a, b = words(target), words(heard)
    fa = [normalise(w) for w in a]
    fb = [normalise(w) for w in b]

    results: list[WordResult] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, fa, fb).get_opcodes():
        if tag == "equal":
            results += [WordResult(a[i], b[j], "correct")
                        for i, j in zip(range(i1, i2), range(j1, j2))]
        elif tag == "replace":
            # Pair them up as far as they go; the remainder is missing or extra.
            for i, j in zip(range(i1, i2), range(j1, j2)):
                results.append(WordResult(a[i], b[j], _word_status(a[i], b[j])))
            for i in range(i1 + (j2 - j1), i2):
                results.append(WordResult(a[i], "", "missing"))
            for j in range(j1 + (i2 - i1), j2):
                results.append(WordResult("", b[j], "extra"))
        elif tag == "delete":
            results += [WordResult(a[i], "", "missing") for i in range(i1, i2)]
        elif tag == "insert":
            results += [WordResult("", b[j], "extra") for j in range(j1, j2)]

    if not a:
        return Attempt(target, heard, 0.0, results)

    # Scored against the target, with extra words penalised but not fatally —
    # a learner who says the phrase and then adds "I think" got the phrase.
    hits = sum(1 for r in results if r.target and r.ok)
    extras = sum(1 for r in results if r.status == "extra")
    score = max(0.0, (hits - 0.5 * extras) / len(a))
    return Attempt(target, heard, round(min(score, 1.0), 3), results)
