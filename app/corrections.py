"""Filtering the grammar corrector's output before it reaches the learner.

The corrector is a small local model asked to grade one utterance. It is good
at case endings and word order and bad at knowing what a name is, because a
name is exactly the thing it has never seen in training. From a real session:

    heard:      "Ich heiße Fahmshidharan."
    corrected:  "Ich heiße Farshid Shidharan."

The learner said their own name correctly. Whisper mangled the spelling, and
the corrector then "fixed" the mangling into a different name entirely and
presented it as a German error. That is worse than saying nothing: it is
confident, authoritative, and about a word the learner never got wrong.

Prompting alone does not fix this — the model cannot tell a name it has never
seen from a misspelled noun. So corrections that turn on a proper noun are
dropped here, after the fact, where the check is cheap and deterministic.
"""

from __future__ import annotations

import difflib
import re

# How close a mangled transcription has to be to a configured name before we
# treat it as that name. Measured against real data, the separation is thin:
# actual manglings of "Vamshee Dharan" score 0.36-1.00, while ordinary German
# words that turn up in genuine corrections reach 0.55 ("waren", "Name").
#
# 0.60 sits above the noise. It lets a few manglings through rather than
# swallowing real grammar feedback, which is the right way round to fail — a
# missed drop costs one bad correction, an over-eager one silently withholds
# the thing the learner is here for. The exact rules below do the real work;
# this is only a backstop for names the utterance did not announce.
NAME_SIMILARITY = 0.60

# A token straight after one of these is a name, whatever the model thinks.
# Deliberately excludes "bin"/"ist": "Ich bin Student" would otherwise protect
# "Student" and suppress the very real correction to "Studentin".
NAMING_VERBS = {
    "heiße", "heisse", "heißt", "heisst", "heiß", "heiss",
    "nenne", "nennt", "nennen",
}

# "Mein Name ist X". The possessive is required: without it, "Der Name ist gut"
# would protect "gut" and suppress a perfectly good vocabulary correction.
POSSESSIVES = {"mein", "meine", "dein", "deine", "sein", "seine", "ihr",
               "ihre", "unser", "unsere"}
NAME_COPULAS = {"ist", "war", "lautet"}

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def tokens(text: str) -> list[str]:
    return _WORD.findall(text)


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def protected_names(learner_name: str, utterance: str) -> set[str]:
    """Every token in this utterance that is somebody's name.

    Two sources: the learner's configured name (in all its parts, since they
    may say only the first), and anything following a naming verb — which
    catches names the config never knew about, like a city or a friend.
    """
    protected = {t.lower() for t in tokens(learner_name)}

    words = [w.lower() for w in tokens(utterance)]
    for i in range(len(words) - 1):
        # German capitalises every noun, so capitalisation alone proves
        # nothing. Position after a naming verb does.
        if words[i] in NAMING_VERBS:
            protected.add(words[i + 1])
        elif (words[i] in NAME_COPULAS and i >= 2
                and words[i - 1] == "name" and words[i - 2] in POSSESSIVES):
            protected.add(words[i + 1])
    return {p for p in protected if len(p) > 2}


def changed_tokens(original: str, corrected: str) -> list[str]:
    """The words that differ between the two, from both sides."""
    a, b = tokens(original), tokens(corrected)
    out: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag != "equal":
            out.extend(a[i1:i2])
            out.extend(b[j1:j2])
    return out


def touches_a_name(original: str, corrected: str, protected: set[str]) -> bool:
    """Does this correction change a proper noun?"""
    if not protected:
        return False
    for token in changed_tokens(original, corrected):
        low = token.lower()
        if low in protected:
            return True
        if any(_similar(low, p) >= NAME_SIMILARITY for p in protected):
            return True
    return False


def filter_corrections(
    corrections: list[dict], learner_name: str, utterance: str
) -> tuple[list[dict], list[dict]]:
    """Split into (kept, dropped). Dropped ones are logged, never shown."""
    protected = protected_names(learner_name, utterance)
    kept, dropped = [], []
    for c in corrections:
        original = str(c.get("original", ""))
        corrected = str(c.get("corrected", ""))
        if not original or not corrected:
            dropped.append(c)
        elif original.strip() == corrected.strip():
            dropped.append(c)          # a "correction" that changes nothing
        elif touches_a_name(original, corrected, protected):
            dropped.append(c)
        else:
            kept.append(c)
    return kept, dropped
