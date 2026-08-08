"""Decide whether a fragment is German or English, by looking at it.

Voice routing used to trust the model's guillemets. That is not safe. Across
two models the marking failed in both directions:

    gemma3:12b:  «Hallo! Wie geht es dir?» (Hello! How are you?)
                 -> English gloss outside the marks (fine), but earlier
                    versions put the gloss *inside* them, sending English to
                    the German voice.
    gemma3:4b:   «Guten Tag!» Wie geht es Ihnen heute?
                 -> German outside the marks, sent to the English voice.

Either way the learner hears phonetic mush and has no idea why. Function words
are the giveaway between these two languages and cost nothing to count, so we
classify the text instead of trusting the formatting. The guillemets are still
used for UI highlighting, and as a tie-breaker when the text is too short to
call.
"""

from __future__ import annotations

import re

DE, EN = "de", "en"

_WORD = re.compile(r"[a-zà-öø-ÿß]+", re.IGNORECASE)

# High-frequency function words. Deliberately excludes words common to both
# ("in", "so", "was" is German but also English past tense, etc. are handled by
# weighting rather than exclusion where they still carry signal).
GERMAN = {
    "ich", "du", "sie", "wir", "ihr", "er", "es", "mich", "dir", "mir", "ihnen",
    "ist", "sind", "bin", "bist", "war", "haben", "hat", "habe", "kann", "können",
    "nicht", "kein", "keine", "und", "oder", "aber", "auch", "noch", "schon",
    "das", "der", "die", "den", "dem", "ein", "eine", "einen", "einem",
    "mit", "für", "auf", "aus", "bei", "von", "zu", "zum", "zur", "im", "am",
    "wie", "wo", "wer", "warum", "wann", "welche",
    "heißt", "heiße", "heißen", "geht", "gehe", "möchte", "möchten", "bitte",
    "danke", "guten", "morgen", "abend", "tag", "hallo", "tschüss", "ja", "nein",
    "gut", "sehr", "mein", "meine", "dein", "deine", "wirklich", "vielleicht",
    "jetzt", "heute", "gern", "gerne", "sehen", "sagen", "sprechen", "üben",
}

ENGLISH = {
    "the", "you", "your", "yours", "are", "is", "was", "were", "been",
    "and", "or", "but", "not", "this", "that", "these", "those",
    "what", "where", "who", "why", "when", "which", "how",
    "hello", "hi", "thanks", "thank", "please", "good", "great", "nice",
    "means", "meaning", "say", "saying", "said", "try", "trying", "repeat",
    "let", "lets", "can", "could", "would", "should", "will",
    "with", "from", "about", "again", "back", "just", "very", "really",
    "i", "we", "they", "he", "she", "it", "me", "my", "our", "their",
    "to", "of", "for", "have", "has", "do", "does", "did", "want", "like",
    "now", "then", "here", "there", "yes", "no", "okay", "ok",
}

# Umlauts and ß are near-decisive on their own.
_GERMAN_CHARS = re.compile(r"[äöüß]", re.IGNORECASE)
UMLAUT_WEIGHT = 2.0


def score(text: str) -> tuple[float, float]:
    """Return (german_score, english_score) for `text`."""
    words = [w.lower() for w in _WORD.findall(text)]
    de = sum(w in GERMAN for w in words)
    en = sum(w in ENGLISH for w in words)
    de += UMLAUT_WEIGHT * len(_GERMAN_CHARS.findall(text))
    return float(de), float(en)


def detect(text: str, default: str = EN) -> str:
    """Classify `text` as German or English.

    Falls back to `default` when there is no evidence either way — which is
    common for very short fragments, where the caller's context (the mode, or
    whether the text was in guillemets) is better evidence than the words are.
    """
    de, en = score(text)
    if de > en:
        return DE
    if en > de:
        return EN
    return default
