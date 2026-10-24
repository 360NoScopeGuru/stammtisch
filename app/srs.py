"""Spaced repetition over everything the learner has met.

Review used to be "least recently seen first", which is a queue: a word got
right ten times comes round exactly as often as one never got right. SM-2
fixes that. Each card's interval grows when it is remembered and collapses
when it is not, so time goes to the words that are actually slipping.

Cards come from four places, all already on disk:

  - vocabulary met in sessions, where the corrector gave a meaning
  - vocabulary from the learner's class notes (app/classlog.py)
  - confirmed entries from ingested handouts (app/handouts.py)
  - recurring mistakes: "fix this sentence", from progress.py

None of them is copied into a card store. The cards are rebuilt from their
sources each time, and the schedule is rebuilt by replaying an append-only log
of reviews — the same rule as progress.py: derive, never keep a parallel copy
that can drift.

Answers are typed and marked by rules, not by a model, for the reason
homework is: a learner cannot tell an unfair mark from a fair one. The rules
are deliberately forgiving about what is not being tested (case, punctuation,
umlauts typed as ae/oe/ue) and deliberately strict about what is — a noun with
the wrong article is wrong, because that is the thing that has to be learned.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from .homework import _contains, normalise

log = logging.getLogger(__name__)

LOG_NAME = "reviews.jsonl"
VOCAB, MISTAKE = "vocab", "mistake"

NEW_PER_DAY = 10
BRIEF_DUE = 8

ARTICLES = {"der", "die", "das"}


@dataclass
class Card:
    id: str
    kind: str                 # VOCAB | MISTAKE
    front: str                # what the learner sees
    back: str                 # the answer
    source: str               # "session" | "class" | deck name | "mistakes"
    wrong: str = ""           # MISTAKE only: the words that must go
    right: str = ""           # MISTAKE only: the words that must appear
    note: str = ""


@dataclass
class State:
    reps: int = 0
    interval: int = 0         # days
    ef: float = 2.5
    due: date | None = None   # None: never reviewed
    reviews: int = 0
    lapses: int = 0
    first: date | None = None

    @property
    def is_new(self) -> bool:
        return self.reviews == 0


def _card_id(kind: str, key: str) -> str:
    return hashlib.sha1(f"{kind}:{normalise(key)}".encode()).hexdigest()[:12]


# ---- collecting cards --------------------------------------------------------

def collect(study_path: Path, progress) -> list[Card]:
    from . import classlog, handouts

    cards: dict[str, Card] = {}

    def add_vocab(de: str, en: str, source: str) -> None:
        de, en = de.strip(), en.strip()
        if not de or not en:
            return          # a card needs both sides
        cid = _card_id(VOCAB, de)
        if cid not in cards:
            cards[cid] = Card(id=cid, kind=VOCAB, front=en, back=de, source=source)

    # Handouts and class notes first: they are the class's own words, so
    # their meaning wins over the corrector's gloss of the same word.
    for deck in handouts.load_decks(study_path):
        for item in deck.entries:
            if item.status == handouts.CONFIRMED:
                add_vocab(item.de, item.en, deck.name)
    for entry in classlog.load_all(study_path):
        for w in entry.vocab:
            # A noun whose article could not be confirmed is not drilled:
            # practising «Brot» without one builds the wrong habit.
            if w.article != "unsure":
                add_vocab(w.de, w.en, "class")
    for w in progress.vocab.values():
        add_vocab(w.text, w.meaning, "session")

    for m in progress.recurring(limit=30):
        if not (m.example and m.corrected) or normalise(m.example) == normalise(m.corrected):
            continue
        cid = _card_id(MISTAKE, f"{m.wrong}>{m.right}")
        cards.setdefault(cid, Card(
            id=cid, kind=MISTAKE,
            front=m.example, back=m.corrected, source="mistakes",
            wrong=m.wrong, right=m.right, note=m.explanation,
        ))
    return list(cards.values())


# ---- the schedule ------------------------------------------------------------

def _log_path(study_path: Path) -> Path:
    return Path(study_path) / LOG_NAME


def read_log(study_path: Path) -> list[dict]:
    path = _log_path(study_path)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue           # a write cut short by a kill
        if isinstance(item, dict) and "card" in item and "q" in item:
            out.append(item)
    return out


def apply(state: State, q: int, day: date) -> State:
    """One SM-2 step. `q` is the quality of recall, 0-5; 3 and up is a pass."""
    q = max(0, min(5, int(q)))
    if state.first is None:
        state.first = day
    state.reviews += 1
    if q >= 3:
        state.reps += 1
        if state.reps == 1:
            state.interval = 1
        elif state.reps == 2:
            state.interval = 6
        else:
            state.interval = max(1, round(state.interval * state.ef))
    else:
        if state.reps:
            state.lapses += 1
        state.reps = 0
        state.interval = 1
    state.ef = max(1.3, state.ef + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    state.due = day + timedelta(days=state.interval)
    return state


def schedule(study_path: Path) -> dict[str, State]:
    states: dict[str, State] = {}
    for item in sorted(read_log(study_path), key=lambda i: str(i.get("at", ""))):
        try:
            day = datetime.fromisoformat(str(item["at"])).date()
        except ValueError:
            continue
        st = states.setdefault(str(item["card"]), State())
        apply(st, int(item["q"]), day)
    return states


def queue(cards: list[Card], states: dict[str, State], today: date,
          new_per_day: int = NEW_PER_DAY) -> list[Card]:
    """What to review now: due cards, most overdue first, then a few new ones.

    New cards are rationed per day. Twenty new words in one sitting is not
    learning twenty words, it is failing nineteen of them tomorrow.
    """
    due = sorted(
        (c for c in cards if (s := states.get(c.id)) and not s.is_new
         and s.due and s.due <= today),
        key=lambda c: states[c.id].due,
    )
    introduced_today = sum(1 for s in states.values() if s.first == today)
    room = max(0, new_per_day - introduced_today)
    # Mistakes before words: they are the learner's own, and they recur.
    new = sorted((c for c in cards if c.id not in states),
                 key=lambda c: (c.kind != MISTAKE, c.source == "session"))
    return due + new[:room]


def record(study_path: Path, card: Card, q: int, given: str) -> None:
    path = _log_path(study_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"at": datetime.now().isoformat(timespec="seconds"),
             "card": card.id, "q": int(q), "given": given[:200],
             "kind": card.kind, "back": card.back}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---- marking -----------------------------------------------------------------

def _plain(text: str) -> str:
    """Lowercase, no punctuation, umlauts kept. Exact-but-forgiving."""
    text = unicodedata.normalize("NFC", text.strip().lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_TRANSLIT = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


def _transliterate(text: str) -> str:
    return text.translate(_TRANSLIT)


def _distance(a: str, b: str) -> int:
    """Levenshtein, for "one letter off" — small strings only."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class Result:
    q: int                    # SM-2 quality, 0-5
    correct: bool
    note: str


def grade(card: Card, given: str) -> Result:
    if not given.strip():
        return Result(0, False, f"It is «{card.back}».")
    if card.kind == MISTAKE:
        return _grade_mistake(card, given)
    return _grade_vocab(card, given)


def _grade_vocab(card: Card, given: str) -> Result:
    want, got = _plain(card.back), _plain(given)
    if got == want:
        return Result(5, True, "Right.")
    # «Brotchen» and «Broetchen» are both a keyboard without umlauts, not a
    # learner who does not know the word.
    if normalise(given) == normalise(card.back) or got == _transliterate(want):
        return Result(4, True, f"Right — spelt «{card.back}».")

    want_words, got_words = normalise(card.back).split(), normalise(given).split()
    # Nouns are learned with their article, so the article is marked.
    if want_words and want_words[0] in ARTICLES:
        noun = want_words[1:]
        if got_words[1:] == noun and got_words[:1] and got_words[0] in ARTICLES:
            return Result(1, False,
                          f"Right noun, wrong article: «{card.back}».")
        if got_words == noun:
            return Result(3, True,
                          f"Right word — learn it with its article: «{card.back}».")

    folded_want, folded_got = normalise(card.back), normalise(given)
    allowed = 1 if len(folded_want) < 10 else 2
    if len(folded_want) >= 5 and _distance(folded_want, folded_got) <= allowed:
        return Result(3, True, f"Nearly — it is spelt «{card.back}».")
    return Result(1, False, f"It is «{card.back}».")


def _grade_mistake(card: Card, given: str) -> Result:
    if _plain(given) == _plain(card.back) or normalise(given) == normalise(card.back):
        return Result(5, True, "Right.")
    tokens = normalise(given).split()
    right = normalise(card.right).split()
    wrong = normalise(card.wrong).split()
    fixed = _contains(tokens, right) if right else True
    kept = _contains(tokens, wrong) if wrong else False
    if fixed and not kept:
        return Result(4, True, f"Right — «{card.back}».")
    return Result(1, False, f"It should be «{card.back}»."
                  + (f" {card.note}" if card.note else ""))


# ---- for the tutor -----------------------------------------------------------

def brief(study_path: Path, progress, today: date) -> str:
    """Words due for review, for the system prompt. "" when nothing is due."""
    try:
        cards = collect(study_path, progress)
        states = schedule(study_path)
    except Exception:
        log.warning("could not build the review schedule", exc_info=True)
        return ""
    due = [c for c in cards if c.kind == VOCAB and (s := states.get(c.id))
           and s.due and s.due <= today][:BRIEF_DUE]
    if not due:
        return ""
    return ("Words due for review, which you can work into the conversation "
            "where they fit: "
            + ", ".join(f"«{c.back}» ({c.front})" for c in due) + ".")


def counts(cards: list[Card], states: dict[str, State], today: date) -> dict:
    due = sum(1 for c in cards if (s := states.get(c.id)) and s.due and s.due <= today)
    new = sum(1 for c in cards if c.id not in states)
    introduced = sum(1 for s in states.values() if s.first == today)
    return {"due": due, "new": min(new, max(0, NEW_PER_DAY - introduced)),
            "total": len(cards),
            "learned": sum(1 for s in states.values() if s.interval >= 21)}
