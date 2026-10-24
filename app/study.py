"""The model-backed steps of the study features.

Each function here does one call (or a short fixed sequence) and hands the
result to the module that checks it: classlog, handouts, objectives. Nothing
in this file decides whether model output is right — that is always the job of
the deterministic code it is handed to.
"""

from __future__ import annotations

import logging
from datetime import date

from . import classlog, documents, handouts, objectives, prompts
from .curriculum import Course
from .llm import LlmClient

log = logging.getLogger(__name__)

# Vision calls take ~6 s a page warm. A class note with a dozen photos would
# keep the learner waiting a minute, and the context would not hold them all.
MAX_CLASS_PAGES = documents.MAX_PAGES_PER_MESSAGE


async def log_class(llm: LlmClient, course: Course | None, note: str,
                    attached: list[tuple[documents.Document, list[documents.Page]]],
                    day: date, max_tokens: int = 1200) -> classlog.Entry | None:
    """Turn a note and some pages into a checked class entry."""
    pages = [p for _, ps in attached for p in ps][:MAX_CLASS_PAGES]
    listing = course.listing() if course else ""
    raw = await llm.complete_json(
        prompts.classlog_messages(note, listing, pages),
        max_tokens=max_tokens,
    )
    if raw is None:
        return None
    # Words are checked against every bit of text the entry came from: the
    # note, plus the text layer of each attached PDF page.
    source_text = "\n".join(p.text for p in pages if p.text)
    entry = classlog.parse_extracted(
        raw, note=note, source_text=source_text,
        sources=[documents.describe(d, ps) for d, ps in attached],
        day=day,
        chapters={c.number for c in course.chapters} if course else set(),
    )
    # Articles get a second, independent opinion — see classlog.apply_lexicon.
    if nouns := classlog.articles_to_check(entry, source_text):
        classlog.apply_lexicon(entry, handouts.parse_lexicon(
            await llm.complete_json(prompts.handout_lexicon_messages(nouns),
                                    max_tokens=1200)))
    return entry


async def read_handout(llm: LlmClient, doc: documents.Document, name: str,
                       pages: list[documents.Page], progress=None
                       ) -> handouts.Deck:
    """Read every page three ways and keep what agrees. See handouts.py."""
    found: list[handouts.Item] = []
    for i, page in enumerate(pages):
        if progress:
            progress(f"Reading page {page.number} ({i + 1} of {len(pages)})…")

        position: list[tuple[str, str]] = []
        if doc.kind == documents.PDF:
            try:
                words = handouts.words_from_pdf_page(doc._source, page.number)
                position = handouts.read_by_position(words)
            except Exception:
                log.warning("positional reading failed on page %s",
                            page.number, exc_info=True)

        vision = handouts.parse_vision(
            await llm.complete_json(prompts.handout_vision_messages(page),
                                    max_tokens=1500))

        nouns = sorted({handouts.clean_noun(t) for t, _ in position}
                       | {n for n, _ in vision})
        nouns = [n for n in nouns if n]
        lexicon: dict = {}
        # In batches: a long list in one call is where models start skipping
        # entries, and a skipped noun is only pending, never wrong — but a
        # page of pending entries is a chore nobody does.
        for start in range(0, len(nouns), 40):
            lexicon.update(handouts.parse_lexicon(await llm.complete_json(
                prompts.handout_lexicon_messages(nouns[start:start + 40]),
                max_tokens=2000)))

        found += handouts.combine(position, vision, lexicon, page=page.number)
        log.info("handout page %s: %d position, %d vision, %d lexicon",
                 page.number, len(position), len(vision), len(lexicon))

    return handouts.new_deck(name, doc.name, found)


async def goal_task(llm: LlmClient, level: str, goal: str, chapter: str
                    ) -> str | None:
    raw = await llm.complete_json(
        prompts.goal_task_messages(level, goal, chapter), temperature=0.5)
    task = str((raw or {}).get("task", "")).strip()
    return task[:400] or None


async def judge_goal(llm: LlmClient, level: str, goal: str, task: str,
                     answer: str) -> tuple[bool, str] | None:
    """(passed, note), or None if the check itself failed."""
    sure = objectives.precheck(answer)
    if sure is not None:
        return sure
    return objectives.parse_verdict(await llm.complete_json(
        prompts.goal_judge_messages(level, goal, task, answer)))
