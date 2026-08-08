"""Check that the live LLM actually obeys the mentor/practice prompt contract.

The voice routing and the mode handoff both depend on the model following two
instructions: wrap German in guillemets, and emit a mode token when asked to
switch. A model that ignores either one breaks the feature silently, so this
asks the real server and inspects what comes back.

Needs the LLM server running. `python scripts/test_prompt.py`
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import prompts, scenarios  # noqa: E402
from app.config import load_config  # noqa: E402
from app.llm import LlmClient  # noqa: E402
from app.langid import DE  # noqa: E402
from app.segments import EN, extract_mode, split_segments  # noqa: E402

STAGE_LEAK = re.compile(r"\b(german voice|english voice|speaking slowly)\b",
                        re.IGNORECASE)


def show_routing(reply: str, tie_break: str) -> list[tuple[str, str]]:
    """Print what each voice will actually be handed."""
    segs = split_segments(reply, tie_break)
    print("      routing:")
    for lang, chunk in segs:
        print(f"        [{lang}] {chunk[:66]}")
    return segs

# Characters that should never be spoken aloud.
BAD_FOR_TTS = re.compile(r"[*_#•·]|^\s*[-\d]+[.)]\s", re.MULTILINE)
GERMAN_HINT = re.compile(
    r"\b(ich|du|ist|nicht|heiße|guten|danke|bitte|und|wie|das|der|die)\b",
    re.IGNORECASE,
)
# Words that are unmistakably English, used to catch English that has been
# wrapped in guillemets and would therefore be read by the German voice.
ENGLISH_HINT = re.compile(
    r"\b(the|you|are|how|what|hello|thank|good|and you|i am|this|that|please)\b",
    re.IGNORECASE,
)


async def ask(client: LlmClient, mode: str, level: str, scenario_key: str,
              user: str) -> str:
    sc = scenarios.get(scenario_key)
    messages = [
        {"role": "system",
         "content": prompts.system_prompt(mode, level, sc.description)},
        {"role": "user", "content": user},
    ]
    return "".join([c async for c in client.stream(messages)]).strip()


async def main() -> int:
    cfg = load_config()
    client = LlmClient(cfg)
    ok, msg = await client.health()
    if not ok:
        print(f"  LLM unreachable: {msg}")
        await client.aclose()
        return 2

    fails = 0

    def check(label, cond, detail=""):
        nonlocal fails
        fails += not cond
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        if not cond and detail:
            print(f"        {detail}")

    print(f"\nmodel: {cfg.llm.model}\n")

    # --- mentor mode -------------------------------------------------
    print("mentor mode (A1, should teach in English):")
    reply = await ask(client, prompts.MENTOR, "A1", "grundlagen",
                      "Hi, I want to learn how to introduce myself.")
    print(f"\n  > {reply}\n")

    segs = show_routing(reply, EN)
    de_text = " ".join(t for lang, t in segs if lang == DE)
    en_text = " ".join(t for lang, t in segs if lang == EN)
    spoken = " ".join(t for _, t in segs)

    check("uses guillemets for German", "«" in reply and "»" in reply,
          "no « » found — voice routing would send German to the English voice")
    check("explains in English (English text present)", len(en_text) > 40,
          f"only {len(en_text)} chars outside guillemets")
    check("no German leaking outside guillemets",
          not GERMAN_HINT.search(en_text),
          f"German-looking words in English part: {en_text[:120]!r}")
    check("no markdown / list syntax (it gets spoken)",
          not BAD_FOR_TTS.search(reply),
          f"found: {BAD_FOR_TTS.findall(reply)[:5]}")
    check("reply is short enough for speech (< 600 chars)", len(reply) < 600,
          f"{len(reply)} chars")
    check("English is routed to the English voice",
          not ENGLISH_HINT.search(de_text),
          f"English landed in a German segment: {de_text[:120]!r}")
    check("stage directions never reach the synthesiser",
          not STAGE_LEAK.search(spoken),
          f"would be read aloud: {spoken[:120]!r}")
    # Any stray token must at least be strippable, so it is never spoken.
    clean, stray = extract_mode(reply)
    check("stray mode tokens are strippable", "[[" not in clean,
          f"leftover markup would be read aloud: {clean[-60:]!r}")
    if stray:
        print(f"  NOTE  model emitted a {stray!r} token; ignored by design")

    # --- practice mode ------------------------------------------------
    # Note: mode switching is driven by app/intents.py, not by the model, so
    # there is nothing to assert about tokens here. See test_intents.py.
    print("\npractice mode (should stay in German):")
    reply3 = await ask(client, prompts.PRACTICE, "A1", "baeckerei",
                       "Hallo!")
    print(f"\n  > {reply3}\n")
    # Segment exactly as tutor.py does in practice mode: DE breaks ties.
    segs3 = show_routing(reply3, DE)
    de3 = " ".join(t for lang, t in segs3 if lang == DE)
    en3 = " ".join(t for lang, t in segs3 if lang == EN)
    check("speaks German", bool(GERMAN_HINT.search(de3 or reply3)))
    check("all German reaches the German voice",
          not GERMAN_HINT.search(en3),
          f"German landed in an English segment: {en3[:140]!r}")
    check("no English reaches the German voice",
          not ENGLISH_HINT.search(de3),
          f"English landed in a German segment: {de3[:140]!r}")
    # Soft check: immersion suffers if it translates everything it says.
    if en3.strip():
        print(f"  NOTE  practice mode still says some English: {en3[:90]!r}")

    await client.aclose()
    print(f"\n{'PASS' if not fails else 'FAIL'} — prompt contract\n")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
