"""Does the correction pass actually stay off the critical path?

The design claims corrections cost nothing because they run asynchronously.
That assumes the LLM server can serve two requests at once. Ollama defaults to
OLLAMA_NUM_PARALLEL=1, which would mean a correction blocks the next reply —
and the reply is the thing the learner is waiting for.

    python scripts/probe_contention.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import prompts, scenarios  # noqa: E402
from app.config import load_config  # noqa: E402
from app.llm import LlmClient  # noqa: E402

REPLY = [
    {"role": "system", "content": prompts.system_prompt(
        prompts.MENTOR, "A1", scenarios.get("grundlagen").description)},
    {"role": "user", "content": "How do I say good morning?"},
]


async def timed_reply(client: LlmClient) -> float:
    t0 = time.perf_counter()
    async for _ in client.stream(REPLY):
        return time.perf_counter() - t0   # time to FIRST token
    return time.perf_counter() - t0


async def timed_correction(client: LlmClient) -> float:
    t0 = time.perf_counter()
    await client.complete_json(
        prompts.corrector_messages("A1", "Ich heiße Vamshee und ich komme aus Indien.")
    )
    return time.perf_counter() - t0


async def main() -> int:
    cfg = load_config()
    client = LlmClient(cfg)
    ok, msg = await client.health()
    if not ok:
        print(f"LLM unreachable: {msg}")
        return 2
    await client.pin_model()

    print(f"\nmodel: {cfg.llm.model}   corrector: {cfg.llm.corrector_model}\n")

    # Baselines, one at a time.
    await timed_reply(client)                      # warm
    solo_reply = await timed_reply(client)
    solo_corr = await timed_correction(client)
    print(f"  reply alone,      time-to-first-token : {solo_reply * 1000:7.0f} ms")
    print(f"  correction alone, full call           : {solo_corr * 1000:7.0f} ms")

    # Now the real sequence: a correction is fired, then a reply is requested
    # immediately after — exactly what happens on every German turn.
    print("\n  firing a correction, then requesting a reply right after:")
    t0 = time.perf_counter()
    corr_task = asyncio.create_task(timed_correction(client))
    await asyncio.sleep(0.05)
    contended_reply = await timed_reply(client)
    await corr_task
    total = time.perf_counter() - t0

    print(f"  reply time-to-first-token             : {contended_reply * 1000:7.0f} ms")
    penalty = (contended_reply - solo_reply) * 1000
    print(f"  penalty vs solo                       : {penalty:+7.0f} ms")
    print(f"  wall clock for both                   : {total * 1000:7.0f} ms")

    serialized = contended_reply > solo_reply + solo_corr * 0.5
    print()
    if serialized:
        print("  VERDICT: requests are SERIALISED. The correction is not free —")
        print("           it is prepended to the learner's wait.")
    else:
        print("  VERDICT: requests overlap; corrections really are off the path.")

    await client.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
