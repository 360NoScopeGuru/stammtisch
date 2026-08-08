"""Measure where the reply latency actually goes.

The number that matters is not tokens/sec, it is **time to first spoken
sentence** — nothing can be voiced until one complete sentence exists. A model
that opens with a long preamble feels slow even at a high token rate.

    python scripts/bench_llm.py                # current configured model
    python scripts/bench_llm.py qwen3:14b ...  # compare candidates
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import prompts, scenarios  # noqa: E402
from app.config import load_config  # noqa: E402
from app.llm import LlmClient  # noqa: E402
from app.sentences import SentenceStreamer  # noqa: E402

TURNS = [
    "Hi, I want to learn how to introduce myself.",
    "Hallo. Ich heiße Vamshee.",
    "What does that mean?",
]


async def one_turn(client: LlmClient, model: str, messages: list[dict]) -> dict:
    prev, client.cfg.llm.model = client.cfg.llm.model, model
    streamer = SentenceStreamer()

    t0 = time.perf_counter()
    ttft = first_sentence = None
    chunks = 0
    text = ""
    first_sentence_text = ""

    try:
        async for delta in client.stream(messages):
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            chunks += 1
            text += delta
            if first_sentence is None:
                got = streamer.feed(delta)
                if got:
                    first_sentence = now - t0
                    first_sentence_text = got[0]
    finally:
        client.cfg.llm.model = prev

    total = time.perf_counter() - t0
    return {
        "ttft": ttft or total,
        "first_sentence": first_sentence or total,
        "total": total,
        "chars": len(text),
        "chunks": chunks,
        "rate": chunks / total if total else 0,
        "first_sentence_text": first_sentence_text or text[:70],
        "reply": text,
    }


async def bench(model: str, cfg, warm: bool) -> None:
    client = LlmClient(cfg)
    sc = scenarios.get("grundlagen")
    system = prompts.system_prompt(prompts.MENTOR, "A1", sc.description)

    print(f"\n{'=' * 66}\n{model}\n{'=' * 66}")

    if warm:
        await one_turn(client, model, [
            {"role": "system", "content": system},
            {"role": "user", "content": "hi"},
        ])

    history: list[dict] = []
    rows = []
    for user in TURNS:
        history.append({"role": "user", "content": user})
        messages = [{"role": "system", "content": system}, *history]
        r = await one_turn(client, model, messages)
        history.append({"role": "assistant", "content": r["reply"]})
        rows.append(r)

        print(f"\n  user> {user}")
        print(f"  ttft {r['ttft'] * 1000:6.0f} ms   "
              f"first sentence {r['first_sentence'] * 1000:6.0f} ms   "
              f"full {r['total'] * 1000:6.0f} ms   "
              f"{r['chars']} chars")
        print(f"  1st: {r['first_sentence_text'][:80]}")

    fs = [r["first_sentence"] * 1000 for r in rows]
    print(f"\n  median time-to-first-sentence: {statistics.median(fs):.0f} ms")
    print(f"  median full reply:             "
          f"{statistics.median([r['total'] * 1000 for r in rows]):.0f} ms")
    print(f"  median reply length:           "
          f"{statistics.median([r['chars'] for r in rows]):.0f} chars")
    await client.aclose()


async def main() -> int:
    cfg = load_config()
    models = sys.argv[1:] or [cfg.llm.model]

    client = LlmClient(cfg)
    ok, msg = await client.health()
    await client.aclose()
    if not ok and len(models) == 1:
        print(f"LLM unreachable: {msg}")
        return 2

    print("\nEach model is warmed once, then measured over 3 turns.")
    print("Target: first sentence under ~800 ms.")
    for m in models:
        try:
            await bench(m, cfg, warm=True)
        except Exception as e:
            print(f"\n  {m}: FAILED — {str(e)[:160]}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
