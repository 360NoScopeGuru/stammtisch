"""A tiny async pub/sub bus.

The conversation loop publishes; the CLI printer and any connected browsers
subscribe. Publishing is non-blocking and never raises — a stalled or dead
subscriber must not be able to wedge the audio pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any

log = logging.getLogger(__name__)

QUEUE_MAX = 256
REPLAY_MAX = 400


class EventBus:
    def __init__(self) -> None:
        self._subs: set[asyncio.Queue[dict]] = set()
        # Late joiners (a browser opened mid-session) get the backlog.
        self._replay: deque[dict] = deque(maxlen=REPLAY_MAX)
        self._seq = 0

    # ---- subscription -------------------------------------------------

    def subscribe(self) -> asyncio.Queue[dict]:
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=QUEUE_MAX)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict]) -> None:
        self._subs.discard(q)

    def replay(self) -> list[dict]:
        return list(self._replay)

    # ---- publishing ---------------------------------------------------

    def publish(self, type_: str, **data: Any) -> dict:
        self._seq += 1
        event = {"seq": self._seq, "type": type_, "t": time.time(), **data}

        if type_ != "status":  # status is ephemeral, not worth replaying
            self._replay.append(event)

        for q in list(self._subs):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop the oldest rather than the newest; a slow client should
                # see recent state, not a stale prefix.
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    log.debug("dropping event for a stalled subscriber")
        return event
