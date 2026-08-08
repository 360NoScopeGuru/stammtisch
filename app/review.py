"""Session logging, end-of-session review, and Anki export."""

from __future__ import annotations

import csv
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TextIO

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class Feedback:
    utterance: str
    corrections: list[dict]
    vocab: list[str]


@dataclass
class SessionLog:
    cfg: Config
    level: str
    scenario: str
    started: datetime = field(default_factory=datetime.now)
    transcript: list[dict] = field(default_factory=list)
    feedback: list[Feedback] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Fixed at construction so that changing scenario mid-session keeps
        # appending to one file instead of forking a new one. Seconds are in
        # there because restarting inside the same minute is common when you
        # are fixing something.
        self._stem = f"{self.started:%Y-%m-%d_%H%M%S}_{self.scenario}"

    def add_turn(self, role: str, text: str, latency_ms: float | None = None) -> None:
        self.transcript.append(
            {
                "role": role,
                "text": text,
                "at": datetime.now().isoformat(timespec="seconds"),
                "latency_ms": round(latency_ms) if latency_ms else None,
            }
        )
        self._autosave()

    def add_feedback(self, utterance: str, corrections: list[dict],
                     vocab: list[str]) -> None:
        self.feedback.append(Feedback(utterance, corrections, vocab))
        self._autosave()

    def _autosave(self) -> None:
        """Persist after every turn.

        Saving used to happen only in `Tutor.aclose()`, which meant it ran only
        on a clean shutdown — and this app is normally ended with Ctrl+C, a
        closed console window, or a kill. The result was that `sessions/` stayed
        empty across every real session, and the review and Anki export, which
        are the entire point of logging, had never once produced a file.

        A session is a few kilobytes, so writing the whole thing each turn is
        cheaper than any incremental format would be to maintain. Never let a
        disk problem take down a conversation.
        """
        try:
            self.save()
        except Exception:
            log.warning("could not autosave session", exc_info=True)

    # ---- output -------------------------------------------------------

    @property
    def all_corrections(self) -> list[dict]:
        return [c for f in self.feedback for c in f.corrections]

    @property
    def all_vocab(self) -> list[str]:
        seen, out = set(), []
        for f in self.feedback:
            for word in f.vocab:
                key = word.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    out.append(word.strip())
        return out

    def summary(self) -> str:
        user_turns = [t for t in self.transcript if t["role"] == "user"]
        mins = (datetime.now() - self.started).total_seconds() / 60
        latencies = [
            t["latency_ms"] for t in self.transcript
            if t["role"] == "assistant" and t["latency_ms"]
        ]

        lines = [
            "",
            "─" * 58,
            f"  Sitzung beendet — {self.scenario} ({self.level})",
            "─" * 58,
            f"  {mins:.0f} min · {len(user_turns)} Redebeiträge · "
            f"{len(self.all_corrections)} Korrekturen",
        ]
        if latencies:
            lines.append(
                f"  Antwortzeit: {sum(latencies)/len(latencies):.0f} ms Median-ish "
                f"(min {min(latencies):.0f} / max {max(latencies):.0f})"
            )

        if self.all_corrections:
            lines += ["", "  Korrekturen:"]
            for c in self.all_corrections[:15]:
                lines.append(f"    ✗ {c.get('original', '')}")
                lines.append(f"    ✓ {c.get('corrected', '')}")
                if expl := c.get("explanation"):
                    lines.append(f"      {expl}")
                lines.append("")
        else:
            lines += ["", "  Keine Fehler gefunden. Stark!"]

        if vocab := self.all_vocab:
            lines += ["  Wortschatz: " + ", ".join(vocab[:20])]

        lines.append("─" * 58)
        return "\n".join(lines)

    def _write_atomic(self, path: Path, write: Callable[[TextIO], None]) -> None:
        """Write via a temp file and rename.

        Called after every turn, so a kill landing mid-write is a real
        possibility — and a half-written session file is worse than none,
        because it looks like data until you try to parse it.
        """
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            write(fh)
        os.replace(tmp, path)

    def save(self) -> tuple[str, str] | None:
        """Write session JSON + an Anki-importable CSV. Returns their paths."""
        if not self.transcript:
            return None

        out_dir = self.cfg.sessions_path
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "started": self.started.isoformat(timespec="seconds"),
            "updated": datetime.now().isoformat(timespec="seconds"),
            # Read live rather than from the snapshot taken at construction,
            # so a mid-session level or scenario change is not lost.
            "level": self.cfg.tutor.level,
            "scenario": self.cfg.tutor.scenario,
            "mode": self.cfg.tutor.mode,
            "started_as": {"level": self.level, "scenario": self.scenario},
            "transcript": self.transcript,
            "corrections": self.all_corrections,
            "vocab": self.all_vocab,
        }

        json_path = out_dir / f"{self._stem}.json"
        self._write_atomic(
            json_path,
            lambda fh: json.dump(payload, fh, ensure_ascii=False, indent=2),
        )

        # Anki: front,back — import with comma as the field separator.
        csv_path = out_dir / f"{self._stem}_anki.csv"

        def write_csv(fh) -> None:
            w = csv.writer(fh)
            for c in self.all_corrections:
                front = c.get("original", "").strip()
                back = c.get("corrected", "").strip()
                if not (front and back):
                    continue
                if expl := c.get("explanation"):
                    back = f"{back}<br><i>{expl}</i>"
                w.writerow([front, back])
            for word in self.all_vocab:
                w.writerow([word, ""])

        self._write_atomic(csv_path, write_csv)

        log.debug("session saved: %s", json_path)
        return str(json_path), str(csv_path)
