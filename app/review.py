"""Session logging, end-of-session review, and Anki export."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

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

    def add_turn(self, role: str, text: str, latency_ms: float | None = None) -> None:
        self.transcript.append(
            {
                "role": role,
                "text": text,
                "at": datetime.now().isoformat(timespec="seconds"),
                "latency_ms": round(latency_ms) if latency_ms else None,
            }
        )

    def add_feedback(self, utterance: str, corrections: list[dict],
                     vocab: list[str]) -> None:
        self.feedback.append(Feedback(utterance, corrections, vocab))

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

    def save(self) -> tuple[str, str] | None:
        """Write session JSON + an Anki-importable CSV. Returns their paths."""
        if not self.transcript:
            return None

        out_dir = self.cfg.sessions_path
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.started:%Y-%m-%d_%H%M}_{self.scenario}"

        json_path = out_dir / f"{stem}.json"
        json_path.write_text(
            json.dumps(
                {
                    "started": self.started.isoformat(timespec="seconds"),
                    "level": self.level,
                    "scenario": self.scenario,
                    "transcript": self.transcript,
                    "corrections": self.all_corrections,
                    "vocab": self.all_vocab,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Anki: front,back — import with comma as the field separator.
        csv_path = out_dir / f"{stem}_anki.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
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

        log.info("session saved: %s", json_path)
        return str(json_path), str(csv_path)
