"""Download the Piper German voice and the Silero VAD model into models_root.

Whisper weights download themselves on first run (into models_root/whisper).
The LLM is pulled separately via Ollama / LM Studio.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config  # noqa: E402

HF = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
SILERO = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/"
    "silero_vad.onnx"
)

VOICES = {
    # key: (path under the piper-voices repo, quality dir)
    "de_DE-thorsten-high": "de/de_DE/thorsten/high/de_DE-thorsten-high",
    "de_DE-thorsten-medium": "de/de_DE/thorsten/medium/de_DE-thorsten-medium",
    "de_DE-eva_k-x_low": "de/de_DE/eva_k/x_low/de_DE-eva_k-x_low",
    "de_DE-kerstin-low": "de/de_DE/kerstin/low/de_DE-kerstin-low",
}


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have  {dest.name}")
        return
    print(f"  get   {dest.name} ...", end="", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f" {dest.stat().st_size / 1e6:.1f} MB")


def main() -> int:
    cfg = load_config()
    voice = cfg.tts.voice
    if voice not in VOICES:
        print(f"Unknown voice {voice!r}. Known: {', '.join(VOICES)}")
        return 2

    out = cfg.models_root / "piper"
    print(f"Piper voice -> {out}")
    stem = VOICES[voice]
    download(f"{HF}/{stem}.onnx", out / f"{voice}.onnx")
    download(f"{HF}/{stem}.onnx.json", out / f"{voice}.onnx.json")

    vad_dir = cfg.models_root / "vad"
    print(f"Silero VAD -> {vad_dir}")
    download(SILERO, vad_dir / "silero_vad.onnx")

    print("\nDone. Whisper downloads on first run.")
    print(f"Make sure your LLM server is serving: {cfg.llm.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
