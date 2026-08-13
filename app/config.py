"""Config loading. Plain dataclasses over YAML — no pydantic version roulette."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PathsCfg:
    models_root: str = "D:/stammtisch-models"
    sessions_dir: str = "./sessions"
    courses_dir: str = "./courses"
    homework_dir: str = "./homework"


@dataclass
class AudioCfg:
    input_sample_rate: int = 16000
    frame_ms: int = 32
    input_device: int | None = None
    output_device: int | None = None

    @property
    def frame_samples(self) -> int:
        return int(self.input_sample_rate * self.frame_ms / 1000)


@dataclass
class VadCfg:
    threshold: float = 0.5
    min_speech_ms: int = 250
    end_silence_ms: int = 450
    max_utterance_ms: int = 30000
    pre_roll_ms: int = 300


@dataclass
class SttCfg:
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "de"
    beam_size: int = 1
    vad_filter: bool = False
    # The only languages the learner can plausibly be speaking. Unrestricted
    # auto-detect picked Urdu, Arabic, Chinese, Italian and Indonesian on a
    # third of one real session's utterances — see Transcriber.transcribe.
    detect_languages: list[str] = field(default_factory=lambda: ["de", "en"])


@dataclass
class LlmCfg:
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "not-needed"
    model: str = "gemma3:12b"
    temperature: float = 0.7
    # Short replies are not a style preference. With barge-in off the learner
    # cannot speak until the tutor stops, so every extra sentence is dead air.
    max_tokens: int = 140
    keep_alive: str = "30m"   # Ollama unloads after 5m idle; a reload costs ~16s
    num_ctx: int = 8192       # system prompt + 12 turns does not fit in 4096
    corrector_model: str = "gemma3:12b"
    corrector_temperature: float = 0.1


@dataclass
class TtsCfg:
    # One Piper voice per language. A German voice reading English (or the
    # reverse) phonemises it with the wrong rules and comes out as gibberish.
    voices: dict[str, str] = field(
        default_factory=lambda: {
            "de": "de_DE-thorsten-high",
            "en": "en_US-lessac-medium",
        }
    )
    primary: str = "de"       # its sample rate becomes the output stream rate
    sample_rate: int = 22050  # overwritten at load from the primary voice
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8


@dataclass
class TutorCfg:
    mode: str = "mentor"      # "mentor" (teaches in English) or "practice"
    level: str = "A1"
    scenario: str = "grundlagen"
    history_turns: int = 12
    barge_in: bool = False
    # Follow a textbook instead of the stock scenarios. `course` is the file
    # stem in paths.courses_dir, blank meaning "the only one there"; `chapter`
    # is 0 to ignore the course entirely and use `scenario` above.
    course: str = ""
    chapter: int = 0
    # Pick up at the chapter the last session ended on. An explicit --chapter
    # or a chapter set in this file still wins.
    resume: bool = True


@dataclass
class LearnerCfg:
    """Who is being taught. Currently just the name, which the corrector needs
    in order to leave it alone — see app/corrections.py."""
    name: str = ""
    native_language: str = "English"


@dataclass
class Config:
    paths: PathsCfg = field(default_factory=PathsCfg)
    audio: AudioCfg = field(default_factory=AudioCfg)
    vad: VadCfg = field(default_factory=VadCfg)
    stt: SttCfg = field(default_factory=SttCfg)
    llm: LlmCfg = field(default_factory=LlmCfg)
    tts: TtsCfg = field(default_factory=TtsCfg)
    tutor: TutorCfg = field(default_factory=TutorCfg)
    learner: LearnerCfg = field(default_factory=LearnerCfg)

    @property
    def models_root(self) -> Path:
        return Path(self.paths.models_root).expanduser()

    @property
    def sessions_path(self) -> Path:
        p = Path(self.paths.sessions_dir).expanduser()
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def courses_path(self) -> Path:
        p = Path(self.paths.courses_dir).expanduser()
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def homework_path(self) -> Path:
        p = Path(self.paths.homework_dir).expanduser()
        return p if p.is_absolute() else REPO_ROOT / p


def _build(cls: type, raw: dict[str, Any]):
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config key(s): {sorted(unknown)}")
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else REPO_ROOT / "config.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    sections = {
        "paths": PathsCfg, "audio": AudioCfg, "vad": VadCfg, "stt": SttCfg,
        "llm": LlmCfg, "tts": TtsCfg, "tutor": TutorCfg,
        "learner": LearnerCfg,
    }
    kwargs = {name: _build(cls, raw.get(name) or {}) for name, cls in sections.items()}
    return Config(**kwargs)
