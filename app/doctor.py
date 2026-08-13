"""Check every precondition and say what to fix.

Almost everything that has gone wrong with this app was invisible at the point
of failure and obvious somewhere else. The microphone "not working" was a VAD
fed 512 samples instead of 576. A five-times slowdown was a laptop on battery
with the GPU drawing 33 W of 175. A `cublas64_12.dll` error was a missing pip
package three layers down. None of those announce themselves; all of them are
one query away if you know where to look.

So this looks in all the places at once, and every failure carries the command
that fixes it. Nothing here loads a model or opens the microphone for real —
it must stay fast enough that running it is never a decision.
"""

from __future__ import annotations

import importlib.util
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .config import Config

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    fix: str = ""

    @property
    def bad(self) -> bool:
        return self.status == FAIL


def _run(cmd: list[str], timeout: float = 4.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


# --- individual checks ---------------------------------------------------

def check_python() -> Check:
    import sys
    v = sys.version_info
    if v[:2] == (3, 12):
        return Check("Python", OK, f"{v.major}.{v.minor}.{v.micro}")
    if v < (3, 10):
        return Check("Python", FAIL, f"{v.major}.{v.minor}",
                     "this needs 3.10+; 3.12 is what it is developed against")
    return Check("Python", WARN, f"{v.major}.{v.minor}",
                 "3.12 is what this is developed against; wheels are more "
                 "reliable there")


def check_packages() -> list[Check]:
    wanted = {
        "faster_whisper": "speech recognition",
        "piper": "speech synthesis",
        "onnxruntime": "VAD and Piper runtime",
        "sounddevice": "microphone and speaker",
        "httpx": "LLM client",
        "fastapi": "web UI",
        "yaml": "config",
    }
    out = []
    for mod, why in wanted.items():
        if importlib.util.find_spec(mod) is None:
            out.append(Check(f"package {mod}", FAIL, f"missing ({why})",
                             "pip install -r requirements.txt"))
    if importlib.util.find_spec("pypdf") is None:
        out.append(Check("package pypdf", WARN,
                         "missing — textbook ingestion will not run",
                         "pip install pypdf"))
    # The one package that must NOT be here.
    if importlib.util.find_spec("torch") is not None:
        out.append(Check("torch", WARN, "installed",
                         "this project has no torch dependency and a "
                         "mismatched torchaudio breaks the VAD import — "
                         "pip uninstall torch torchaudio"))
    if not out:
        out.append(Check("packages", OK, f"{len(wanted)} required imports present"))
    return out


def check_models(cfg: Config) -> list[Check]:
    root = cfg.models_root
    out = []
    if not root.is_dir():
        return [Check("model files", FAIL, f"{root} does not exist",
                      "python scripts/fetch_models.py")]

    vad = list(root.rglob("silero_vad*.onnx"))
    out.append(
        Check("Silero VAD", OK, vad[0].name) if vad
        else Check("Silero VAD", FAIL, f"not found under {root}",
                   "python scripts/fetch_models.py")
    )

    for lang, voice in cfg.tts.voices.items():
        found = list(root.rglob(f"{voice}.onnx"))
        out.append(
            Check(f"Piper voice ({lang})", OK, voice) if found
            else Check(f"Piper voice ({lang})", FAIL, f"{voice}.onnx not found",
                       "python scripts/fetch_models.py")
        )

    whisper = root / "whisper"
    if whisper.is_dir() and any(whisper.iterdir()):
        out.append(Check("Whisper cache", OK, f"{cfg.stt.model} in {whisper}"))
    else:
        out.append(Check("Whisper cache", WARN, "empty",
                         f"{cfg.stt.model} downloads on first run (~1.5 GB)"))
    return out


def check_llm(cfg: Config) -> list[Check]:
    """Reachability and whether the configured model is actually pulled.

    Deliberately synchronous and socket-level first: an unreachable port is
    the single most common failure, and it should not take an HTTP timeout to
    discover.
    """
    url = urlparse(cfg.llm.base_url)
    host, port = url.hostname or "localhost", url.port or 80

    try:
        with socket.create_connection((host, port), timeout=2):
            pass
    except OSError:
        return [Check("LLM server", FAIL, f"nothing listening on {host}:{port}",
                      "start it — for Ollama that is `ollama serve`")]

    out = [Check("LLM server", OK, f"{host}:{port} reachable")]

    tags = _run(["curl", "-s", "-m", "4", f"http://{host}:{port}/api/tags"])
    if not tags:
        out.append(Check("LLM model", WARN, "cannot list models",
                         "not Ollama? then check the model manually"))
        return out

    import json
    try:
        names = {m["name"] for m in json.loads(tags).get("models", [])}
    except (ValueError, KeyError, TypeError):
        return out + [Check("LLM model", WARN, "unexpected /api/tags response")]

    for label, model in (("reply", cfg.llm.model),
                         ("corrector", cfg.llm.corrector_model)):
        if model in names:
            out.append(Check(f"model ({label})", OK, model))
        else:
            out.append(Check(f"model ({label})", FAIL, f"{model} not pulled",
                             f"ollama pull {model}"))
    return out


def check_power() -> list[Check]:
    """The check that has saved the most time in this project.

    On battery with a throttled profile this laptop's GPU sat at 96-100%
    utilisation while drawing 33 W of a 175 W limit — a fifth of its speed,
    with every diagnostic saying the GPU was busy. Hours went into optimising
    code around what was a power setting.
    """
    out: list[Check] = []

    smi = shutil.which("nvidia-smi")
    if not smi:
        return [Check("GPU", WARN, "nvidia-smi not found",
                      "without an NVIDIA GPU, set stt.device: cpu in config.yaml")]

    info = _run([smi, "--query-gpu=name,memory.used,memory.total,power.draw,"
                      "power.limit", "--format=csv,noheader,nounits"])
    if not info:
        return [Check("GPU", WARN, "nvidia-smi returned nothing")]

    parts = [p.strip() for p in info.splitlines()[0].split(",")]
    name, used, total = parts[0], parts[1], parts[2]
    try:
        used_mb, total_mb = float(used), float(total)
        free_gb = (total_mb - used_mb) / 1024
        status = OK if free_gb >= 5 else (WARN if free_gb >= 3 else FAIL)
        out.append(Check("GPU memory", status,
                         f"{free_gb:.1f} GB free of {total_mb/1024:.1f} GB "
                         f"({name})",
                         "" if status == OK else
                         "close other GPU users (LM Studio, browsers) — "
                         "Whisper needs ~1.5 GB on top of the LLM"))
    except ValueError:
        out.append(Check("GPU", WARN, info[:60]))

    # Power draw against the limit, which is the number that matters.
    try:
        draw, limit = float(parts[3]), float(parts[4])
        if limit > 0 and draw / limit < 0.25:
            out.append(Check("GPU power", WARN,
                             f"{draw:.0f} W of {limit:.0f} W available",
                             "idle is fine; but if the app feels slow, this is "
                             "the first thing to check"))
    except (ValueError, IndexError, ZeroDivisionError):
        pass

    return out


def check_battery() -> Check:
    out = _run(["powershell", "-NoProfile", "-Command",
                "(Get-CimInstance Win32_Battery).BatteryStatus"], timeout=8)
    if not out:
        return Check("Power", OK, "no battery — desktop or on mains")
    status = out.splitlines()[0].strip()
    # 1 = discharging. 2 = on AC. Everything else is a charging/charged state.
    if status == "1":
        return Check("Power", WARN, "running on battery",
                     "plug in. On battery this GPU has run at a fifth of its "
                     "speed while reporting 100% utilisation")
    return Check("Power", OK, "on mains")


def check_audio() -> list[Check]:
    try:
        import sounddevice as sd
        devices = sd.query_devices()
    except Exception as e:
        return [Check("Audio devices", FAIL, str(e)[:70],
                      "check that a sound device exists and drivers are loaded")]

    ins = [d for d in devices if d.get("max_input_channels", 0) > 0]
    outs = [d for d in devices if d.get("max_output_channels", 0) > 0]
    out = []
    if ins:
        try:
            default = devices[sd.default.device[0]]["name"]
        except Exception:
            default = ins[0]["name"]
        out.append(Check("Microphone", OK, f"{len(ins)} input(s), using {default}"))
    else:
        out.append(Check("Microphone", FAIL, "no input device",
                         "python main.py --list-devices"))
    out.append(
        Check("Speaker", OK, f"{len(outs)} output(s)") if outs
        else Check("Speaker", FAIL, "no output device")
    )
    return out


def check_cuda_dlls(cfg: Config) -> Check:
    """Whisper on CUDA needs cuBLAS and cuDNN, which torch used to supply."""
    if cfg.stt.device != "cuda":
        return Check("CUDA libraries", OK, f"not needed (device={cfg.stt.device})")
    missing = [n for n in ("nvidia.cublas", "nvidia.cudnn")
               if importlib.util.find_spec(n) is None]
    if missing:
        return Check("CUDA libraries", FAIL, ", ".join(missing) + " missing",
                     "pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
    return Check("CUDA libraries", OK, "cublas + cudnn present")


def check_disk(cfg: Config) -> list[Check]:
    out = []
    for label, path in (("models", cfg.models_root),
                        ("sessions", cfg.sessions_path)):
        target = path
        while not target.exists() and target.parent != target:
            target = target.parent
        try:
            free_gb = shutil.disk_usage(target).free / 1e9
        except OSError:
            continue
        need = 3 if label == "models" else 0.2
        status = OK if free_gb > need else WARN
        out.append(Check(f"Disk ({label})", status,
                         f"{free_gb:.1f} GB free on {target.anchor or target}",
                         "" if status == OK else
                         f"{label} needs room; change paths in config.yaml"))
    return out


def check_course(cfg: Config) -> Check:
    from . import curriculum
    course = curriculum.find(cfg.courses_path, cfg.tutor.course)
    if course is None:
        return Check("Course", WARN, "no textbook ingested",
                     'python scripts/ingest_textbook.py "path/to/book.pdf" '
                     "— without one the tutor uses the stock scenarios")
    return Check("Course", OK,
                 f"{len(course.chapters)} chapters — {course.title[:44]}")


def check_progress(cfg: Config) -> Check:
    from . import progress
    p = progress.build(cfg.sessions_path)
    if p.is_empty:
        return Check("History", OK, "no sessions yet — first run")
    return Check("History", OK,
                 f"{p.sessions} session(s), {len(p.vocab)} words, "
                 f"last chapter {p.last_chapter or '—'}")


# --- runner ---------------------------------------------------------------

def run_all(cfg: Config) -> list[Check]:
    checks: list[Check] = [check_python()]
    checks += check_packages()
    checks.append(check_cuda_dlls(cfg))
    checks += check_models(cfg)
    checks += check_llm(cfg)
    checks += check_audio()
    checks.append(check_battery())
    checks += check_power()
    checks += check_disk(cfg)
    checks.append(check_course(cfg))
    checks.append(check_progress(cfg))
    return checks


def report(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks) + 1
    lines = ["", "  Stammtisch — Diagnose", "  " + "─" * 62]
    for c in checks:
        mark = {OK: "  ok ", WARN: "warn ", FAIL: "FAIL "}[c.status]
        lines.append(f"  {mark} {c.name:<{width}} {c.detail}")
        if c.fix:
            lines.append(f"        {' ' * width} → {c.fix}")

    fails = [c for c in checks if c.status == FAIL]
    warns = [c for c in checks if c.status == WARN]
    lines.append("  " + "─" * 62)
    if fails:
        lines.append(f"  {len(fails)} problem(s) will stop the app from working.")
    elif warns:
        lines.append(f"  Ready. {len(warns)} thing(s) worth knowing about above.")
    else:
        lines.append("  Everything checks out.")
    lines.append("")
    return "\n".join(lines)
