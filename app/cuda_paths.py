"""Register the pip-installed CUDA DLLs on Windows.

CTranslate2 (which powers faster-whisper) links against cuBLAS and cuDNN but
does not know where pip put them. torch used to register these directories as a
side effect of being imported; with torch gone we have to do it ourselves, or
loading the Whisper model fails with:

    RuntimeError: Library cublas64_12.dll is not found or cannot be loaded

Import this module *before* faster_whisper.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_done = False


def register() -> list[str]:
    """Add every nvidia/*/bin directory to the DLL search path. Idempotent."""
    global _done
    if _done or sys.platform != "win32":
        return []

    import site

    roots: list[Path] = []
    for base in {*site.getsitepackages(), site.getusersitepackages()}:
        nvidia = Path(base) / "nvidia"
        if nvidia.is_dir():
            roots.append(nvidia)

    added: list[str] = []
    for nvidia in roots:
        for bin_dir in sorted(nvidia.glob("*/bin")):
            if not any(bin_dir.glob("*.dll")):
                continue
            try:
                os.add_dll_directory(str(bin_dir))
            except OSError:  # pragma: no cover - path vanished
                continue
            # PATH too: some loaders bypass the added-directory list.
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            added.append(str(bin_dir))

    if added:
        log.debug("registered CUDA dll dirs: %s", added)
    else:
        log.warning(
            "no nvidia/*/bin directories found — if Whisper fails to load, run: "
            "pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"
        )

    _done = True
    return added
