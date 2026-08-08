"""Silero VAD on onnxruntime — no torch, no torchaudio.

The silero-vad pip package imports torchaudio at module scope, which drags in
~3 GB of torch for a 2 MB model. Whisper runs on CTranslate2 and Piper runs on
onnxruntime, so torch was this project's only reason to exist. We load the ONNX
graph directly instead.

Model signature (silero_vad.onnx, v5+):
    input  [batch, 576]     float32   — 64 context samples + 512 new samples
    state  [2, batch, 128]  float32   — carried across frames
    sr     []               int64
  ->
    output [batch, 1]       float32   — speech probability
    stateN [2, batch, 128]  float32

The 64-sample context prefix is mandatory and easy to miss: the graph accepts a
bare 512-sample input without complaint and silently returns near-zero for
obvious speech. We keep the tail of each frame and prepend it to the next.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)

FRAME_SAMPLES = 512   # hop: how much new audio we consume per call (32 ms @ 16 kHz)
CONTEXT_SAMPLES = 64  # mandatory lookback prefix; total model input is 576
_STATE_SHAPE = (2, 1, 128)


class SileroVad:
    def __init__(self, model_path: str | Path, sample_rate: int = 16000) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Silero VAD model not found: {model_path}\n"
                f"Run: python scripts/fetch_models.py"
            )

        opts = ort.SessionOptions()
        # One frame every 32 ms — threads cost more in contention than they save.
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3

        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self.session.get_inputs()}
        # Must be a 0-d ndarray — onnxruntime rejects a bare numpy scalar.
        self.sample_rate = np.array(sample_rate, dtype=np.int64)
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        log.info("silero vad loaded (onnxruntime, cpu)")

    def reset(self) -> None:
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Speech probability for one 512-sample float32 frame."""
        if frame.shape[-1] != FRAME_SAMPLES:
            raise ValueError(
                f"expected {FRAME_SAMPLES} samples, got {frame.shape[-1]}"
            )
        frame = frame.astype(np.float32, copy=False)

        inputs = {
            "input": np.concatenate([self._context, frame]).reshape(1, -1),
            "state": self._state,
        }
        # The 16 kHz-only export drops the `sr` input.
        if "sr" in self._input_names:
            inputs["sr"] = self.sample_rate

        out, self._state = self.session.run(None, inputs)
        self._context = frame[-CONTEXT_SAMPLES:].copy()
        return float(out[0][0])
