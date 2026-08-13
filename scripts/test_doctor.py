"""The doctor is only worth having if it catches real problems.

A diagnostic that passes everything on a working machine proves nothing — the
whole value is in what it says when something is broken. So this deliberately
breaks the configuration in the ways it actually breaks in practice and checks
that the right thing fails, **with a fix attached**. A `FAIL` with no fix line
is only a nicer traceback.

Pure stdlib plus the app. `python scripts/test_doctor.py`
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import doctor  # noqa: E402
from app.config import load_config  # noqa: E402


def find(checks, prefix):
    return next((c for c in checks if c.name.lower().startswith(prefix.lower())),
                None)


def main() -> int:
    fails = 0

    def check(label, cond, extra=""):
        nonlocal fails
        fails += not cond
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        if not cond and extra:
            print(f"        {extra}")

    print("\nmissing model files:")
    cfg = load_config()
    with tempfile.TemporaryDirectory() as td:
        cfg.paths.models_root = str(Path(td) / "nope")
        results = doctor.check_models(cfg)
        c = results[0]
        check("a missing models directory fails", c.status == doctor.FAIL,
              f"got {c.status}: {c.detail}")
        check("and names the command that fixes it",
              "fetch_models" in c.fix, f"fix={c.fix!r}")

        # Directory exists but is empty: the VAD and both voices must fail
        # individually, not as one vague "models missing".
        (Path(td) / "empty").mkdir()
        cfg.paths.models_root = str(Path(td) / "empty")
        results = doctor.check_models(cfg)
        vad = find(results, "Silero")
        check("an empty directory still fails the VAD",
              vad is not None and vad.status == doctor.FAIL)
        voices = [c for c in results if c.name.startswith("Piper voice")]
        check("every configured voice is checked separately",
              len(voices) == len(cfg.tts.voices) and
              all(v.status == doctor.FAIL for v in voices),
              f"got {[(v.name, v.status) for v in voices]}")

    print("\nLLM unreachable:")
    cfg = load_config()
    # Port 1 is reserved and nothing will be listening on it.
    cfg.llm.base_url = "http://127.0.0.1:1/v1"
    results = doctor.check_llm(cfg)
    c = results[0]
    check("a dead port fails fast", c.status == doctor.FAIL, f"{c.detail}")
    check("and says how to start the server", "ollama serve" in c.fix,
          f"fix={c.fix!r}")
    check("it does not go on to guess about models", len(results) == 1)

    print("\nmodel not pulled:")
    cfg = load_config()
    cfg.llm.model = "definitely-not-pulled:99b"
    results = doctor.check_llm(cfg)
    server = results[0]
    if server.status != doctor.OK:
        print("  SKIP  no LLM server running — cannot test this path")
    else:
        c = find(results, "model (reply)")
        check("an unpulled model fails",
              c is not None and c.status == doctor.FAIL,
              f"got {c.status if c else None}")
        check("and gives the exact pull command",
              c is not None and c.fix == "ollama pull definitely-not-pulled:99b",
              f"fix={c.fix if c else None!r}")
        corrector = find(results, "model (corrector)")
        check("the corrector model is checked too", corrector is not None)

    print("\nCUDA libraries:")
    cfg = load_config()
    cfg.stt.device = "cpu"
    c = doctor.check_cuda_dlls(cfg)
    check("not required on cpu", c.status == doctor.OK, c.detail)

    # find_spec returns None for a missing leaf but *raises* when the parent
    # package is absent — which is exactly a machine with no CUDA, i.e. the
    # machine most likely to be running the doctor. This crashed CI.
    check("probing a package whose parent does not exist does not raise",
          doctor._has_module("nvidia.cublas") in (True, False))
    check("a wholly absent namespace is just False",
          doctor._has_module("definitely_not_installed.sub") is False)
    cfg.stt.device = "cuda"
    c = doctor.check_cuda_dlls(cfg)
    check("the cuda check returns a verdict either way",
          c.status in (doctor.OK, doctor.FAIL), c.status)

    print("\nno course ingested:")
    cfg = load_config()
    with tempfile.TemporaryDirectory() as td:
        cfg.paths.courses_dir = td
        c = doctor.check_course(cfg)
        check("a missing course warns rather than fails",
              c.status == doctor.WARN, f"got {c.status}")
        check("and points at the ingest script", "ingest_textbook" in c.fix)

    print("\nreport formatting:")
    checks = [
        doctor.Check("Something", doctor.OK, "fine"),
        doctor.Check("Broken", doctor.FAIL, "very broken", "do the thing"),
        doctor.Check("Iffy", doctor.WARN, "hmm", "consider this"),
    ]
    text = doctor.report(checks)
    check("failures are visible", "FAIL" in text)
    check("fixes are printed", "do the thing" in text)
    check("a failure is summarised at the end",
          "1 problem(s) will stop the app" in text, text.splitlines()[-2])
    clean = doctor.report([doctor.Check("All good", doctor.OK, "yes")])
    check("a clean run says so", "Everything checks out" in clean)
    warned = doctor.report([doctor.Check("Iffy", doctor.WARN, "hmm")])
    check("warnings alone still mean ready", "Ready." in warned)

    print("\nexit status:")
    check("bad is true only for FAIL",
          doctor.Check("x", doctor.FAIL).bad is True
          and doctor.Check("x", doctor.WARN).bad is False
          and doctor.Check("x", doctor.OK).bad is False)

    print("\nspeed:")
    import time
    cfg = load_config()
    t0 = time.perf_counter()
    doctor.run_all(cfg)
    elapsed = time.perf_counter() - t0
    # If running it is a decision, nobody runs it.
    check(f"whole diagnosis is quick ({elapsed:.1f}s)", elapsed < 15,
          "too slow to run casually")

    print(f"\n{'PASS' if not fails else 'FAIL'} — doctor\n")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
