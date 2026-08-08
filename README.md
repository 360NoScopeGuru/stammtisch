# Stammtisch

A local German tutor you talk to out loud. It teaches in English, practises in
German, and reviews your mistakes afterwards. Everything runs on your own GPU —
no API keys, no subscription, no audio leaving the machine.

## Two modes

**Mentor** (the default) is a teacher. It explains in English, introduces German
a piece at a time, and asks you to say things back:

> That's a good place to start. «Guten Morgen» means "good morning" — try saying
> it back to me.

**Practice** is immersion. It stays in German and behaves like a conversation
partner rather than a teacher, correcting nothing out loud.

Say "can we practise now?" and it switches; say "I don't understand" or "explain
that" and it comes back. There are also buttons in the UI. Switching is decided
by `app/intents.py` reading your transcript, **not** by the model — see
[Mode switching](#mode-switching) for why.

Each language gets its own Piper voice, routed per phrase. A German voice
reading English produces gibberish, so the model marks German with guillemets
(`«…»`) and each fragment goes to the matching voice.

## How it works

```
mic ──► Silero VAD ──► Whisper ──► LLM (streaming) ──► sentence split ──► Piper ──► speaker
             endpoint      de→text        │                                   ▲
                                          └── async correction pass ──► transcript / Anki
```

The reply is spoken **sentence by sentence as it generates** — the tutor starts
talking while the model is still writing. That is the difference between a
conversation and a walkie-talkie.

Corrections deliberately run *off* the critical path, after the turn. The tutor
never interrupts you to correct grammar; mistakes surface in the session review.

## Requirements

- NVIDIA GPU with ~10 GB free VRAM (RTX 5080 Laptop: comfortable)
- An OpenAI-compatible LLM server — Ollama, LM Studio, or `llama.cpp --server`
- **Headphones.** There is no echo cancellation. On speakers, the tutor hears
  itself and interrupts itself. Or set `tutor.barge_in: false`.

## Setup

Use **Python 3.12**. Nothing here needs 3.13+, and wheels are more reliable.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Piper German voice + Silero VAD (~110 MB) -> D:/stammtisch-models
python scripts/fetch_models.py

# The LLM
ollama pull gemma3:12b

# Verify every local model before touching a microphone
python scripts/test_pipeline.py
```

> **Do not `pip install torch`.** This project has no torch dependency by
> design. Whisper runs on CTranslate2 and both Piper and the VAD run on
> onnxruntime. Installing torch pulls `torchaudio` at a mismatched version and
> breaks the VAD import with a `ctypes` DLL error.

Model weights go to `D:/stammtisch-models` because C: is nearly full. Change
`paths.models_root` in `config.yaml` to move them.

### Verifying without a microphone

Four scripts, none of which need a microphone:

| Script | Covers | Needs |
|---|---|---|
| `test_sentences.py` | streaming sentence splitter, German abbreviations | nothing |
| `test_segments.py` | language routing, guillemet-aware splitting | nothing |
| `test_intents.py` | mode-switch detection, including false positives | nothing |
| `test_loop.py` | conversation loop, controls, echo drain, failure recovery | nothing |
| `test_web.py` | WebSocket delivery, backlog replay, controls | nothing |
| `test_pipeline.py` | Piper → VAD → Whisper round trip, both voices | models |
| `test_prompt.py` | whether the live model obeys the prompt contract | LLM |

`test_pipeline.py` synthesizes German with Piper, pushes it through the VAD, and
transcribes it with Whisper. If it prints `PASS`, only the LLM server is left to
check.

`test_prompt.py` is worth running whenever you change models. It asks the live
LLM for one mentor turn and one practice turn, then checks the things the audio
pipeline depends on: German wrapped in guillemets, no English inside them, no
markdown leaking into speech, replies short enough to speak. A model that fails
these will sound broken in ways that are hard to trace back from audio.

## Run

```powershell
python main.py                                  # terminal, mentor mode, A1
python main.py --web                            # browser UI at :8420
python main.py --mode practice --level A2       # skip straight to German
python main.py --scenario baeckerei
python main.py --list-scenarios
python main.py --list-devices                   # if the wrong mic is picked up
```

Ollama (or whichever server you configured) must be running first, or startup
fails at preflight with `cannot reach http://localhost:11434/v1`.

Talk. Stop talking. It replies. `Ctrl+C` ends the session and prints your review.

Sessions are written to `sessions/` as JSON plus an Anki-importable CSV.

## The web UI

`python main.py --web` serves a transcript view at http://127.0.0.1:8420.

**Audio never touches the browser.** The mic and speaker stay in Python; the
page is a view over an event stream and a set of controls. Routing PCM through
a WebSocket would add latency to the exact path the whole design protects — and
browser mic capture would mean a second VAD and a second set of device
permissions for no gain.

What it gives you over the terminal:

- Corrections land **retroactively on the bubble that caused them**. They arrive
  a second or two after the turn, which in a scrolling terminal means they show
  up detached from the sentence they refer to.
- A running vocabulary list and session stats.
- Level and scenario switching mid-session. Changing scenario clears history and
  speaks the new opener; changing level keeps the conversation and just shifts
  register.

Both front-ends drive the same `ConversationRunner` and differ only in how they
render events, so the terminal stays fully usable.

## Tuning

The knobs that actually matter, in `config.yaml`:

| Setting | Effect |
|---|---|
| `vad.end_silence_ms` | **The main latency dial.** 450 ms is responsive; raise to 700+ if it cuts you off mid-thought. |
| `vad.threshold` | Raise toward 0.7 in a noisy room. |
| `tts.length_scale` | 1.15–1.3 makes the tutor speak slower. Good at A1/A2. |
| `tutor.mode` | `mentor` teaches in English; `practice` is German immersion. |
| `tts.voices` | One Piper voice per language. Swap `en` for `en_GB-alba-medium` if you prefer British English. |
| `llm.max_tokens` | Keep it low. Long tutor turns kill conversation flow. |
| `stt.model` | Drop to `distil-large-v3` or `medium` if VRAM gets tight. |

## Mode switching

Switching between teaching and practice is driven by a small regex layer over
your transcript (`app/intents.py`), not by the LLM.

The original design asked the model to emit a `[[PRACTICE]]` token. `gemma3:12b`
failed that in both directions: it ignored a direct *"can we practise now
please?"*, and once the instruction was strengthened it emitted the token
unprompted in reply to *"I want to learn how to introduce myself"* — which would
have dropped a beginner into roleplay before being taught anything.

Since the request is nearly always stated in plain words, reading the transcript
is both cheaper and more reliable. The detector is deliberately conservative:
missing a switch is a minor annoyance, but a false switch yanks you out of a
lesson mid-sentence. `scripts/test_intents.py` covers both directions, including
near-misses like *"I practise German every day"* that must **not** trigger.

Stray tokens are still stripped from replies so they can never be read aloud.

## The echo guard

Speaker bleed is the first thing that breaks this app on a laptop. A real
session produced exactly this:

```
assistant: "Hallo! Schön, dich zu sehen. Wie war denn dein Tag heute?"
user:      "Hallo, schön dich zu sehen. Wie war denn dein Tag heute?"   ← itself
```

Since there is no AEC, `app/session.py` compares each transcript against what
the tutor just said and discards anything ≥ `ECHO_SIMILARITY` (0.75) within
`ECHO_WINDOW_S` (2.5 s). Measured separation is wide — verbatim echo scores
0.95–1.00, genuine speech 0.04–0.52 — including the case where the learner
legitimately repeats the tutor's question back ("Und wie war dein Tag?", 0.52).

`barge_in` now defaults to **false** for the same reason. Turn it on once you're
on headphones; the guard stays active either way as a backstop.

German quality varies a lot more than benchmarks suggest, and it is the single
biggest driver of whether this feels useful. Worth A/B-ing yourself:

- **`gemma3:12b`** — strong European-language coverage. Good default.
- **`mistral-nemo:12b`** — explicitly multilingual, natural-sounding German.
- **`qwen3:14b`** — strong, occasionally stiffer register.

Swap `llm.model` and talk to each for five minutes. You will hear the difference
faster than any benchmark will tell you.

## Known limits

- **No pronunciation scoring.** Whisper returns text, not phonemes, so the app
  can tell you *what* it heard but not how close your vowels were. Real scoring
  needs forced alignment (wav2vec2 + a GOP model) — a later phase.
- **No true echo cancellation.** On laptop speakers the tutor's voice re-enters
  the mic, and Whisper transcribes it cleanly enough that it enters the history
  as a user turn — the tutor starts answering itself. There is a guard for this
  (see below), but headphones remain the real fix.
- **First transcription after startup costs ~20s** of CUDA kernel setup. `Tutor.preflight()`
  pays this during warm-up so it never lands mid-conversation — don't remove it.
- Whisper will occasionally "hear" fluent German where you mumbled — it is a
  strong language model and it guesses. The corrector is told to ignore likely
  transcription artefacts, which helps but does not eliminate this.
