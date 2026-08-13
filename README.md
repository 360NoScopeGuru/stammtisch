# Stammtisch

[![tests](https://github.com/360NoScopeGuru/stammtisch/actions/workflows/tests.yml/badge.svg)](https://github.com/360NoScopeGuru/stammtisch/actions/workflows/tests.yml)

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
reading English produces gibberish, so every fragment is classified by
`app/langid.py` and sent to the matching voice.

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

## Teaching from your own textbook

Scenarios ("at the bakery", "office small talk") are fine for conversation but
they are not a syllabus. Nothing says what comes first, nothing says when
something has been learned, and a beginner ends up being taught whatever the
model happens to think of next.

You already have an order — your course book. Point Stammtisch at it:

```powershell
python scripts/ingest_textbook.py "path/to/Akademie Deutsch A1+ Band 1.pdf"
python main.py --list-chapters
python main.py --chapter 3
```

The tutor then teaches chapter 3, using chapter 3's vocabulary, chapter 3's
grammar and the set phrases the book itself prints — so what you practise at
home matches what your class is doing.

```
  1. LOS GEHT’S                      Personalpronomen, Verbkonjugation Präsens
  2. DEUTSCHE SPRACHE, SCHWERE …     Artikel, Plural, Komposita
  3. LECKER!                         Verben mit Akkusativobjekt, Nullartikel
  ...
```

What gets extracted per chapter: the title and section titles, the topic
keywords, the **can-do statements** (the book's own CEFR objectives), the
grammar it introduces, and the end-of-chapter grammar summary, which is handed
to the model as reference so it teaches the book's forms rather than inventing
its own.

The parse is heuristic — it is a PDF — so it is **checked structurally before
anything is written**, and refuses to write on any doubt. This matters more
than it sounds: an early version produced exactly the right number of chapters
with every grammar list shifted one chapter along, so the food chapter was
labelled with the previous chapter's articles and plurals. Counting things did
not catch it. `scripts/test_curriculum.py` now pins the alignment, and
`validate()` rejects any parse whose can-do statements are not contiguous text
on the page.

Currently this understands the *Akademie Deutsch* contents layout. Other books
will need their own parser; the failure is loud, not silent.

> Ingested courses land in `courses/`, which is **gitignored** — the content is
> copyrighted and this repository is public. `*.pdf` is ignored too.

## It remembers you

Sessions are aggregated into a picture of what you have actually done, rebuilt
from the saved session files every launch:

```powershell
python main.py --progress
```

```
  6 sessions · 84 turns · 152 min total
  Last session: 2026-08-12 (chapter 3)

  Keeps getting wrong:
    «der» → «das»   (6x)
      e.g. Ich sehe der Buch.
```

The tutor is given a short version of this, so it stops re-teaching «Guten
Morgen» every launch, picks up at the chapter you stopped on, and works a
recurring mistake back in when the conversation gives it an opening.

The interesting part is finding those mistakes. Grouping corrections by their
sentence finds nothing — nobody makes the same mistake in the same words twice.
Grouping by *the words that changed inside them* turns twenty scattered
corrections into "der → das, six times", which is a lesson. A one-off is not
promoted to a habit, and a sentence rewritten end to end is not a pattern at
all.

Progress is **derived from the sessions, never stored beside them**. A parallel
store would drift the first time a session was deleted or hand-edited.

## Repeat after me

Say *"let me try that"*, or press **Say it back**, and the tutor repeats its
last German phrase for you to say back. Your attempt is scored word by word:

```
Ich   hätte   gern   ein   B̶r̶ö̶t̶c̶h̶e̶n̶            80% · Close — try again
```

> Close. The word «Brötchen» did not quite come through — try that one again.

**This is not pronunciation scoring and is not sold as one.** Whisper returns
text, not phonemes; it cannot tell you your vowel was too far forward. Real
scoring needs forced alignment and a goodness-of-pronunciation model, which
means torch, which this project does not have.

What it does tell you is whether a German speech recogniser, listening in
German, heard the words you were aiming at. Weaker claim, genuinely useful: if
Whisper hears «Brötchen» when you say it, you said something close enough to be
understood.

The failure to design against is the **false negative** — telling someone their
German was wrong when the recogniser merely spelled it differently. So
«heisse»/«heiße» and «Brotchen»/«Brötchen» are folded together before
comparison, and the attempt is pinned to German rather than auto-detected: a
shaky first attempt otherwise gets read as English and scored as nonsense.

Three goes at one phrase, then it moves on. Drilling past that stops being
practice and becomes a wall.

## Homework

Talking is not studying. A tutor that only holds conversations leaves nothing
to do between sessions and never makes you produce German under your own steam,
which is where the gaps actually show.

```powershell
python main.py --homework new      # set work on the chapter you are on
python main.py --homework do       # answer it — no mic, no models needed
python main.py --homework          # see it again, with marks
```

```
  1. Fill the gap: Ich ___ Anna.
     → heiße
     ✓
  2. You meet someone for the first time. A) Auf Wiedersehen! B) Hallo! C) Guten Tag!
     → C) Guten Tag!
     ✓
  4. Complete the sentence: Er ist ____.
     ✗ left blank

  4 of 5 marked correct
```

### Closed and open, and why the split matters

**Closed** exercises have one right answer, known when the exercise is written,
so marking is a string comparison — **no model involved**. Gap-fills, choosing
an article, conjugating a verb, writing a number as a word.

**Open** exercises — translate this, write a short dialogue — have no possible
answer key, so a model has to judge them.

Everything that *can* be closed is, because model marking is the weak link: it
is inconsistent between runs and it invents faults to look useful. Only the
open exercises are ever sent to the LLM, and a model verdict can never overturn
a mechanical one.

The workbook that ships with the textbook has no answer key either — the
`Lösungen` in it are the word "solutions" inside exercise instructions, not a
marking section — so there is no shortcut available here.

**Marks are advisory.** This is practice, not an exam.

### The failure this is built around

An unfair mark is worse than no mark, because you cannot tell it from a real
one, and one of them teaches you to distrust every other. The first version
marked both of these wrong:

```
key "C"         you wrote "C) Guten Tag!"     ← a person answering a question
key "heiße"     you wrote "Ich heiße Anna"    ← filling the gap in context
```

Both now count as correct. Case, spacing, punctuation and umlaut spelling are
folded away; the expected answer is accepted anywhere inside your answer as
whole words, so «ist» is not found inside «Christian».

Model output is treated as untrusted throughout. Asked to mark exercise 4 the
model will happily return index 0 or 5, and an answer silently left unmarked
looks exactly like one the marker ignored — so when the indices do not line up
but the count does, they are matched positionally instead.

Homework is set on the chapter you are actually on, and includes an exercise
targeting whatever you keep getting wrong.

> Your answers live in `homework/`, which is gitignored.

## Requirements

- NVIDIA GPU with ~6 GB free VRAM
- An OpenAI-compatible LLM server — Ollama, LM Studio, or `llama.cpp --server`
- **Mains power.** See below — this matters more than anything else here.

## Plug the laptop in

On a laptop this is the single largest factor in whether the app feels usable.

Measured on an RTX 5080 Laptop, unplugged, at 21% battery, Windows "Silent"
power profile:

```
GPU utilisation : 96–100 %      (it really is on the GPU)
GPU power       : 33 W          (of a 175 W limit)
gemma3:12b      : ~5 tokens/sec
full reply      : 11–15 seconds
```

The GPU was pinned at full utilisation while drawing a fifth of its power
budget, so it ran at roughly a fifth of its speed. Nothing in the code can
compensate for that. Plug in and turn off any "Silent"/battery-saver profile
before concluding the app is slow.

`python scripts/bench_llm.py` reports time-to-first-sentence and full reply time
for any model, which is the honest way to check.

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

### When something is wrong

```powershell
python main.py --doctor
```

```
    ok  Python             3.12.10
    ok  packages           7 required imports present
    ok  CUDA libraries     cublas + cudnn present
    ok  Silero VAD         silero_vad.onnx
  FAIL  model (reply)      gemma3:12b not pulled
        → ollama pull gemma3:12b
  warn  Power              running on battery
        → plug in. On battery this GPU has run at a fifth of its speed
          while reporting 100% utilisation
```

Under two seconds, loads nothing, opens no microphone. Every failure carries
the command that fixes it — a `FAIL` with no fix line is just a nicer
traceback.

It checks the things that have actually gone wrong here: the LLM server being
down, the model not being pulled, Piper voices or the VAD missing, the CUDA
libraries that `torch` used to supply, free VRAM, free disk, whether a
microphone exists — and **whether the laptop is on battery**, which has cost
more debugging time in this project than any bug in the code.

`scripts/test_doctor.py` deliberately breaks the configuration and checks the
right thing fails with the right fix, because a diagnostic that only passes on
a working machine proves nothing.

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
| `test_segments.py` | segmentation, guillemet-aware splitting, merging | nothing |
| `test_langid.py` | language classification + every real routing regression | nothing |
| `test_intents.py` | mode-switch detection, including false positives | nothing |
| `test_drills.py` | repeat-after-me scoring, and every false negative | nothing |
| `test_homework.py` | fair marking, and untrusted model marks | nothing |
| `test_progress.py` | cross-session memory, recurring-mistake detection | nothing |
| `test_curriculum.py` | textbook parsing and chapter/grammar alignment | nothing |
| `test_corrections.py` | name protection, and the corrections that must survive it | nothing |
| `test_loop.py` | conversation loop, controls, echo drain, failure recovery | nothing |
| `test_persistence.py` | session survives `kill -9` mid-conversation | nothing |
| `test_web.py` | WebSocket delivery, backlog replay, controls | nothing |
| `test_doctor.py` | the diagnostics, against deliberately broken configs | nothing |
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
python main.py --chapter 3                      # chapter 3 of your textbook
python main.py --list-chapters
python main.py --progress                       # what you have covered so far
python main.py --homework new                   # set written work
python main.py --scenario baeckerei
python main.py --list-scenarios
python main.py --doctor                         # why isn't it working?
python main.py --list-devices                   # if the wrong mic is picked up
```

Ollama (or whichever server you configured) must be running first, or startup
fails at preflight with `cannot reach http://localhost:11434/v1`.

Talk. Stop talking. It replies. `Ctrl+C` ends the session and prints your review.

Sessions are written to `sessions/` as JSON plus an Anki-importable CSV, **after
every turn**. This used to happen only on a clean shutdown, which meant it never
happened at all: the app is normally ended with Ctrl+C or by closing the console,
so `sessions/` stayed empty after every real session and the review and Anki
export had never once produced a file. Writes are atomic, so a kill landing
mid-write cannot leave a half-parsed session behind.

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
- The current chapter's grammar and objectives, and a chapter switcher.
- Your recurring mistakes across every past session.
- **Say it back**: drill the tutor's last German phrase, scored word by word.
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
| `llm.model` | The biggest latency lever, and the biggest quality one. See [Choosing a model](#choosing-a-model) for measured numbers. |
| `llm.max_tokens` | 140 by default. With barge-in off, a long reply is enforced silence. |
| `llm.keep_alive` | Stops Ollama unloading between turns; a reload costs ~16s. |

## Voice routing

German goes to the German voice, English to the English voice. Getting this
wrong is not subtle — the wrong voice phonemises the text with the wrong rules
and the learner hears mush with no idea why.

Brackets split a segment as well as guillemets, because models gloss themselves
in them however firmly they are told not to. `gemma3:4b` produced
`«Guten Tag! Wie heißt du? (What is your name?)»` in *practice* mode, where
translating defeats the whole exercise. Splitting on the bracket at least sends
the English to the English voice instead of feeding it to the German one.

The first version trusted the model to wrap German in guillemets. That failed in
both directions, on both models tried:

| Model | Emitted | Result |
|---|---|---|
| `gemma3:12b` | `«Hallo! Wie geht es dir?» (Hello! How are you?)` | English gloss handled, but earlier variants put it *inside* the marks → English to the German voice |
| `gemma3:4b` | `«Guten Tag!» Wie geht es Ihnen heute?` | German *outside* the marks → German to the English voice |
| `gemma3:4b` | `(German voice) «Hallo!»` | stage direction read aloud |

So `app/langid.py` classifies each fragment by its own function words and
umlauts, and the guillemets are demoted to a tie-breaker for fragments too short
to call ("Ja."). Stage directions are stripped before synthesis. Adjacent
same-language fragments are merged, since Piper resets prosody per call and
splitting a sentence makes it choppy.

`scripts/test_langid.py` pins every one of the strings above.

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

## Speech recognition language

Whisper is told which language to expect, and this is mode-dependent.

In **practice** mode it is pinned to German. In **mentor** mode it auto-detects,
because the learner is speaking mostly English — and forcing German onto English
speech does not fail loudly. It produces confident German-shaped nonsense:

```
said:  "Hi, my name is Vamshee"
heard: "Bamsi Taran ist ein bisschen komisch"
```

which then reached the corrector, which duly "corrected" a phrase the learner
never said. Corrections now only run when the utterance was actually German.

Auto-detect is **restricted to `stt.detect_languages`** (`[de, en]`). Left
unrestricted it is far worse than it looks. From one real session's log:

```
ur 0.60   zh 0.53   ur 0.40   it 0.40   ur 0.43   ur 0.72   ar 0.90   id 0.49
```

Seven of twenty-one utterances came back as Urdu, Chinese, Arabic, Italian or
Indonesian — and were transcribed as such, so the tutor spent those turns
replying to gibberish. Note the confidences: 0.30, 0.37, 0.40. It was guessing.
Two seconds of an accented beginner is not enough signal to choose among 99
languages, but it is plenty to choose between two.

The restriction is nearly free. The first pass already computes the full
distribution, so a second pass runs only when the global argmax falls outside
the candidate set — exactly the case that was producing garbage anyway.

`scripts/probe_langdetect.py` measures this. Be aware that Piper's German is too
clean to reproduce the failure: synthesized speech is detected correctly even
when deliberately wrecked with noise and pitch shifting. The evidence for the
restriction comes from production logs, not from synthetic audio — which is
also why `test_pipeline.py` was passing throughout.

## Corrections never touch names

The corrector is a small local model, good at case endings and bad at knowing
what a name is. From a real session:

```
heard:      "Ich heiße Fahmshidharan."
corrected:  "Ich heiße Farshid Shidharan."
```

The learner said their own name correctly; Whisper mangled the spelling, and the
corrector "fixed" the mangling into a different name and presented it as a
German error.

Prompting alone cannot fix this — the model cannot distinguish a name it has
never seen from a misspelled noun. So `app/corrections.py` drops any correction
that changes a proper noun, using `learner.name` from `config.yaml` plus
anything following a naming verb («heiße», «Mein Name ist …»).

The filter is deliberately narrow, because dropping a genuine correction is
invisible — the learner simply never finds out they were wrong. Real manglings
of a name score 0.36–1.00 against it; ordinary German words that appear in
genuine corrections reach 0.55 («waren», «Name»). The fuzzy threshold sits at
0.60, above the noise, and the exact naming-verb rules do the real work.
`scripts/test_corrections.py` pins both directions, including the cases that
must **not** be filtered, like «Ich bin Student» → «Studentin».

**Set `learner.name` in `config.yaml`** to whatever you actually say when you
introduce yourself.

## The echo guard

Speaker bleed is the first thing that breaks this app on a laptop. A real
session produced exactly this:

```
assistant: "Hallo! Schön, dich zu sehen. Wie war denn dein Tag heute?"
user:      "Hallo, schön dich zu sehen. Wie war denn dein Tag heute?"   ← itself
```

There are two defences, in order of importance:

1. **The mic is gated shut while the tutor speaks** (when `barge_in` is off,
   the default), plus 250 ms afterwards to cover the audio device's buffer.
   This stops the echo at the source rather than recognising it later.
2. **A similarity guard** as backstop: any transcript ≥ `ECHO_SIMILARITY`
   (0.75) similar to what the tutor just said, within `ECHO_WINDOW_S` (2.5 s),
   is discarded. Separation is wide — verbatim echo scores 0.95–1.00, genuine
   speech 0.04–0.52, including a learner legitimately repeating the tutor's
   question back ("Und wie war dein Tag?", 0.52).

Turn `barge_in` on once you're on headphones; the guard stays active either way.

## Choosing a model

German quality varies far more than benchmarks suggest, and it is the single
biggest driver of whether this feels useful.

The default is **`gemma3:12b`**. Several features here are carried entirely by
the prompt — teaching the right textbook chapter, not translating itself during
practice — and instruction-following is exactly where the bigger model earns
its keep. `gemma3:4b` needed the prompt hardened twice and still glosses itself
in brackets during practice mode, which defeats the point of practising.

Measured on an RTX 5080 Laptop, on mains power, three mentor turns each:

| | time to first sentence | full reply | reply length |
|---|---|---|---|
| `gemma3:4b`  | 485 ms | 582 ms  | 85 chars |
| `gemma3:12b` | 929 ms | 2560 ms | 133 chars |

The full-reply column matters much less than it looks, because replies are
spoken sentence by sentence — the tutor starts talking at the first-sentence
mark, not the last. Under a second to first audio is fine. The reply-length
column is the real cost: 12b writes ~55% more, and with barge-in off every
extra word is silence you have to sit through.

`scripts/bench_llm.py gemma3:4b gemma3:12b` reproduces this. Run it on mains
power or the numbers are meaningless.

Others worth trying: **`mistral-nemo:12b`** (explicitly multilingual, natural
German) and **`qwen3:14b`** (strong, occasionally stiffer register). Swap
`llm.model`, run `test_prompt.py`, then talk to it for five minutes.

The corrector deliberately points at the **same** model. Ollama then holds one
copy rather than two — 7 GB instead of 11 on a 16 GB card — and the contention
is negligible: a correction firing concurrently with a reply costs **+56 ms**
(`scripts/probe_contention.py`), because the requests overlap rather than
queueing.

> With `12b` loaded, plus Whisper and Piper, this sits at about 13.5 GB of 16 GB.
> If you also run LM Studio or another local server, check `nvidia-smi` before
> assuming the app is at fault for being slow.

## Where this is going

See [ROADMAP.md](ROADMAP.md) — what is planned, and what this deliberately
will not become (no cloud, no accounts, no torch, no shipped textbooks).

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
