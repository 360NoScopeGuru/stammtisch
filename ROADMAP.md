# Roadmap

Where this is going, and what it is deliberately not going to become.

Stammtisch started as a local alternative to a paid German-tutor app. The
question that shapes everything below is: **what would make someone use this
instead of Pingo, not because it is free, but because it is better?**

Three things can be better, and none of them are "a bigger model":

1. It follows **your** course book, so home practice and class line up.
2. It **remembers** you — across sessions, not just within one.
3. Everything stays on your machine, so there is no reason not to record
   everything about how you are doing.

---

## Done

| | |
|---|---|
| Voice loop | mic → Silero VAD → Whisper → LLM → Piper, streamed sentence by sentence |
| Two modes | mentor teaches in English, practice is German immersion |
| Voice routing | per-fragment language classification, one Piper voice each |
| Textbook as syllabus | `ingest_textbook.py` → ordered chapters, objectives, grammar, reference |
| Cross-session memory | recurring mistakes, vocabulary met, resume at last chapter |
| Repeat-after-me | word-by-word scoring of a spoken attempt |
| Web UI | transcript, corrections, lesson panel, drill view |
| Session persistence | written every turn, atomically, survives a kill |

## Next

### 1. Survivable by a stranger

The gap between "my script" and "a product" is what happens in the first two
minutes. Today: clone, run, get a traceback about a port, or silence because
Ollama is not running, or a slow session because the laptop is on battery.

- **`--doctor`** — one command that checks every precondition and says what to
  fix: LLM reachable, model pulled, Piper voices present, VAD present, CUDA
  DLLs resolvable, a microphone that exists, free VRAM, free disk, **and mains
  power**, which has cost more debugging time here than any bug.
- **First-run flow** — no course ingested, no models downloaded: say so and
  give the exact command, rather than failing later and elsewhere.
- **Errors that name the fix.** `cannot reach http://localhost:11434` should
  say "start Ollama with `ollama serve`".
- **CI** — GitHub Actions running the model-free suite on every push. Eleven
  scripts that need nothing but Python are worth running automatically.

### 2. Homework

The tutor talks. It does not yet make you *work*.

- Assign a small set of exercises from the current chapter — gap-fill,
  translation, sentence-building, a short written piece.
- You do them in your own time, in the web UI or the terminal.
- It marks them, explains what went wrong, and the mistakes join the same
  cross-session store that already tracks spoken errors.

**Honest limitation:** the workbook PDF has no answer key — the `Lösungen`
hits in it are the word "solutions" inside exercise instructions, not a marking
section. So grading is a model judging free-form German. That is fine for
"you used the dative here and should have used the accusative", and it is not
fine for anything that pretends to be an exam mark. Exercises with a
mechanically checkable answer (gap-fill from a word bank, conjugation tables)
should be checked mechanically, and only the open-ended ones sent to the model.

### 3. Learning that actually compounds

Current vocabulary review is "least recently seen", which is not spaced
repetition, it is a queue.

- **SM-2 scheduling** over the words and the recurring mistakes, so review
  intervals grow as things stick and collapse when they do not.
- **Chapter objectives marked complete** when demonstrated, so the book
  actually progresses instead of the learner deciding by feel.
- **Placement** — a short opening conversation that sets the level, rather
  than defaulting to A1 and hoping.

### 4. Your class's own vocabulary

The handouts in the user's course folder list nouns in `der` / `das` / `die`
columns — better A1 material than the textbook, because it is what the class
is actually using.

Blocked on a real problem: the column alignment is destroyed by text
extraction, and `pypdf`'s coordinates collapse in those docx-export PDFs, so a
row with a missing middle column cannot be read positionally. **Guessing is not
acceptable here** — a wrong article teaches wrong German, and the learner has
no way to know.

Planned approach: take the article the column order implies, ask the model
independently, and keep only the entries where the two agree. Everything else
is surfaced for a human to confirm rather than silently included.

### 5. Other languages

Everything German-specific is in known places: the `langid` word lists, the
Piper voice pair, the prompts, the CEFR descriptions. Pulling those into a
language pack would let the same machinery teach Spanish or French from any
ingested book.

Real reach, but it helps other people rather than the person building it, so
it sits behind the rest.

---

## Not doing

Saying no is most of what keeps a project finishable.

- **No cloud, no accounts, no telemetry.** The whole point is that it runs on
  your hardware and your transcripts never leave it.
- **No torch.** Whisper is CTranslate2, Piper and the VAD are onnxruntime.
  Torch drags in a mismatched `torchaudio` that breaks the VAD, and it has
  already cost this project one full debugging session.
- **No real pronunciation scoring** until it can be done honestly. Whisper
  returns text, not phonemes. Forced alignment plus a goodness-of-pronunciation
  model would need torch, so the drill feature says plainly what it does
  measure: whether a German recogniser heard the words you aimed at.
- **No browser audio.** The mic and speaker stay in Python. Routing PCM over a
  WebSocket would add latency to the exact path the whole design protects.
- **No model-driven control flow.** Mode switching was tried as a
  `[[PRACTICE]]` token and failed in both directions on `gemma3:12b`. Anything
  that must be reliable is a regex over the transcript or a deterministic
  check, not a request to a model.
- **No shipping a textbook.** Courses are built from a PDF the user already
  owns, and `courses/` is gitignored.

---

## How work here gets checked

Two rules, both learned the hard way in this project.

**Synthetic evidence does not transfer.** `test_pipeline.py` passed happily
through a session where Whisper labelled a third of the utterances Urdu and
transcribed them as such — Piper's German is too clean to reproduce the
failure even when deliberately wrecked with noise. Anything about real audio
has to be checked against real logs.

**Counting is not validating.** The textbook parser produced exactly the right
number of chapters, correctly titled, with every grammar list shifted one
chapter along. Structure checks have to assert relationships, not totals.
