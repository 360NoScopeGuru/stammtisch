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
| Homework | closed exercises marked by rules, open ones by the model; web tab; wrong answers feed the mistake history |
| `--doctor` | every precondition checked, each failure carrying its fix — voice parts only when voice is on |
| CI | the model-free suite on every push |
| Text first | type or talk, either at any time; replies stream, and are read aloud or not |
| Reading pages | PDFs and photos shown to the model: read first, answer key solved apart, then taught |
| Class notes | what the class covered, from a note or a photo, checked, practised first |
| Handouts | der/das/die columns by glyph position, kept only where independent readings agree |
| Spaced repetition | SM-2 over words met, class vocabulary, handout nouns and recurring mistakes |
| Chapter goals | can-do statements checked by a task, or marked by hand; chapter done when all are |
| The whole course | A1–B2 joined as one; the B1 and B2 scans read by vision, by agreement |

## Next

### 1. Use it

Everything above has been tested against the real books and the real model,
but not yet through a month of evenings. Two weeks of real use will say more
about what to build next than any list here, and the sessions it leaves behind
are what the memory and the flashcards run on.

### 2. Things that are known to be rough

- **First reply to a page is slow** — ~20 s: reading it, then answering. The
  read could stream, or start the moment a file is attached rather than sent.
- **Homework questions are weak at times.** Marking is fair; the setting is
  the model's. Seeding exercises from the workbook's own prompts, now that its
  pages can be read, is the obvious fix.
- **A shared misreading on a scan gets through.** Voting cannot catch a slip
  every reading makes; the spelling pass flags some, not all. The B1 and B2
  contents were checked by hand once; another book would need the same.
- **End-of-chapter grammar summaries are missing for B1 and B2** — finding them
  in a scan means looking at every page. A reading of the pages just before
  each chapter's start would do it, now that start pages are known.

### 3. Placement

A short opening conversation that sets the level, rather than defaulting to
the level of the chapter. Less urgent than it was: with the course ingested,
the book the class is in says the level.

### 4. Other languages

Everything German-specific is in known places: the `langid` word lists, the
Piper voice pair, the prompts, the CEFR descriptions, the column headings the
book reader looks for. Pulling those into a language pack would let the same
machinery teach Spanish or French from any ingested book.

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
- **No model output trusted unchecked.** Every reading of a page — class
  notes, handouts, the book's contents — is checked against text where there is
  some, and against an independent reading where there is not, and refuses or
  flags rather than guess.
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
