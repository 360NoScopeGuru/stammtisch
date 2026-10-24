"""FastAPI + WebSocket front-end.

The browser is a *view*: it renders the event stream and sends messages and
controls. Audio never touches it — the mic and speaker stay in Python, where
the latency budget lives. Typed messages go over the socket like any other
control; files come up over HTTP, because a photographed worksheet is
megabytes and the socket is the conversation.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import shutil
import tempfile
from dataclasses import asdict
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

from . import (classlog, documents, handouts, homework, objectives, progress,
               srs, study)
from .config import Config
from .events import EventBus
from .session import ConversationRunner
from .tutor import Tutor

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# Files kept loaded for attaching. A session rarely needs more, and each
# holds its PDF in memory.
MAX_DOCUMENTS = 12


def _assignment(a: homework.Assignment | None) -> dict | None:
    if a is None:
        return None
    return {
        "id": a.id, "title": a.chapter_title, "level": a.level,
        "submitted": a.submitted, "marked": a.is_marked,
        "score": list(a.score), "comment": a.comment,
        "exercises": [
            {"kind": e.kind, "prompt": e.prompt, "hint": e.hint,
             # The key is shown only once marked: an answer visible in the
             # page source is an answer copied.
             "answer": e.answer if a.is_marked else ""}
            for e in a.exercises
        ],
        "answers": [asdict(x) for x in sorted(a.answers, key=lambda x: x.index)],
    }


def _card(c: srs.Card | None) -> dict | None:
    if c is None:
        return None
    return {"id": c.id, "kind": c.kind, "front": c.front, "source": c.source}


def create_app(cfg: Config) -> FastAPI:
    bus = EventBus()
    tutor = Tutor(cfg, bus=bus)
    runner = ConversationRunner(tutor, bus)

    # Uploaded files live here for the life of the server. Pages of the
    # learner's books, so not in the repository and not kept afterwards.
    uploads = Path(tempfile.mkdtemp(prefix="stammtisch-uploads-"))

    app = FastAPI(title="Stammtisch")
    app.state.bus = bus
    app.state.tutor = tutor
    app.state.runner = runner
    app.state.task = None

    @app.on_event("startup")
    async def _startup() -> None:
        async def guarded() -> None:
            try:
                await runner.run()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # keep the server alive to show the error
                log.exception("conversation loop died")
                bus.publish("fatal", message=str(e))

        app.state.task = asyncio.create_task(guarded())

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        runner.request_stop()
        task = app.state.task
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await tutor.aclose()
        shutil.rmtree(uploads, ignore_errors=True)

    async def body(request: Request) -> dict:
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(400, "expected a JSON body")
        if not isinstance(data, dict):
            raise HTTPException(400, "expected a JSON object")
        return data

    def resolve(refs) -> list[tuple[documents.Document, list[documents.Page]]]:
        out = []
        for ref in (refs or [])[:4]:
            if not isinstance(ref, dict):
                continue
            doc = tutor.documents.get(str(ref.get("id", "")))
            if doc is None:
                raise HTTPException(404, "That file is no longer loaded — "
                                         "attach it again.")
            try:
                out.append((doc, doc.select(str(ref.get("pages", "")))))
            except documents.DocumentError as e:
                raise HTTPException(400, str(e))
        return out

    # ---- pages -----------------------------------------------------------

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/state")
    async def state() -> JSONResponse:
        prog = tutor.progress
        course = tutor.course
        today = date.today()
        cards = srs.collect(cfg.study_path, prog)
        return JSONResponse(
            {
                "level": cfg.tutor.level,
                "scenario": tutor.scenario.key,
                "title": tutor.scenario.title,
                "voice_input": tutor.listening,
                "speak": tutor.speaking_aloud,
                "corrections": tutor.session.all_corrections,
                "vocab": tutor.session.all_vocab,
                "transcript": tutor.session.transcript,
                "course": None if not course else {
                    "title": course.title,
                    "level": course.level,
                    "current": tutor.chapter.number if tutor.chapter else None,
                    "chapters": [
                        {"number": c.number, "title": c.title,
                         "topics": c.topics, "grammar": c.grammar,
                         "can_do": c.can_do}
                        for c in course.chapters
                    ],
                },
                "goals": [asdict(g) for g in tutor.goals()],
                "progress": {
                    "sessions": prog.sessions,
                    "turns": prog.turns,
                    "minutes": round(prog.minutes),
                    "words": len(prog.vocab),
                    "last_seen": prog.last_seen,
                    "mistakes": [
                        {"wrong": m.wrong, "right": m.right, "count": m.count,
                         "explanation": m.explanation}
                        for m in prog.recurring(limit=8)
                    ],
                },
                "review": srs.counts(cards, srs.schedule(cfg.study_path), today),
                "classes": [asdict(e) for e in classlog.load_all(cfg.study_path)[-5:]],
                "decks": [{"slug": d.slug, "name": d.name, "counts": d.counts()}
                          for d in handouts.load_decks(cfg.study_path)],
                "homework": _assignment(homework.latest(cfg.homework_path)),
            }
        )

    # ---- files -----------------------------------------------------------

    @app.post("/api/upload")
    async def upload(request: Request, name: str = "upload") -> JSONResponse:
        """The file is the raw request body, streamed to disk.

        Not base64 in JSON: the learner's B1 course book is 215 MB, which
        as a base64 string in a JSON body is ~290 MB held in memory twice.
        On disk, a PDF is opened a page at a time and costs nothing more.
        """
        name = Path(name).name[:120] or "upload"
        doc_id = secrets.token_hex(6)
        path = uploads / f"{doc_id}{Path(name).suffix[:8]}"
        size = 0
        with path.open("wb") as fh:
            async for chunk in request.stream():
                size += len(chunk)
                if size > documents.MAX_PDF_BYTES:
                    fh.close()
                    path.unlink(missing_ok=True)
                    raise HTTPException(413, f"{name} is too large")
                fh.write(chunk)
        try:
            doc = await asyncio.to_thread(documents.load_path, path, name)
        except documents.DocumentError as e:
            path.unlink(missing_ok=True)
            raise HTTPException(400, str(e))
        while len(tutor.documents) >= MAX_DOCUMENTS:
            tutor.documents.pop(next(iter(tutor.documents)))
        tutor.documents[doc_id] = doc
        return JSONResponse({"id": doc_id, "name": doc.name, "kind": doc.kind,
                             "pages": doc.page_count})

    @app.get("/api/doc/{doc_id}/page/{number}")
    async def page_image(doc_id: str, number: int) -> Response:
        doc = tutor.documents.get(doc_id)
        if doc is None:
            raise HTTPException(404, "no such file")
        try:
            page = await asyncio.to_thread(doc.page, number)
        except documents.DocumentError as e:
            raise HTTPException(404, str(e))
        return Response(page.image, media_type=page.mime,
                        headers={"Cache-Control": "private, max-age=3600"})

    # ---- class notes -----------------------------------------------------

    @app.post("/api/class")
    async def log_class(request: Request) -> JSONResponse:
        data = await body(request)
        note = str(data.get("note", "")).strip()[:4000]
        attached = await asyncio.to_thread(resolve, data.get("attachments"))
        if not note and not attached:
            raise HTTPException(400, "Write a note or attach a page first.")
        try:
            day = date.fromisoformat(str(data.get("day") or date.today()))
        except ValueError:
            day = date.today()
        bus.publish("notice", message="Reading your class notes…")
        entry = await study.log_class(tutor.llm, tutor.course, note, attached, day)
        if entry is None:
            raise HTTPException(502, "The model did not return anything usable. "
                                     "Try again.")
        classlog.save(entry, cfg.study_path)
        bus.publish("class_logged", entry=asdict(entry),
                    current=tutor.chapter.number if tutor.chapter else None)
        return JSONResponse(asdict(entry))

    @app.delete("/api/class/{entry_id}")
    async def delete_class(entry_id: str) -> JSONResponse:
        if not classlog.delete(cfg.study_path, entry_id):
            raise HTTPException(404, "no such class note")
        return JSONResponse({"ok": True})

    # ---- handouts --------------------------------------------------------

    @app.post("/api/handout")
    async def read_handout(request: Request) -> JSONResponse:
        data = await body(request)
        attached = await asyncio.to_thread(resolve, data.get("attachments"))
        if not attached:
            raise HTTPException(400, "Attach the handout first.")
        doc, pages = attached[0]
        name = str(data.get("name") or Path(doc.name).stem)[:60]
        deck = await study.read_handout(
            tutor.llm, doc, name, pages,
            progress=lambda m: bus.publish("notice", message=m))
        if not deck.entries:
            raise HTTPException(422, "No nouns in article columns were found "
                                     "on those pages.")
        handouts.save_deck(deck, cfg.study_path)
        return JSONResponse({"slug": deck.slug, "name": deck.name,
                             "counts": deck.counts(),
                             "entries": [asdict(e) for e in deck.entries]})

    @app.get("/api/decks/{slug}")
    async def deck(slug: str) -> JSONResponse:
        d = handouts.find_deck(cfg.study_path, slug)
        if d is None:
            raise HTTPException(404, "no such deck")
        return JSONResponse({"slug": d.slug, "name": d.name, "counts": d.counts(),
                             "entries": [asdict(e) for e in d.entries]})

    @app.post("/api/decks/{slug}/{index}")
    async def decide(slug: str, index: int, request: Request) -> JSONResponse:
        data = await body(request)
        item = handouts.decide(cfg.study_path, slug, index,
                               str(data.get("status", "")),
                               de=data.get("de"), en=data.get("en"))
        if item is None:
            raise HTTPException(400, "could not update that entry")
        return JSONResponse(asdict(item))

    # ---- flashcards ------------------------------------------------------

    def next_card() -> tuple[srs.Card | None, dict]:
        cards = srs.collect(cfg.study_path, tutor.progress)
        states = srs.schedule(cfg.study_path)
        today = date.today()
        q = srs.queue(cards, states, today)
        return (q[0] if q else None), srs.counts(cards, states, today)

    @app.get("/api/review")
    async def review() -> JSONResponse:
        card, counts = next_card()
        return JSONResponse({"card": _card(card), "counts": counts})

    @app.post("/api/review")
    async def answer_card(request: Request) -> JSONResponse:
        data = await body(request)
        cards = {c.id: c for c in srs.collect(cfg.study_path, tutor.progress)}
        card = cards.get(str(data.get("card", "")))
        if card is None:
            raise HTTPException(404, "that card no longer exists")
        given = str(data.get("given", ""))[:200]
        result = srs.grade(card, given)
        srs.record(cfg.study_path, card, result.q, given)
        nxt, counts = next_card()
        return JSONResponse({"correct": result.correct, "note": result.note,
                             "answer": card.back, "next": _card(nxt),
                             "counts": counts})

    # ---- homework --------------------------------------------------------

    @app.get("/api/homework")
    async def get_homework() -> JSONResponse:
        return JSONResponse({"homework": _assignment(
            homework.latest(cfg.homework_path))})

    @app.post("/api/homework/new")
    async def new_homework() -> JSONResponse:
        bus.publish("notice", message="Writing your homework…")
        a = await tutor.set_homework()
        if a is None:
            raise HTTPException(502, "The model did not produce anything "
                                     "usable. Try again.")
        return JSONResponse({"homework": _assignment(a)})

    @app.post("/api/homework/submit")
    async def submit_homework(request: Request) -> JSONResponse:
        data = await body(request)
        current = homework.latest(cfg.homework_path)
        if current is None or str(data.get("id")) != current.id:
            raise HTTPException(409, "That homework is no longer the current one.")
        answers = [str(x)[:500] for x in (data.get("answers") or [])]
        current.submit(answers[:len(current.exercises)])
        homework.save(current, cfg.homework_path)
        bus.publish("notice", message="Marking…")
        marked = await tutor.mark_homework(current)
        # A wrong answer is now a mistake on record; let the tutor know.
        tutor.progress = progress.build(cfg.sessions_path, cfg.homework_path)
        return JSONResponse({"homework": _assignment(marked)})

    # ---- the conversation ------------------------------------------------

    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        await sock.accept()
        q = bus.subscribe()

        async def pump() -> None:
            """Bus -> browser."""
            while True:
                await sock.send_json(await q.get())

        try:
            # Catch the new client up on everything it missed.
            for event in bus.replay():
                await sock.send_json(event)
            runner._publish_config()
            runner.publish_goals()

            pump_task = asyncio.create_task(pump())
            try:
                while True:
                    msg = await sock.receive_json()
                    if not (isinstance(msg, dict) and msg.get("action")):
                        continue
                    # Stopping a reply cannot wait in the queue behind the
                    # reply it is meant to stop.
                    if msg["action"] == "interrupt":
                        runner.interrupt()
                    else:
                        runner.submit(msg)
            finally:
                pump_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pump_task

        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("websocket error")
        finally:
            bus.unsubscribe(q)

    return app


def serve(cfg: Config, host: str = "127.0.0.1", port: int = 8420) -> None:
    import uvicorn

    print(f"\n  Stammtisch UI -> http://{host}:{port}\n")
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")
