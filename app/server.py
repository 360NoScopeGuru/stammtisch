"""FastAPI + WebSocket front-end.

The browser is a *view*: it renders the event stream and sends control
messages. Audio never touches it — the mic and speaker stay in Python, where
the latency budget lives.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from .config import Config
from .events import EventBus
from .session import ConversationRunner
from .tutor import Tutor

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app(cfg: Config) -> FastAPI:
    bus = EventBus()
    tutor = Tutor(cfg, bus=bus)
    runner = ConversationRunner(tutor, bus)

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

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/state")
    async def state() -> JSONResponse:
        return JSONResponse(
            {
                "level": cfg.tutor.level,
                "scenario": tutor.scenario.key,
                "corrections": tutor.session.all_corrections,
                "vocab": tutor.session.all_vocab,
                "transcript": tutor.session.transcript,
            }
        )

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

            pump_task = asyncio.create_task(pump())
            try:
                while True:
                    msg = await sock.receive_json()
                    if isinstance(msg, dict) and msg.get("action"):
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
