"""Exercise the web layer without loading models or opening the mic.

Stubs out the Tutor, the conversation loop and the model, then drives the real
server: event delivery, backlog replay for a late-joining browser, control
round-trips, file uploads, class notes, handout decks, flashcard review and
homework. Personal data goes to temporary folders, never the real ones.
`python scripts/test_web.py`
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app import homework, server  # noqa: E402
from app.config import load_config  # noqa: E402
from app.events import QUEUE_MAX, EventBus  # noqa: E402
from app.progress import Mistake, Progress, Word  # noqa: E402
from app.tutor import Tutor  # noqa: E402


class FakeLlm:
    """Answers each kind of request the way a well-behaved model would."""

    def __init__(self):
        self.calls = []

    async def complete_json(self, messages, **kw):
        system = messages[0]["content"]
        self.calls.append(system[:40])
        if "class handout" in system:          # reading the page
            return {"nouns": [{"noun": "Tisch", "article": "der"},
                              {"noun": "Lampe", "article": "die"},
                              {"noun": "Buch", "article": "der"}]}
        if "definite article" in system:       # the separate lexicon check
            return {"nouns": [{"noun": "Tisch", "article": "der", "en": "table"},
                              {"noun": "Lampe", "article": "die", "en": "lamp"},
                              {"noun": "Buch", "article": "das", "en": "book"}]}
        if "record of their German class" in system:
            return {"chapter": 7, "pages": "60", "summary": "Ordering food.",
                    "topics": ["Essen"], "grammar": ["Akkusativ"],
                    "vocab": [{"de": "das Brötchen", "en": "bread roll"},
                              {"de": "der Zauberstab", "en": "magic wand"}],
                    "phrases": ["Ich hätte gern"]}
        return None

    async def aclose(self):
        pass


class StubSession:
    transcript: list = []
    all_corrections: list = []
    all_vocab: list = []

    def summary(self) -> str:
        return "stub summary"


class StubTutor:
    """Only what create_app / _publish_config / /api/state actually touch."""

    def __init__(self, cfg, bus=None):
        self.cfg = cfg
        self.bus = bus or EventBus()
        self.scenario = types.SimpleNamespace(
            key="baeckerei", title="At the bakery",
            opener="Guten Morgen!", intro="Today we go to the bakery.",
        )
        self.session = StubSession()
        self.levels: list[str] = []
        self.scenarios: list[str] = []
        self.modes: list[str] = []
        self.chapters: list[int] = []
        self.last_german = "Ich hätte gern ein Brötchen"
        self.chapter = types.SimpleNamespace(number=3)
        self.course = types.SimpleNamespace(
            title="Akademie Deutsch A1+", level="A1",
            listing=lambda: "   3. LECKER!",
            chapters=[types.SimpleNamespace(
                number=3, title="LECKER!", topics=["Essen"],
                grammar=["Nullartikel"], can_do=["Bestellungen aufgeben"])],
        )
        self.progress = Progress(
            sessions=4, turns=51, minutes=88.0, last_seen="2026-08-12",
            vocab={"danke": Word("danke", meaning="thanks"),
                   "bitte": Word("bitte")},
            mistakes=[Mistake(wrong="der", right="das", count=6,
                              explanation="Buch is neuter",
                              example="Ich sehe der Buch.",
                              corrected="Ich sehe das Buch.")],
        )
        self.documents = {}
        self.llm = FakeLlm()
        self.listening = False
        self.speaking_aloud = False
        self.interrupts = 0

    def goals(self):
        return []

    def interrupt(self):
        self.interrupts += 1

    # The real marking path: with every exercise closed it never needs the
    # model, which is exactly the property under test.
    mark_homework = Tutor.mark_homework

    def set_chapter(self, n):
        self.chapters.append(n)
        return False  # skip the opener path, which would need audio

    @property
    def mode(self):
        return self.cfg.tutor.mode

    def set_mode(self, v):
        self.modes.append(v)
        return False  # skip the opener path, which would need audio

    def set_level(self, v):
        self.levels.append(v)
        self.cfg.tutor.level = v
        return True

    def set_scenario(self, v):
        self.scenarios.append(v)
        return False  # skip the opener path, which would need audio

    async def aclose(self):
        pass


def _png() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), "white").save(buf, "PNG")
    return buf.getvalue()


def _pdf(pages: int = 3) -> bytes:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument.new()
    for _ in range(pages):
        pdf.new_page(595, 842)
    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()


def main() -> int:
    cfg = load_config()
    # Never the learner's real records.
    cfg.paths.study_dir = tempfile.mkdtemp()
    cfg.paths.homework_dir = tempfile.mkdtemp()
    failures = 0

    def check(label, cond, extra=""):
        nonlocal failures
        failures += not cond
        print(f"  {'PASS' if cond else 'FAIL'}  {label}{'' if cond else '  ' + extra}")

    # Neutralise the parts that need hardware.
    server.Tutor = StubTutor
    orig_run = server.ConversationRunner.run

    async def noop_run(self):
        return None

    server.ConversationRunner.run = noop_run

    try:
        app = server.create_app(cfg)
        with TestClient(app) as client:
            tutor = app.state.tutor
            bus = app.state.bus

            r = client.get("/")
            check("GET / serves the page", r.status_code == 200
                  and "Stammtisch" in r.text, f"status={r.status_code}")

            r = client.get("/api/state")
            check("GET /api/state returns JSON", r.status_code == 200
                  and "level" in r.json())

            # The course and the cross-session history are true before the
            # first turn, so the page fetches them rather than waiting for an
            # event that would never arrive on a quiet session.
            s = r.json()
            check("state carries the course", bool(s.get("course"))
                  and s["course"]["current"] == 3, f"course={s.get('course')}")
            check("state carries chapter detail",
                  s["course"]["chapters"][0]["grammar"] == ["Nullartikel"])
            check("state carries cross-session progress",
                  s.get("progress", {}).get("sessions") == 4
                  and s["progress"]["words"] == 2, f"progress={s.get('progress')}")
            check("recurring mistakes reach the browser",
                  s["progress"]["mistakes"][0]["wrong"] == "der"
                  and s["progress"]["mistakes"][0]["count"] == 6)

            # Publish BEFORE connecting: a late browser must still see it.
            bus.publish("user_turn", text="Ich hätte gern ein Brot.")

            with client.websocket_connect("/ws") as ws:
                seen, deadline = [], 25
                cfgs = 0
                while len(seen) < 1 or cfgs < 1:
                    if deadline <= 0:
                        break
                    e = ws.receive_json()
                    deadline -= 1
                    if e["type"] == "user_turn":
                        seen.append(e)
                    elif e["type"] == "config":
                        cfgs += 1

                check("backlog replayed to a late client",
                      any("Brot" in e.get("text", "") for e in seen))
                check("config sent on connect", cfgs >= 1)

                # Live event after connect.
                bus.publish("tutor_turn", text="Gerne!", latency_ms=880)
                live = None
                for _ in range(20):
                    e = ws.receive_json()
                    if e["type"] == "tutor_turn":
                        live = e
                        break
                check("live event delivered", live is not None
                      and live["text"] == "Gerne!")
                check("latency travels with the turn",
                      bool(live) and live.get("latency_ms") == 880)

                # Corrections arriving late must carry their payload intact.
                bus.publish("corrections", utterance="ein Brot",
                            items=[{"original": "ein Brot",
                                    "corrected": "ein Brot, bitte",
                                    "explanation": "politeness"}],
                            vocab=["die Bäckerei"])
                corr = None
                for _ in range(20):
                    e = ws.receive_json()
                    if e["type"] == "corrections":
                        corr = e
                        break
                check("corrections delivered with items+vocab",
                      bool(corr) and corr["items"][0]["corrected"] == "ein Brot, bitte"
                      and corr["vocab"] == ["die Bäckerei"])

                # Browser -> server controls. Do NOT receive_json() here: the
                # server has nothing queued and TestClient's recv blocks.
                ws.send_json({"action": "say", "value": "Hallo!",
                              "attachments": [{"id": "x", "pages": "2"}]})
                # Stopping a reply is not queued behind the reply.
                ws.send_json({"action": "interrupt"})
                ws.send_json({"action": "set_level", "value": "B2"})
                ws.send_json({"action": "set_scenario", "value": "arzttermin"})
                ws.send_json({"action": "set_mode", "value": "practice"})
                ws.send_json({"action": "set_chapter", "value": 5})

                # A drill result must survive the wire with its per-word
                # verdicts: the score alone tells the learner nothing they can
                # act on, the words are the whole feature.
                bus.publish("drill_result", target="Ich hätte gern ein Brötchen",
                            heard="Ich hätte gern ein Brot", score=0.8,
                            verdict="close",
                            words=[{"target": "Brötchen", "heard": "Brot",
                                    "status": "wrong"}])
                drill = None
                for _ in range(20):
                    e = ws.receive_json()
                    if e["type"] == "drill_result":
                        drill = e
                        break
                check("drill result delivered with per-word verdicts",
                      bool(drill) and drill["verdict"] == "close"
                      and drill["words"][0]["status"] == "wrong"
                      and drill["words"][0]["heard"] == "Brot")
                # Round-trip a publish to prove both sends were processed.
                bus.publish("ping")
                for _ in range(20):
                    if ws.receive_json()["type"] == "ping":
                        break

            # We replaced run(), so nothing is draining the control queue.
            # Apply them by hand to verify they were enqueued correctly.
            runner = app.state.runner
            queued = []
            while not runner.controls.empty():
                queued.append(runner.controls.get_nowait())
            check("typed message queued with its attachments",
                  queued and queued[0]["action"] == "say"
                  and queued[0]["attachments"][0]["pages"] == "2",
                  f"queued={queued[:1]}")
            check("interrupt applied at once, not queued",
                  tutor.interrupts == 1
                  and all(m["action"] != "interrupt" for m in queued))
            queued = [m for m in queued if m["action"] != "say"]
            asyncio.run(_apply_all(runner, queued))

            check("set_level control reached the tutor", "B2" in tutor.levels,
                  f"got {tutor.levels}")
            check("set_scenario control reached the tutor",
                  "arzttermin" in tutor.scenarios, f"got {tutor.scenarios}")
            check("set_chapter control reached the tutor",
                  tutor.chapters == [5], f"chapters={tutor.chapters}")
            check("set_mode control reached the tutor",
                  "practice" in tutor.modes, f"got {tutor.modes}")

            # Bus must not wedge when a subscriber stops draining.
            q = bus.subscribe()
            for i in range(QUEUE_MAX + 300):
                bus.publish("spam", i=i)
            check("bus survives a stalled subscriber", q.qsize() <= QUEUE_MAX,
                  f"qsize={q.qsize()}")
            bus.unsubscribe(q)

            # A streamed reply is hundreds of fragments, superseded by the
            # finished turn. Replaying them would rebuild it word by word.
            bus.publish("tutor_delta", text="Hal")
            check("streamed fragments are not replayed",
                  all(e["type"] != "tutor_delta" for e in bus.replay()))

            # ---- files ----------------------------------------------------
            r = client.post("/api/upload", params={"name": "board.png"},
                            content=_png())
            img = r.json() if r.status_code == 200 else {}
            check("an image uploads", img.get("kind") == "image"
                  and img.get("pages") == 1, r.text[:200])
            r = client.post("/api/upload", params={"name": "blatt.pdf"},
                            content=_pdf(3))
            pdf = r.json() if r.status_code == 200 else {}
            check("a PDF uploads with its page count",
                  pdf.get("kind") == "pdf" and pdf.get("pages") == 3, r.text[:200])
            r = client.post("/api/upload", params={"name": "notes.txt"},
                            content=b"just text")
            check("a file that is neither is refused, with a reason",
                  r.status_code == 400 and "PDF or an image" in r.text)
            r = client.post("/api/upload", params={"name": "x.pdf"}, content=b"")
            check("an empty upload is refused", r.status_code == 400)
            r = client.post("/api/upload", params={"name": "../../evil.pdf"},
                            content=_pdf(1))
            check("a path in the file name goes nowhere",
                  r.status_code == 200 and r.json()["name"] == "evil.pdf")
            r = client.get(f"/api/doc/{pdf.get('id')}/page/2")
            check("a PDF page renders for the preview",
                  r.status_code == 200 and r.headers["content-type"].startswith("image/"))
            r = client.get(f"/api/doc/{pdf.get('id')}/page/9")
            check("a page that does not exist is a 404", r.status_code == 404)

            # ---- class notes ----------------------------------------------
            # The note names chapter 3; the model says 7. What the learner
            # typed wins, and a chapter not in the book would be dropped
            # anyway. «Zauberstab» is in neither the note nor any text, but
            # the note is too short to check against, so it is kept unchecked.
            r = client.post("/api/class", json={
                "note": "Kapitel 3, Seite 44-45: im Café bestellen",
                "attachments": [{"id": img.get("id"), "pages": ""}],
                "day": "2026-10-21"})
            entry = r.json() if r.status_code == 200 else {}
            check("a class note is read and saved", r.status_code == 200, r.text[:200])
            check("the chapter the learner typed beats the model's",
                  entry.get("chapter") == 3, f"chapter={entry.get('chapter')}")
            check("the page range the learner typed is kept",
                  entry.get("pages") == "44-45", f"pages={entry.get('pages')}")
            check("words from a photo are marked unchecked",
                  entry.get("vocab") and all(w["checked"] is None
                                             for w in entry["vocab"]))
            r = client.post("/api/class", json={"note": "", "attachments": []})
            check("an empty class note is refused", r.status_code == 400)
            s = client.get("/api/state").json()
            check("the class shows in the state",
                  s["classes"] and s["classes"][-1]["chapter"] == 3)

            # ---- handouts --------------------------------------------------
            # Page reading and lexicon agree on Tisch and Lampe; they
            # disagree on Buch, which must wait for the learner.
            r = client.post("/api/handout", json={
                "attachments": [{"id": img.get("id"), "pages": ""}],
                "name": "Nomen 1"})
            deck = r.json() if r.status_code == 200 else {}
            by = {e["de"].split(" ", 1)[1]: e for e in deck.get("entries", [])}
            check("agreeing readings are confirmed",
                  by.get("Tisch", {}).get("status") == "confirmed"
                  and by.get("Lampe", {}).get("status") == "confirmed",
                  json.dumps(deck)[:300])
            check("a disagreement waits for the learner",
                  by.get("Buch", {}).get("status") == "pending"
                  and by["Buch"]["readings"] == {"vision": "der", "lexicon": "das"})
            check("the meaning comes along",
                  by.get("Tisch", {}).get("en") == "table")
            idx = [e["de"].split(" ", 1)[1] for e in deck["entries"]].index("Buch")
            r = client.post(f"/api/decks/{deck['slug']}/{idx}",
                            json={"status": "confirmed", "de": "das Buch"})
            check("the learner's decision is saved",
                  r.status_code == 200 and r.json()["de"] == "das Buch")
            r = client.post(f"/api/decks/{deck['slug']}/99",
                            json={"status": "confirmed"})
            check("a decision on a missing entry is refused", r.status_code == 400)

            # ---- flashcards ------------------------------------------------
            r = client.get("/api/review")
            first = r.json()
            check("a card is offered", first.get("card") is not None, r.text[:200])
            check("mistakes come first",
                  first["card"]["kind"] == "mistake", f"card={first.get('card')}")
            check("the answer is not sent with the question",
                  "Ich sehe das Buch" not in json.dumps(first["card"]))
            r = client.post("/api/review", json={"card": first["card"]["id"],
                                                 "given": "ich sehe das buch"})
            res = r.json()
            check("a right answer is marked right", res.get("correct") is True,
                  r.text[:200])
            # Find the Tisch card and give the wrong article.
            seen_ids, tisch = set(), None
            nxt = res.get("next")
            while nxt and nxt["id"] not in seen_ids:
                seen_ids.add(nxt["id"])
                if nxt["front"] == "table":
                    tisch = nxt
                    break
                nxt = client.post("/api/review", json={
                    "card": nxt["id"], "given": "x"}).json().get("next")
            check("handout words become cards", tisch is not None)
            if tisch:
                res = client.post("/api/review", json={"card": tisch["id"],
                                                       "given": "die Tisch"}).json()
                check("a wrong article is wrong", res["correct"] is False
                      and "article" in res["note"], res["note"])
            r = client.post("/api/review", json={"card": "nope", "given": "x"})
            check("an unknown card is a 404", r.status_code == 404)

            # ---- homework --------------------------------------------------
            a = homework.parse_generated(
                {"exercises": [
                    {"kind": "closed", "prompt": "Fill the gap: Ich ___ Anna.",
                     "answer": "heiße"},
                    {"kind": "closed", "prompt": "Fill the gap: Du ___ Tom.",
                     "answer": "heißt"}]},
                chapter=3, chapter_title="Kapitel 3", level="A1")
            homework.save(a, cfg.homework_path)
            h = client.get("/api/homework").json()["homework"]
            check("homework reaches the page", h and len(h["exercises"]) == 2)
            check("the answer key is not sent before marking",
                  all(e["answer"] == "" for e in h["exercises"]))
            r = client.post("/api/homework/submit",
                            json={"id": a.id, "answers": ["heisse", "heißen"]})
            h = r.json().get("homework") or {}
            check("handing in marks it", h.get("marked") and h["score"] == [1, 2],
                  r.text[:300])
            check("the key is shown once marked",
                  h.get("exercises") and h["exercises"][0]["answer"] == "heiße")
            check("the homework mistake reaches progress",
                  any(m.wrong == "heißen" and m.right == "heißt"
                      for m in tutor.progress.mistakes),
                  f"mistakes={[(m.wrong, m.right) for m in tutor.progress.mistakes]}")
            r = client.post("/api/homework/submit",
                            json={"id": "old", "answers": []})
            check("handing in a stale homework is refused", r.status_code == 409)

    finally:
        server.ConversationRunner.run = orig_run

    print(f"\n{'PASS' if not failures else 'FAIL'} — web layer\n")
    return 1 if failures else 0


async def _apply_all(runner, msgs):
    for m in msgs:
        await runner._apply_control(m)


if __name__ == "__main__":
    raise SystemExit(main())
