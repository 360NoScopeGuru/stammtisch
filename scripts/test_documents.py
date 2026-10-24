"""PDFs and images into pages for the model. No model needed.

`python scripts/test_documents.py`
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from app import documents  # noqa: E402
from app.documents import DocumentError  # noqa: E402
from pdfmaker import make_pdf  # noqa: E402

failures = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global failures
    failures += not cond
    print(f"  {'PASS' if cond else 'FAIL'}  {label}"
          + (f"\n        {detail}" if detail and not cond else ""))


def raises(fn, *a) -> str:
    try:
        fn(*a)
    except DocumentError as e:
        return str(e)
    return ""


def png(size=(300, 200), mode="RGB", colour="white") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, colour).save(buf, "PNG")
    return buf.getvalue()


print("\npage ranges:")
check("blank means every page", documents.parse_pages("", 4) == [1, 2, 3, 4])
check("ranges and lists", documents.parse_pages("1, 3-4", 5) == [1, 3, 4])
check("an en dash works", documents.parse_pages("2–3", 5) == [2, 3])
check("reversed ranges are put right", documents.parse_pages("4-2", 5) == [2, 3, 4])
check("repeats are dropped", documents.parse_pages("2,2,1-2", 5) == [2, 1])
check("past the end is an error that says so",
      "outside 1-5" in raises(documents.parse_pages, "6", 5))
check("nonsense is an error that says how",
      "e.g." in raises(documents.parse_pages, "chapter two", 5)
      or "like" in raises(documents.parse_pages, "chapter two", 5))

print("\nPDFs:")
pdf = make_pdf([[(72, 760, f"Seite {n}: Ich heiße Anna.")] for n in range(1, 9)])
doc = documents.load(pdf, "blatt.pdf")
check("a PDF is recognised", doc.kind == documents.PDF and doc.page_count == 8)
check("pages render only when asked for", doc.pages == {})
page = doc.page(3)
check("a page renders to an image",
      page.image[:4] in (b"\x89PNG", b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xdb")
      or page.image[:3] == b"\xff\xd8\xff")
check("its text layer comes with it, umlauts intact",
      "Seite 3: Ich heiße Anna." in page.text, repr(page.text))
w, h = Image.open(io.BytesIO(page.image)).size
check("rendered at the size the model wants",
      max(w, h) <= documents.MAX_SIDE and max(w, h) >= documents.MAX_SIDE - 2,
      f"{w}x{h}")
check("more than six pages at once is refused, with the way out",
      "at most 6" in raises(doc.select, ""))
check("six is fine", len(doc.select("1-6")) == 6)
check("the note names the file and the pages",
      documents.describe(doc, doc.select("2-4,7"))
      == "[attached blatt.pdf, pages 2-4, 7 of 8]",
      documents.describe(doc, doc.select("2-4,7")))
check("a page past the end is refused", "no page 9" in raises(doc.page, 9))
check("the whole text layer is available", "Seite 8" in doc.text)

print("\nimages:")
img = documents.load(png(), "board.png")
check("an image is one page", img.kind == documents.IMAGE and img.page_count == 1)
check("the note says it is an image",
      documents.describe(img, img.select("")) == "[attached image: board.png]")

# A photo: millions of colours. Noise stands in for one.
import os  # noqa: E402
big =Image.frombytes("RGB", (1200, 900), os.urandom(1200 * 900 * 3)).resize((4000, 3000))
buf = io.BytesIO()
big.save(buf, "JPEG", quality=90)
photo = documents.load(buf.getvalue(), "IMG_0014.JPG").page(1)
pw, ph = Image.open(io.BytesIO(photo.image)).size
check("a phone photo is shrunk for the model", max(pw, ph) == documents.MAX_SIDE,
      f"{pw}x{ph}")
check("and sent as JPEG, not a huge PNG", photo.mime == "image/jpeg")

# A photo taken sideways: stored landscape, EXIF says rotate.
side = Image.new("RGB", (400, 200), "white")
exif = Image.Exif()
exif[0x0112] = 6        # rotate 90 CW to display
buf = io.BytesIO()
side.save(buf, "JPEG", exif=exif)
upright = documents.load(buf.getvalue(), "side.jpg").page(1)
uw, uh = Image.open(io.BytesIO(upright.image)).size
check("a sideways phone photo is turned upright", uh > uw, f"{uw}x{uh}")

clear = documents.load(png(mode="RGBA", colour=(0, 0, 0, 0)), "shot.png").page(1)
pixel = Image.open(io.BytesIO(clear.image)).convert("RGB").getpixel((5, 5))
check("transparency becomes white, not black", pixel == (255, 255, 255), str(pixel))

print("\nrefusals:")
check("an empty file", "empty" in raises(documents.load, b"", "x.png"))
check("something that is neither",
      "not a PDF or an image" in raises(documents.load, b"hello world", "x.txt"))
check("a broken PDF", "does not open" in raises(documents.load,
                                                 b"%PDF-1.4 garbage", "x.pdf"))
check("data URL for the model",
      img.page(1).data_url().startswith("data:image/png;base64,"))
parts = documents.image_parts([img.page(1)])
check("content parts in the OpenAI shape Ollama reads",
      parts[0]["type"] == "image_url" and parts[0]["image_url"]["url"].startswith("data:"))

print(f"\n{'PASS' if not failures else 'FAIL'} — documents\n")
sys.exit(1 if failures else 0)
