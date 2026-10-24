"""PDFs and images, turned into pages the model can look at.

The tutor's model (gemma3) reads images, so a worksheet, a photo of the board
or a page of the course book can be handed to it directly. That matters more
than it sounds: the text layer of these PDFs is not reliable. The workbook
that ships with the course book prints every exercise twice in its text layer,
interleaved, with the gaps and column structure gone — «Wir ___ Lisa und
Marie» comes out as «(1) Lisa und Marie. Wir». The page image has none of
those problems.

So every page is rendered to an image, and the text layer, where there is one,
is kept alongside it for the jobs that need exact strings rather than a
reading.

Measured on the course workbook with gemma3:12b, warm: about 6 s per page. A
focused question ("transcribe exercise 15") reads far better than a request to
transcribe everything on the page, which is why pages are sent along with the
learner's actual question rather than pre-digested into text.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

PDF, IMAGE = "pdf", "image"
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

# gemma3 resizes every image to 896x896 internally, so a whole A4 page reaches
# it with its small print squeezed to a few pixels a letter. Pages are kept at
# a higher resolution than that, so that they can be sent as two overlapping
# halves, each getting the model's full 896 — see `model_images`.
MAX_SIDE = 1960
SEND_SIDE = 1400

# Halving a page helps only while there are few of them: past this, the
# doubled images cost more context than the sharper print is worth.
TILE_UP_TO = 3

# How many pages one message may carry. Each image costs ~256 tokens of
# context and a few seconds of encoding; a whole chapter at once would push the
# conversation itself out of the context window.
MAX_PAGES_PER_MESSAGE = 6

# A phone photo of a worksheet is routinely 10+ MB. Anything past this is not a
# page, it is a mistake, and decoding it would stall the server.
MAX_BYTES = 40 * 1024 * 1024
# Course books are another matter: the B1 volume of the learner's course is
# 215 MB with its audio embedded. PDFs are opened from disk, page by page, so
# their size costs nothing until a page is rendered.
MAX_PDF_BYTES = 1024 * 1024 * 1024


class DocumentError(ValueError):
    """The file could not be read as a PDF or an image. Message is user-facing."""


@dataclass
class Page:
    number: int                  # 1-based, as printed in a PDF viewer
    text: str = ""               # PDF text layer; "" for images and scans
    image: bytes = b""           # PNG/JPEG bytes, sized for the model
    mime: str = "image/png"

    def data_url(self) -> str:
        return f"data:{self.mime};base64," + base64.b64encode(self.image).decode()


@dataclass
class Document:
    name: str
    kind: str                    # PDF | IMAGE
    page_count: int
    pages: dict[int, Page] = field(default_factory=dict)   # rendered on demand
    _source: bytes | str = b""   # PDF bytes, or the path of a PDF on disk

    def page(self, number: int) -> Page:
        """Page `number` (1-based), rendering it on first use."""
        if number < 1 or number > self.page_count:
            raise DocumentError(
                f"{self.name} has {self.page_count} page(s); there is no page {number}")
        if number not in self.pages:
            self.pages[number] = _render_pdf_page(self._source, number)
        return self.pages[number]

    def select(self, spec: str = "") -> list[Page]:
        """Pages named by `spec` ("3", "2-4", "1,5-6"), all of them if blank.

        Capped at MAX_PAGES_PER_MESSAGE, and loudly: silently sending the
        first six pages of a twenty-page request would have the tutor answer
        questions about pages it never saw.
        """
        numbers = parse_pages(spec, self.page_count)
        if len(numbers) > MAX_PAGES_PER_MESSAGE:
            raise DocumentError(
                f"that is {len(numbers)} pages — at most {MAX_PAGES_PER_MESSAGE} "
                f"can be looked at in one go. Pick the pages you need, "
                f"e.g. \"{numbers[0]}-{numbers[0] + 2}\".")
        return [self.page(n) for n in numbers]

    @property
    def text(self) -> str:
        """The whole text layer, page by page. Cheap for PDFs; "" for images."""
        if self.kind != PDF:
            return ""
        return "\n\n".join(t for t in _pdf_texts(self._source) if t.strip())


def parse_pages(spec: str, page_count: int) -> list[int]:
    """'1,3-5' -> [1, 3, 4, 5]. Blank means every page."""
    spec = (spec or "").strip()
    if not spec:
        return list(range(1, page_count + 1))
    out: list[int] = []
    for part in re.split(r"[,\s]+", spec):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:\s*[-–]\s*(\d+))?", part)
        if not m:
            raise DocumentError(f"cannot read the page range {spec!r} — "
                                f"use something like 3 or 2-4 or 1,5")
        lo = int(m.group(1))
        hi = int(m.group(2) or lo)
        if lo > hi:
            lo, hi = hi, lo
        if lo < 1 or hi > page_count:
            raise DocumentError(f"pages {part} are outside 1-{page_count}")
        for n in range(lo, hi + 1):
            if n not in out:
                out.append(n)
    return out


# ---- loading ---------------------------------------------------------------

def _too_big(name: str, size: int, limit: int) -> DocumentError:
    return DocumentError(f"{name} is {size // (1024 * 1024)} MB — "
                         f"the limit is {limit // (1024 * 1024)} MB")


def load(data: bytes, name: str) -> Document:
    """Read `data` as a PDF or an image. Raises DocumentError on anything else."""
    if not data:
        raise DocumentError(f"{name} is empty")
    if data[:5] == b"%PDF-":
        if len(data) > MAX_PDF_BYTES:
            raise _too_big(name, len(data), MAX_PDF_BYTES)
        return _load_pdf(data, name)
    if len(data) > MAX_BYTES:
        raise _too_big(name, len(data), MAX_BYTES)
    return _load_image(data, name)


def load_path(path: str | Path, name: str = "") -> Document:
    """Open a file on disk. A PDF stays on disk and is read a page at a time."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise DocumentError(f"no such file: {p}")
    name = name or p.name
    size = p.stat().st_size
    if size == 0:
        raise DocumentError(f"{name} is empty")
    with p.open("rb") as fh:
        head = fh.read(5)
    if head == b"%PDF-":
        if size > MAX_PDF_BYTES:
            raise _too_big(name, size, MAX_PDF_BYTES)
        return _load_pdf(str(p), name)
    if size > MAX_BYTES:
        raise _too_big(name, size, MAX_BYTES)
    return _load_image(p.read_bytes(), name)


def _load_pdf(data: bytes | str, name: str) -> Document:
    try:
        import pypdfium2 as pdfium
    except ImportError as e:  # pragma: no cover - dependency missing
        raise DocumentError("reading PDFs needs pypdfium2: "
                            "pip install -r requirements.txt") from e
    try:
        pdf = pdfium.PdfDocument(data)
        count = len(pdf)
        pdf.close()
    except Exception as e:
        raise DocumentError(f"{name} does not open as a PDF ({e})") from e
    if count == 0:
        raise DocumentError(f"{name} has no pages")
    return Document(name=name, kind=PDF, page_count=count, _source=data)


def _load_image(data: bytes, name: str) -> Document:
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as e:  # pragma: no cover - dependency missing
        raise DocumentError("reading images needs Pillow: "
                            "pip install -r requirements.txt") from e
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise DocumentError(f"{name} is not a PDF or an image this can read") from e

    # Phone photos are stored sideways with an EXIF note saying so. The model
    # does not read EXIF, and a worksheet rotated 90 degrees reads as noise.
    img = ImageOps.exif_transpose(img)
    page = Page(number=1, **_encode(img))
    return Document(name=name, kind=IMAGE, page_count=1, pages={1: page})


def _encode(img) -> dict:
    """Downscale for the model and encode. Photos as JPEG, scans as PNG."""
    from PIL import Image

    if img.mode not in ("RGB", "L"):
        # Flatten transparency onto white: a transparent PNG screenshot
        # otherwise becomes black text on a black background.
        bg = Image.new("RGB", img.size, "white")
        rgba = img.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    img.thumbnail((MAX_SIDE, MAX_SIDE))

    # A photo with millions of colours compresses ten times smaller as JPEG at
    # no cost to legibility; a rendered page of black text is the reverse.
    many_colours = img.mode == "RGB" and img.getcolors(maxcolors=4096) is None
    buf = io.BytesIO()
    if many_colours:
        img.save(buf, "JPEG", quality=85)
        return {"image": buf.getvalue(), "mime": "image/jpeg"}
    img.save(buf, "PNG", optimize=True)
    return {"image": buf.getvalue(), "mime": "image/png"}


def _render_pdf_page(data: bytes | str, number: int) -> Page:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    try:
        page = pdf[number - 1]
        w, h = page.get_size()
        # Render straight at the size the model wants rather than rendering
        # big and shrinking, which costs seconds on a dense page.
        scale = min(3.0, MAX_SIDE / max(w, h, 1))
        img = page.render(scale=scale).to_pil()
        text = page.get_textpage().get_text_range() or ""
    finally:
        pdf.close()
    return Page(number=number, text=_clean_text(text), **_encode(img))


def _pdf_texts(data: bytes | str) -> list[str]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    try:
        return [_clean_text(pdf[i].get_textpage().get_text_range() or "")
                for i in range(len(pdf))]
    finally:
        pdf.close()


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    # Icon fonts in the course books (audio, pair-work symbols) come through
    # as private-use code points: not text, and unprintable in a console.
    text = re.sub("[-]", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---- for the model ---------------------------------------------------------

def model_images(pages: list[Page], tile: bool = True) -> list[str]:
    """Data URLs for the model: a tall page as two overlapping halves.

    Measured on the course workbook, transcribing one exercise from a full
    page: 0.87 similarity to the printed text whole, 0.93 halved — the halves
    stopped it merging two gaps into one and losing a question. The overlap
    keeps a line cut by the split readable in at least one half.
    """
    from PIL import Image

    tile = tile and len(pages) <= TILE_UP_TO
    urls = []
    for p in pages:
        img = Image.open(io.BytesIO(p.image))
        w, h = img.size
        if tile and h > w * 1.2:
            crops = [img.crop((0, 0, w, int(h * 0.55))),
                     img.crop((0, int(h * 0.45), w, h))]
        else:
            crops = [img]
        for c in crops:
            c.thumbnail((SEND_SIDE, SEND_SIDE))
            urls.append(_data_url(_encode(c)))
    return urls


def _data_url(encoded: dict) -> str:
    return (f"data:{encoded['mime']};base64,"
            + base64.b64encode(encoded["image"]).decode())


def image_parts(pages: list[Page], tile: bool = True) -> list[dict]:
    """OpenAI-style content parts, which Ollama passes to a vision model."""
    return [{"type": "image_url", "image_url": {"url": url}}
            for url in model_images(pages, tile)]


def describe(doc: Document, pages: list[Page]) -> str:
    """One line naming what was attached, for the transcript and the model."""
    if doc.kind == IMAGE:
        return f"[attached image: {doc.name}]"
    nums = [p.number for p in pages]
    if len(nums) == doc.page_count:
        span = "all pages" if doc.page_count > 1 else "1 page"
    elif len(nums) == 1:
        span = f"page {nums[0]}"
    else:
        span = f"pages {_ranges(nums)}"
    return f"[attached {doc.name}, {span} of {doc.page_count}]"


def _ranges(nums: list[int]) -> str:
    out, start, prev = [], None, None
    for n in sorted(nums):
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = n
    if start is not None:
        out.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(out)
