"""Write a minimal PDF with text at exact positions, for the tests.

Real course PDFs are copyrighted and gitignored, so the tests build their own.
Positioned text matters: the handout reader works from where each word sits
on the page, and a fixture without real coordinates would test nothing.
"""

from __future__ import annotations


def _escape(text: str) -> bytes:
    raw = text.encode("cp1252")
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def make_pdf(pages: list[list[tuple[float, float, str]]], size: float = 12) -> bytes:
    """`pages` is a list of pages, each a list of (x, y, text) in points from
    the bottom-left corner of an A4 page."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
               b"/Encoding /WinAnsiEncoding >>")
    pages_id = len(objects) + 1 + 2 * len(pages)   # reserved below
    kids = []
    for items in pages:
        stream = b"".join(
            b"BT /F1 %d Tf %.1f %.1f Td (" % (size, x, y) + _escape(t) + b") Tj ET\n"
            for x, y, t in items)
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream
                      + b"endstream")
        kids.append(add(
            b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
            % (pages_id, font, content)))
    assert add(b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % k for k in kids), len(kids))) == pages_id
    catalog = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id)

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objects) + 1, catalog, xref))
    return bytes(out)
