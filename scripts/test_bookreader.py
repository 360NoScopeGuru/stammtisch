"""The book reader's deterministic half: finding rows on a page, and deciding
what to believe when readings disagree. No model needed.

The vision readings themselves are checked against the real books — see the
README — because a synthetic page proves nothing about how a model reads a
scan. What is checked here is everything that decides *whether to trust* a
reading, which is the part that has to be right every time.

`python scripts/test_bookreader.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from app import bookreader as br  # noqa: E402
from app import curriculum  # noqa: E402

failures = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global failures
    failures += not cond
    print(f"  {'PASS' if cond else 'FAIL'}  {label}"
          + (f"\n        {detail}" if detail and not cond else ""))


def page(rows: int, skew: float = 0.0, rules: bool = True, border: bool = False,
         left_ends: float = 0.56, bullets: bool = True):
    """A syllabus page: headings, then `rows` rows of two columns, with
    dotted rules between them. `skew` is the drop in px across the width."""
    w, h = 1400, 1960
    img = Image.new("RGB", (w, h), (250, 250, 248))
    d = ImageDraw.Draw(img)
    d.rectangle((120, 150, 600, 200), fill=(40, 40, 40))         # SPRACHHANDLUNGEN
    d.rectangle((860, 150, 1150, 200), fill=(40, 40, 40))        # GRAMMATIK
    top, step = 260, 170
    for r in range(rows):
        y = top + r * step
        for line in range(3):                                    # can-do lines
            d.rectangle((140, y + 20 + line * 30, int(w * left_ends), y + 36 + line * 30),
                        fill=(30, 30, 30))
        for line in range(3):                                    # grammar bullets
            if bullets:
                d.ellipse((840, y + 22 + line * 30, 850, y + 32 + line * 30),
                          fill=(30, 30, 30))
            d.rectangle((890, y + 20 + line * 30, 1200, y + 36 + line * 30),
                        fill=(30, 30, 30))
        if rules and r < rows - 1:
            ry = y + step - 10
            for x in range(120, 1300, 9):                        # dotted rule
                dy = int(skew * x / w)
                d.rectangle((x, ry + dy, x + 3, ry + dy + 2), fill=(90, 90, 90))
    if border:
        d.rectangle((0, 0, w, 6), fill=(60, 60, 60))
        d.rectangle((0, h - 6, w, h), fill=(60, 60, 60))
    return img


print("\nthe gutter:")
g = br._gutter(page(9))
check("found between the columns, not in the middle of the text",
      int(1400 * 0.56) < g < 840, f"gutter at {g}")
check("not in the narrow strip between bullets and their text",
      not 850 < g < 890, f"gutter at {g}")

print("\nrows:")
strips = br.row_strips(page(9), 9)
check("a ruled page divides into its rows", strips is not None and len(strips) == 9,
      str(strips))
if strips:
    check("rows are contiguous and in order",
          all(a[1] == b[0] for a, b in zip(strips, strips[1:])))
    check("the first row starts below the headings", strips[0][0] > 200,
          str(strips[0]))
check("a skewed scan still divides",
      (s := br.row_strips(page(9, skew=12), 9)) is not None and len(s) == 9)
check("a scan's own border is not taken for a rule",
      (s := br.row_strips(page(9, border=True), 9)) is not None and len(s) == 9)
check("the wrong number of rows is a refusal, not a guess",
      br.row_strips(page(8), 9) is None)
check("a page with no rules is a refusal",
      br.row_strips(page(9, rules=False), 9) is None)

print("\nreading the model's answers:")
check("statements printed on one line with | are split",
      br._strings(["Uhrzeiten erfragen | sich verabreden", "• Termine absagen"])
      == ["Uhrzeiten erfragen", "sich verabreden", "Termine absagen"])
check("soft hyphens are typesetting, not text",
      br._strings(["trenn\xadbare Verben"]) == ["trennbare Verben"])
check("junk is dropped", br._strings([None, 3, "", " | ", "Futur"]) == ["Futur"])

print("\nagreement:")
a = (["Nachrichten formulieren"],
     ["Präteritum", "temporale Nebensätze mit bis, seit(dem), bevor, während"])
split = (["Nachrichten formulieren"],
         ["Präteritum", "temporale Nebensätze mit bis, seit(dem),", "bevor, während"])
check("a bullet split at its line break still agrees", br._row_agrees(a, split))
shifted = (["Gründe und Folgen ausdrücken"], ["Konjunktiv II", "Genitiv-Attribute"])
check("a row from another chapter does not", not br._row_agrees(a, shifted))
check("compare_rows names the chapter that differs",
      br.compare_rows([a, a], [a, shifted], [15, 16])
      == ["chapter 16: the can-do statements differ",
          "chapter 16: the grammar differs"])

print("\nvoting on spelling:")
r1 = (["sich über Kinderinnerungen austauschen"], ["Präteritum"])
r2 = (["sich über Kindheitserinnerungen austauschen"], ["Präteritum"])
r3 = (["sich über Kindheitserinnerungen austauschen"], ["Präteritum"])
row, unsure = br._vote(r1, r2, r3)
check("two readings that agree outvote the one that slipped",
      row[0] == ["sich über Kindheitserinnerungen austauschen"] and not unsure,
      str(row))
r3b = (["sich über Kinderheitserinnerungen austauschen"], ["Präteritum"])
row, unsure = br._vote(r2, r1, r3b)
check("three different spellings: kept as the first reading, and named",
      row[0] == r2[0] and unsure == r2[0], str((row, unsure)))
row, unsure = br._vote(a, split, split)
check("a split bullet is backed by the unsplit text, not flagged", not unsure,
      str(unsure))

print("\nthe text layer:")
ch = [curriculum.Chapter(number=4, title="x", can_do=["sich verabreden"],
                         grammar=["trennbare Verben"])]
text = "SPRACHHANDLUNGEN sich verabreden | Termine absagen GRAMMATIK trenn\xadbare Verben " * 3
check("items on the page pass", br.check_against_text(ch, text) == [])
ch[0].grammar = ["trenbare Verben"]
check("a misreading fails where there is text to check",
      br.check_against_text(ch, text) == ["chapter 4: 'trenbare Verben' is not on the page"])
check("a scan with no text is not failed for it",
      br.check_against_text(ch, "") == [])

print("\nwhat the model is told:")
lay = br._layout({"left": "grammar"})
check("a page without a can-do column says so outright",
      "no SPRACHHANDLUNGEN column" in lay and '"can_do" is always []' in lay, lay)
check("and the ignored column is named as ignored",
      "right column is not part of the syllabus" in lay)

print(f"\n{'PASS' if not failures else 'FAIL'} — book reader\n")
sys.exit(1 if failures else 0)
