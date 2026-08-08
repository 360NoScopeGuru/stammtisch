"""The corrector must never "fix" a name, and must never swallow real grammar.

The second half matters more than the first. Dropping a genuine correction is
invisible — the learner simply never finds out they were wrong — so the filter
has to be provably narrow, not merely effective.

Pure stdlib. `python scripts/test_corrections.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.corrections import (  # noqa: E402
    filter_corrections, protected_names, touches_a_name,
)

NAME = "Vamshee Dharan"

# (utterance, original, corrected, should_be_dropped, label)
CASES = [
    # --- the real failure, captured from a live session -------------------
    ("Ich heiße Fahmshidharan.",
     "Ich heiße Fahmshidharan.", "Ich heiße Farshid Shidharan.", True,
     "the actual bug: invented a different name"),
    ("Ich heiße Vamshee.",
     "Ich heiße Vamshee.", "Ich heiße Vamshee Dharan.", True,
     "appending to a name is still rewriting it"),
    ("Mein Name ist Wamsi.",
     "Mein Name ist Wamsi.", "Mein Name ist Wamsig.", True,
     "'Mein Name ist X' protects X"),
    ("Ich heiße Vamshee und komme aus Indien.",
     "Ich heiße Vamshee und komme aus Indien.",
     "Ich heiße Vamsi und komme aus Indien.", True,
     "name touched even though the rest is fine"),

    # --- genuine corrections that MUST survive ----------------------------
    ("Ich bin Student.",
     "Ich bin Student.", "Ich bin Studentin.", False,
     "gender ending — 'bin' must not protect the noun"),
    ("Ich habe ein Buch.",
     "Ich habe ein Buch.", "Ich habe ein Buch gelesen.", False,
     "missing participle"),
    ("Ich gehe zu die Schule.",
     "zu die Schule", "zur Schule", False,
     "dative contraction"),
    ("Wir waren in Berlin.",
     "Wir waren in Berlin.", "Wir war in Berlin.", False,
     "'waren' scores 0.55 against 'dharan' — must stay below threshold"),
    ("Ich heiße Vamshee und ich habe ein Buch.",
     "ich habe ein Buch", "ich habe ein Buch gelesen", False,
     "name in the utterance, but the correction is elsewhere"),
    ("Der Name ist gut.",
     "Der Name ist gut.", "Der Name ist schön.", False,
     "'Name ist' bigram, but the changed word is not the protected one"),
    ("Ich komme aus Indien.",
     "Ich komme aus Indien.", "Ich komme aus Indien her.", False,
     "place name untouched by the change"),
]


def main() -> int:
    fails = 0

    def check(label, got, want):
        nonlocal fails
        ok = got == want
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            print(f"        wanted {want}, got {got}")

    print("\nprotected_names:")
    check("learner name is always protected",
          "vamshee" in protected_names(NAME, "Hallo."), True)
    check("name after 'heiße' is picked up",
          "fahmshidharan" in protected_names(NAME, "Ich heiße Fahmshidharan."),
          True)
    check("'bin' does NOT protect the following noun",
          "student" in protected_names(NAME, "Ich bin Student."), False)
    check("'Name ist' protects the following token",
          "wamsi" in protected_names(NAME, "Mein Name ist Wamsi."), True)

    print("\ndrop / keep:")
    for utterance, original, corrected, want_drop, label in CASES:
        protected = protected_names(NAME, utterance)
        check(label, touches_a_name(original, corrected, protected), want_drop)

    print("\nfilter_corrections:")
    kept, dropped = filter_corrections(
        [
            {"original": "Ich heiße Fahmshidharan.",
             "corrected": "Ich heiße Farshid Shidharan."},
            {"original": "ich habe ein Buch",
             "corrected": "ich habe ein Buch gelesen"},
            {"original": "same", "corrected": "same"},
            {"original": "", "corrected": "something"},
        ],
        NAME, "Ich heiße Fahmshidharan und ich habe ein Buch.",
    )
    check("keeps only the real grammar correction", len(kept), 1)
    check("the kept one is the participle fix",
          kept[0]["corrected"], "ich habe ein Buch gelesen")
    check("drops the name rewrite, the no-op and the empty", len(dropped), 3)

    print(f"\n{'PASS' if not fails else 'FAIL'} — correction filtering\n")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
