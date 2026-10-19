"""Baseline 1: find structured PII with hand-written patterns.
Cannot find PERSON / ORGANIZATION / LOCATION (needs understanding, not patterns);
that gap is part of the project's finding."""
import re

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December|Sept|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"

# (label, pattern, optional extra check). Earlier entries win when matches overlap.
PATTERNS = [
    ("EMAIL", r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", None),
    ("URL", r"https?://[^\s]*[^\s.,;:)]", None),
    ("IP_ADDRESS", r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.]*\d)|(?<![\w:])(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}(?![\w:])", None),
    ("PAN", r"\b[A-Z]{5}\d{4}[A-Z]\b", None),                      # AAAAA9999A
    ("AADHAAR", r"(?<!\d)[2-9]\d{3}\s?\d{4}\s?\d{4}(?!\d)", None),   # 12 digits, optional grouping
    ("DATE", rf"\b\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b"
             rf"|\b\d{{1,2}}(?:st|nd|rd|th)? (?:{MONTHS})\.?,? \d{{4}}\b"          # 15 October 2026, 15th Oct 2026
             rf"|\b(?:{MONTHS})\.? \d{{1,2}}(?:st|nd|rd|th)?,? \d{{4}}\b", None),  # Oct 15, 2026
    ("ID_NUMBER", r"\b(?:EMP|CASE|ACC|INV)\d{4,6}\b|\b[A-Z]{1,3}\d{6,14}\b", None),
    # phone: digits with spaces/dashes/brackets, 10-13 digits in total
    ("PHONE", r"(?<![\w.])\+?\d[\d\s().-]{8,}\d(?!\w)", lambda s: 10 <= sum(c.isdigit() for c in s) <= 13),
]


def detect(text):
    """Return non-overlapping spans [{start, end, label}] sorted by position."""
    found = []
    for label, pat, check in PATTERNS:
        for m in re.finditer(pat, text):
            if check and not check(m.group()):
                continue
            if all(m.end() <= s["start"] or m.start() >= s["end"] for s in found):  # no overlap with earlier winners
                found.append({"start": m.start(), "end": m.end(), "label": label})
    return sorted(found, key=lambda s: s["start"])
