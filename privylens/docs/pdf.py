"""PDF redaction: PDF in -> redacted PDF + report + private vault out. Runs entirely on this computer.

How it works
  1. read each page: the text AND the position of every word (PyMuPDF)
  2. run our privacy engine on the page text (regex + tagger); one session for the whole file,
     so the same person gets the same placeholder on every page
  3. turn the detected character ranges back into word boxes on the page
  4. document sweep: every other place where a detected value appears is hidden too ("Manan Dedhia" found once
     -> hidden everywhere), because a value missed once is usually just as private
  5. TRUE redaction: the text under each black box is deleted from the file, not just covered.
     Also removed: links pointing at hidden text, and the document metadata (author, title...)
  6. check: re-read the finished PDF and list any detected word that is still visible

Not covered (ponytail): scanned PDFs (no text layer -> needs OCR), text drawn as pictures.
Run:  python -m privylens.docs.pdf [file.pdf ...]       (default: every PDF in pdf/, output in pdf_out/)
"""
import argparse
import glob
import json
import os
import re
import sys
import pymupdf
from privylens.engine.pipeline import Engine

MIN_SWEEP = 3          # values shorter than this are not swept through the rest of the document
FONT = 7               # size of the placeholder label inside a black box (skipped silently if the box is too small)


def page_words(page):
    """-> (text, words). Words on the same visual line are joined by spaces, lines by newlines, so the model
    reads the page the way a person does (PDF-internal block/line numbering can split "Manan Dedhia" into two lines).
    words[i] = (rect, start, end, line_id): the word's box and its character range inside `text`.
    ponytail: a very wide gap (columns, tab stops) starts a new line; true multi-column layout reading order is not solved."""
    raw = sorted(page.get_text("words"), key=lambda w: (w[1] + w[3]) / 2)       # top to bottom by vertical centre
    lines = []
    for w in raw:
        cy, h = (w[1] + w[3]) / 2, w[3] - w[1]
        if lines and abs(cy - lines[-1][0]) < 0.5 * h:
            lines[-1][1].append(w)
        else:
            lines.append([cy, [w]])
    text, words, n = "", [], 0
    for _, line in lines:
        line.sort(key=lambda w: w[0])                                           # left to right
        prev = None
        for x0, y0, x1, y1, w, *_ in line:
            if text:
                text += "\n" if prev is None or x0 - prev > 3 * (y1 - y0) else " "
            if prev is None or x0 - prev > 3 * (y1 - y0):
                n += 1
            words.append((pymupdf.Rect(x0, y0, x1, y1), len(text), len(text) + len(w), n))
            text += w
            prev = x1
    return text, words


def span_rects(span, words):
    """One box per line covering every word the span touches (a half-touched word is hidden whole: safer)."""
    boxes = {}
    for rect, a, b, key in words:
        if a < span["end"] and b > span["start"]:
            boxes[key] = boxes[key] | rect if key in boxes else pymupdf.Rect(rect)
    return list(boxes.values())


def add_box(page, rect, label):
    page.add_redact_annot(rect, text=label, fontsize=FONT, fill=(0, 0, 0), text_color=(1, 1, 1),
                          align=pymupdf.TEXT_ALIGN_CENTER)


def redact_pdf(src, dst, engine, keep=(), hide=()):
    """keep: words/phrases that must NEVER be hidden (e.g. company names the model wrongly flags).
    hide: words/phrases that must ALWAYS be hidden wherever they appear (e.g. the document owner's name)."""
    doc = pymupdf.open(src)
    keep_l = {w.lower() for k in keep for w in k.split()}          # matched word by word

    # ---- phase 1: detect on every page (one session = consistent placeholders across pages) ----
    sid, pages, values, blind = None, [], {}, []    # values: original text -> (placeholder, label); blind: pages without text
    for page in doc:
        text, words = page_words(page)
        if not text.strip():
            blind.append(page.number + 1)            # no text layer (scan / picture): nothing can be detected here
        r = engine.sanitise(text, sid) if text.strip() else {"spans": [], "flagged": [], "session_id": sid}
        sid = r["session_id"]
        flagged = {(s["start"], s["end"]) for s in r["flagged"]}
        spans = []
        for s in r["spans"]:
            s["flagged"] = (s["start"], s["end"]) in flagged
            s["value"] = text[s["start"]:s["end"]]
            parts = [p.strip() for p in s["value"].split("\n")]   # a value can cross a line break; sweep each line part
            if all(w.lower().strip(".,;:") in keep_l for w in s["value"].split()):
                continue                             # the user said: never hide this (every word of the value is on the keep list)
            for part in parts:
                values.setdefault(part, (s["placeholder"], s["label"]))
            spans.append(s)
        pages.append((spans, words))
    custom = {}                                      # --hide values get their own placeholders
    for i, h in enumerate(hide, start=1):
        custom[f"[CUSTOM_{i}]"] = h
        values.setdefault(h, (f"[CUSTOM_{i}]", "CUSTOM"))

    # ---- phase 2: draw the boxes (own detections + document sweep), then delete what is underneath ----
    hits = []
    for pno, (page, (spans, words)) in enumerate(zip(doc, pages), start=1):
        done = []                                    # boxes already placed on this page
        for s in spans:
            for rect in span_rects(s, words):
                add_box(page, rect, s["placeholder"]); done.append(rect)
            hits.append({"page": pno, "label": s["label"], "placeholder": s["placeholder"],
                         "confidence": round(s["conf"], 3), "found_by": s["source"], "low_confidence": s["flagged"],
                         "swept": False})
        for value, (ph, label) in values.items():    # sweep: same value elsewhere on the page
            if len(value) < MIN_SWEEP:
                continue
            for rect in page.search_for(value):
                covered = sum((rect & d).get_area() for d in done)      # how much of this spot is already under a box
                if covered < 0.8 * rect.get_area():
                    add_box(page, rect, ph); done.append(rect)
                    hits.append({"page": pno, "label": label, "placeholder": ph, "confidence": None,
                                 "found_by": "sweep", "low_confidence": False, "swept": True})
        for link in page.get_links():                # a hidden word can still be a clickable link to its text
            if any(pymupdf.Rect(link["from"]).intersects(d) for d in done):
                page.delete_link(link)
        page.apply_redactions()                      # removes the text under the boxes for real

    doc.set_metadata({})                             # author / title / creator can hold the very same names
    doc.del_xml_metadata()
    doc.save(dst, garbage=4, deflate=True)           # garbage=4 drops the now-unused old objects
    doc.close()

    # ---- phase 3: check the finished file ----
    after = pymupdf.open(dst)
    text_after = "\n".join(p.get_text() for p in after)
    links_after = [l.get("uri", "") for p in after for l in p.get_links()]
    after.close()
    survivors = sorted({w for v in values for w in re.findall(r"\w[\w.@/:-]*", v)
                        if len(w) >= MIN_SWEEP and w.lower() not in keep_l and re.search(rf"(?<![\w]){re.escape(w)}(?![\w])", text_after)})
    link_leaks = [u for u in links_after if any(v in u for v in values if len(v) >= MIN_SWEEP)]
    return {"file": os.path.basename(src), "pages": len(pages), "redactions": hits,
            "counts": {l: sum(h["label"] == l for h in hits) for l in sorted({h["label"] for h in hits})},
            "still_visible_words": survivors, "links_still_containing_hidden_text": link_leaks,
            "pages_without_text_NOT_REDACTED": blind,
            "low_confidence_boxes": sum(h["low_confidence"] for h in hits)}, \
        {ph: v for ph, v in {**engine.sessions.get(sid, {}), **custom}.items() if ph in {h["placeholder"] for h in hits}}   # only placeholders actually used


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Redact private values in PDFs (local only).")
    ap.add_argument("files", nargs="*", help="PDFs to redact (default: every PDF in pdf/)")
    ap.add_argument("--keep", nargs="*", default=[], help='never hide these, e.g. --keep Microsoft Coursera "Big Data"')
    ap.add_argument("--hide", nargs="*", default=[], help='always hide these everywhere, e.g. --hide "Manan Dedhia"')
    a = ap.parse_args()
    files = a.files or sorted(glob.glob("pdf/*.pdf"))
    os.makedirs("pdf_out", exist_ok=True)
    engine = Engine.from_checkpoint()
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        report, vault = redact_pdf(f, f"pdf_out/{name}.redacted.pdf", engine, keep=a.keep, hide=a.hide)
        json.dump(report, open(f"pdf_out/{name}.report.json", "w", encoding="utf-8"), indent=1)
        json.dump(vault, open(f"pdf_out/{name}.vault.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)  # SENSITIVE: real values
        print(f"{report['file']}: {len(report['redactions'])} boxes {report['counts']}")
        if report["pages_without_text_NOT_REDACTED"]:
            print(f"   WARNING: pages {report['pages_without_text_NOT_REDACTED']} have no text layer (scanned image). "
                  "NOTHING was redacted there; this file is NOT safe to share (OCR needed).")
        print(f"   still visible: {report['still_visible_words'] or 'nothing'} | links leaking: {report['links_still_containing_hidden_text'] or 'none'}"
              f" | low-confidence boxes: {report['low_confidence_boxes']}")
