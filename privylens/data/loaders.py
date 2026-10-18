"""Convert the public ai4privacy dataset into our unified format, merge it with
our synthetic data, and write one combined file.

ai4privacy has 28 labels; we map them onto our 11 (see privylens/schema.py).
Its documents are long, so we cut them into short chunks the model can handle.

Run (after python -m privylens.data.synthetic):
    python -m privylens.data.loaders
In : data/raw/ai4_train_en.jsonl  (download from HF; licence = academic use only,
     credit AI4Privacy, never commit/redistribute)  + data/synthetic/synthetic.jsonl
Out: data/processed/all.jsonl
"""
import json
import random
from collections import Counter
from privylens.schema import ENTITIES

# their label -> our label. Anything not listed is dropped (stays plain text).
LABEL_MAP = {
    "GIVENNAME1": "PERSON", "GIVENNAME2": "PERSON",
    "LASTNAME1": "PERSON", "LASTNAME2": "PERSON", "LASTNAME3": "PERSON",
    "CITY": "LOCATION", "STATE": "LOCATION", "COUNTRY": "LOCATION",
    "STREET": "LOCATION", "BUILDING": "LOCATION", "SECADDRESS": "LOCATION",
    "POSTCODE": "LOCATION",
    "EMAIL": "EMAIL", "TEL": "PHONE", "IP": "IP_ADDRESS",
    "DATE": "DATE", "BOD": "DATE",
    "IDCARD": "ID_NUMBER", "SOCIALNUMBER": "ID_NUMBER", "PASSPORT": "ID_NUMBER",
    "DRIVERLICENSE": "ID_NUMBER",
    "USERNAME": "ID_NUMBER", "PASS": "ID_NUMBER",  # credentials: hide them (recall > precision)
}
CHUNK_CHARS = 400      # target chunk size; lines are packed up to this
EMPTY_KEEP = 0.1       # keep 10% of chunks with no PII, as "nothing to hide" examples


def convert_spans(text, mask):
    """Map labels, drop unknown ones, merge 'John' + 'Smith' into one PERSON span."""
    spans = sorted(
        ({"start": m["start"], "end": m["end"], "label": LABEL_MAP[m["label"]]}
         for m in mask if m["label"] in LABEL_MAP),
        key=lambda s: s["start"])
    out = []
    for s in spans:
        prev = out[-1] if out else None
        if (prev and s["label"] == prev["label"] == "PERSON"
                and not text[prev["end"]:s["start"]].strip()):   # only whitespace between
            prev["end"] = s["end"]
        else:
            out.append(s)
    return out


def chunk(text, spans):
    """Cut a document at newlines into ~400-char pieces; re-base span offsets."""
    # find line boundaries, then greedily pack lines into chunks
    lines, pos = [], 0
    for ln in text.split("\n"):
        lines.append((pos, pos + len(ln)))
        pos += len(ln) + 1
    chunks, cur = [], None
    for a, b in lines:
        if cur and b - cur[0] <= CHUNK_CHARS:
            cur[1] = b
        else:
            cur and chunks.append(tuple(cur))
            cur = [a, b]
    cur and chunks.append(tuple(cur))
    for a, b in chunks:
        inside = [s for s in spans if s["start"] >= a and s["end"] <= b]
        crossing = any(s["start"] < b and s["end"] > a and s not in inside for s in spans)
        if b > a and not crossing:          # skip chunks that would cut a span in half
            yield text[a:b], [{**s, "start": s["start"] - a, "end": s["end"] - a} for s in inside]


def load_ai4privacy(path):
    for doc_i, line in enumerate(open(path, encoding="utf-8")):
        d = json.loads(line)
        spans = convert_spans(d["source_text"], d["privacy_mask"])
        for j, (text, sp) in enumerate(chunk(d["source_text"], spans)):
            if sp or random.random() < EMPTY_KEEP:
                yield {"id": f"ai4_{doc_i:06d}_{j}", "text": text, "spans": sp,
                       "source": "ai4privacy", "template_id": None,
                       "doc_id": f"ai4_{doc_i:06d}"}   # chunks of one doc must stay in one split


if __name__ == "__main__":
    random.seed(0)
    rows = list(load_ai4privacy("data/raw/ai4_train_en.jsonl"))
    rows += [json.loads(l) for l in open("data/synthetic/synthetic.jsonl", encoding="utf-8")]
    for r in rows:
        r.setdefault("doc_id", r["id"])
        for s in r["spans"]:
            assert s["label"] in ENTITIES and r["text"][s["start"]:s["end"]].strip(), r
    random.shuffle(rows)
    with open("data/processed/all.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"{len(rows)} examples written")
    for src in ("ai4privacy", "synthetic"):
        c = Counter(s["label"] for r in rows if r["source"] == src for s in r["spans"])
        print(src, dict(c.most_common()))
    print("empty examples:", sum(not r["spans"] for r in rows))
