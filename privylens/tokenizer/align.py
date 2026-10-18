"""Turn character spans into per-token BIO labels (training targets) and back.

to_bio:    gold spans + token offsets -> tag ids   (first token of a span = B-X, rest = I-X)
to_spans:  tag ids + token offsets    -> char spans (how predictions become redactions)

Run (checks how well tokens line up with gold spans on val):
    python -m privylens.tokenizer.align tokenizers/bpe_8k.json
"""
import json
import sys
from privylens.schema import ID2TAG, TAG2ID


def to_bio(offsets, spans):
    tags, started = [], set()
    for a, b in offsets:
        # which span (if any) does this token overlap?
        hit = next((i for i, s in enumerate(spans) if a < s["end"] and b > s["start"]), None)
        if hit is None:
            tags.append("O")
        else:
            tags.append(("I-" if hit in started else "B-") + spans[hit]["label"])
            started.add(hit)
    return [TAG2ID[t] for t in tags]


def to_spans(offsets, tag_ids):
    spans, cur = [], None
    for (a, b), t in zip(offsets, tag_ids):
        tag = ID2TAG[t]
        if tag == "O":
            cur = None
        elif tag.startswith("B-") or cur is None or cur["label"] != tag[2:]:
            cur = {"start": a, "end": b, "label": tag[2:]}   # new span (also repairs a stray I-)
            spans.append(cur)
        else:
            cur["end"] = b                                    # continue current span
    return spans


if __name__ == "__main__":
    from privylens.tokenizer.bpe import BPE
    tok = BPE.load(sys.argv[1])
    rows = [json.loads(l) for l in open("data/splits/val.jsonl", encoding="utf-8")]
    ok = total = long = 0
    for r in rows:
        ids, offs = tok.encode(r["text"])
        long += len(ids) > 128
        got = {(s["start"], s["end"], s["label"]) for s in to_spans(offs, to_bio(offs, r["spans"]))}
        gold = {(s["start"], s["end"], s["label"]) for s in r["spans"]}
        ok += len(gold & got); total += len(gold)
    print(f"gold spans recoverable from tokens: {ok}/{total} = {ok/total:.4f}  (ceiling for any model)")
    print(f"examples longer than 128 tokens: {long}/{len(rows)}")
