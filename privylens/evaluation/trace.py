"""Records every number the production tagger computes for one fixed sentence, for the website's Live view.

Run:  python -m privylens.evaluation.trace      (writes web/src/trace.json; re-run if the model changes)
The forward pass is re-done by hand here (same maths as models/layers.py, dropout off) so each step can be
saved; an assert checks it ends at exactly the scores the real model gives.
Only the first K=8 of the 128 numbers per token are kept: enough to show, small enough to ship.
"""
import json
import math
import torch
from privylens.baselines.regex_detector import detect as regex_detect
from privylens.engine.pipeline import Engine, merge, redact, snap
from privylens.models.layers import pad_mask
from privylens.schema import TAGS
from privylens.tokenizer.align import to_spans
from privylens.tokenizer.bpe import BPE
from privylens.tokenizer.pretokenize import pretokenize

TEXT = "Send the report to Rahul Sharma at rahul@example.com. His employee ID is ZX-48213-K."
K = 8
r = lambda t, n=3: [round(v, n) for v in t.tolist()]
head = lambda t: r(t[:K])                                      # first K numbers of a vector
stats = lambda t: [round(t.mean().item(), 3), round(t.std(unbiased=False).item(), 3)]


@torch.no_grad()
def trace(engine, tok, text):
    m = engine.model
    ids, offs = tok.encode(text)
    x_ids = torch.tensor([ids])
    words = pretokenize(text)
    word_of = [next(i for i, (_, a, b) in enumerate(words) if a <= s < b) for s, _ in offs]

    emb = m.emb(x_ids)[0]
    pe = m.pos.pe[: len(ids)]
    x = (emb + pe)[None]
    layers = []
    for blk in m.blocks:
        a = blk.attn
        T = x.size(1)
        split = lambda t: t.view(1, T, a.h, a.dk).transpose(1, 2)
        q, k, v = split(a.q(x)), split(a.k(x)), split(a.v(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(a.dk)                       # (1,h,T,T)
        mixed = (scores.softmax(-1) @ v).transpose(1, 2).reshape(1, T, -1)
        attn_out = a.o(mixed)
        resid1 = x + attn_out
        ln1 = blk.n1(resid1)
        hidden = blk.ff.net[1](blk.ff.net[0](ln1))                               # 128 -> 256, GELU
        resid2 = ln1 + blk.ff.net[3](hidden)                                     # 256 -> 128, add input back
        x = blk.n2(resid2)
        layers.append({
            "scores": [[r(row, 2) for row in s] for s in scores[0]],             # [head][query][key]
            "attn_out": [head(t) for t in attn_out[0]], "resid1": [head(t) for t in resid1[0]],
            "ln1": [head(t) for t in ln1[0]], "ln1_stats": [stats(t) for t in resid1[0]],
            "hidden": [head(t) for t in hidden[0]], "resid2": [head(t) for t in resid2[0]],
            "out": [head(t) for t in x[0]], "out_stats": [stats(t) for t in resid2[0]],
        })
    logits = m.head(x)[0]
    assert torch.allclose(logits, m(x_ids)[0], atol=1e-5), "hand-made forward pass differs from the model"

    probs = logits.softmax(-1)
    tags = probs.argmax(-1).tolist()
    spans = to_spans(offs, tags)
    regex = regex_detect(text)
    model_spans = snap(text, [dict(s) for s in spans])
    found = [{**s, "source": "regex"} for s in regex] + [{**s, "source": "model"} for s in model_spans]
    final = merge([{**s, "conf": 1.0} for s in found])
    vault = {}
    out = redact(text, [dict(s) for s in final], vault)
    return {
        "text": text, "d_model": m.head.in_features, "heads": m.blocks[0].attn.h, "d_ff": m.blocks[0].ff.net[0].out_features,
        "tags": TAGS, "shown": K,
        "words": [{"text": w.lstrip("▁"), "start": a, "end": b} for w, a, b in words],
        "tokens": [{"id": i, "text": text[a:b], "start": a, "end": b, "word": w, "emb": head(emb[n]), "pe": head(pe[n]),
                    "x0": head(emb[n] + pe[n]), "logits": r(logits[n]), "tag": TAGS[tags[n]], "p": round(probs[n].max().item(), 3)}
                   for n, ((i, (a, b)), w) in enumerate(zip(zip(ids, offs), word_of))],
        "layers": layers,
        "spans": [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in spans],
        "regex": regex, "model": [{k: s[k] for k in ("start", "end", "label")} for s in model_spans],
        "final": [{"start": s["start"], "end": s["end"], "label": s["label"], "source": s["source"]} for s in final],
        "redacted": out, "vault": {v: ph for v, ph in vault.items()},
    }


if __name__ == "__main__":
    eng = Engine.from_checkpoint()
    from privylens.models.encoder_tagger import load_tagger
    eng.model = load_tagger("checkpoints/tagger.pt")
    data = trace(eng, BPE.load(eng.model.tokenizer_path), TEXT)
    json.dump(data, open("web/src/trace.json", "w", encoding="utf-8"), separators=(",", ":"))
    print(data["redacted"], "|", len(data["tokens"]), "tokens")
