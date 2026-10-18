"""Scoring for the encoder-decoder: it writes free text, so we check what it wrote.

  exact_match      output == the expected sanitised text
  placeholder F1   did it use the right [LABEL]s, the right number of times? (ignores position)
  leaked_span      original private values still present in the output (copy-through)
  preservation     how much of the NON-private text survived unchanged (1.0 = untouched)
  hallucinated / dropped / text_changed   share of rows with an invented placeholder / a missing one /
                   altered ordinary words
"""
import re
from collections import Counter
from difflib import SequenceMatcher
import torch
from privylens.data.seq2seq_pairs import make_pair
from privylens.tokenizer.bpe import PAD

PH = re.compile(r"\[[A-Z_]+\]")
words = lambda s: PH.sub("", s).split()


def evaluate_seq2seq(model, tok, rows, device="cpu", bs=64):
    model.eval()
    tp = fp = fn = exact = leaked = n_spans = hall = drop = changed = 0
    pres, bad = [], []
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        src = [tok.encode(r["text"])[0] for r in batch]
        T = max(map(len, src))
        gen = model.generate(torch.tensor([s + [PAD] * (T - len(s)) for s in src], device=device)).tolist()
        for r, g in zip(batch, gen):
            exp = tok.decode(tok.encode(make_pair(r["text"], r["spans"])[1])[0])   # expected, after the same normalisation
            got = tok.decode(g)
            e, o = Counter(PH.findall(exp)), Counter(PH.findall(got))
            tp += sum((e & o).values()); fp += sum((o - e).values()); fn += sum((e - o).values())
            hall += o - e != Counter(); drop += e - o != Counter()
            exact += got == exp
            for s in r["spans"]:
                v = r["text"][s["start"]:s["end"]]
                leaked += bool(re.search(rf"(?<!\w){re.escape(v)}(?!\w)", got)); n_spans += 1
            p = SequenceMatcher(None, words(exp), words(got)).ratio()
            pres.append(p)
            changed += (e == o) and p < 1
            if got != exp and len(bad) < 30:
                bad.append({"source": r["text"], "expected": exp, "got": got})
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    n = len(rows)
    return {"n": n, "exact_match": exact / n, "placeholder_P": P, "placeholder_R": R,
            "placeholder_F1": 2 * P * R / max(P + R, 1e-9), "leaked_span": leaked / max(n_spans, 1),
            "preservation": sum(pres) / n, "hallucinated": hall / n, "dropped": drop / n,
            "text_changed": changed / n, "examples": bad}


def report(name, res, show=3):
    print(f"\n== {name} (n={res['n']}) ==")
    print("  " + "  ".join(f"{k}={res[k]:.3f}" for k in ("exact_match", "placeholder_F1", "leaked_span",
                                                      "preservation", "hallucinated", "dropped", "text_changed")))
    for ex in res["examples"][:show]:
        print(f"  SRC: {ex['source'][:150]!r}\n  EXP: {ex['expected'][:150]!r}\n  GOT: {ex['got'][:150]!r}\n")
