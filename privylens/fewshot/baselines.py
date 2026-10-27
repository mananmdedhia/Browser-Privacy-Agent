"""The comparison methods for the few-shot experiment. Each builds `text -> spans` from the user's K examples ((text,start,end)).

  induce_regex   regex written FROM the examples (letters -> [A-Z]{n}, digits -> \\d{n}, separators kept). What a user could do by hand.
  finetune       ordinary fine-tuning: a classifier pre-trained on the train companies, then trained further on the K examples
                 (+ shape-preserving fakes of them) against generic decoys. Same encoder as the few-shot model, same example budget.
  tagger         our existing Transformer tagger (fixed tag set): only its ID_NUMBER spans count. Cannot adapt; needs no examples.
"""
import copy
import itertools
import random
import re
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from privylens.fewshot.data import candidates
from privylens.fewshot.model import Proto, feats, pool


def _cls(c): return "d" if c.isdigit() else "u" if c.isupper() else "l" if c.islower() else c


def _one(k, n): return {"d": rf"\d{{{n}}}", "u": f"[A-Z]{{{n}}}", "l": f"[a-z]{{{n}}}"}.get(k, re.escape(k) * n)


def induce_regex(sup, label="CUSTOM"):
    """One pattern per distinct example shape. ponytail: exact run lengths, so 'ORX-728' and 'ORX-72841' need an example each;
    widen runs to {min,max} if lengths vary in practice."""
    pats = {"".join(_one(k, len(list(g))) for k, g in itertools.groupby(t[a:b], _cls)) for t, a, b in sup}
    rx = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(sorted(pats)) + r")(?![A-Za-z0-9])")
    return lambda text: [{"start": m.start(), "end": m.end(), "label": label} for m in rx.finditer(text)]


class Clf(nn.Module):
    def __init__(self):
        super().__init__()
        self.p, self.h = Proto("both"), nn.Linear(64, 1)

    def forward(self, S, C): return self.h(self.p.raw(S, C)).squeeze(-1)


def pretrain_clf(companies, steps=800, seed=0):
    """'Is this an employee ID of SOME company, or a decoy?' on the train companies. Returns (model, pool of decoys)."""
    rng = random.Random(seed); torch.manual_seed(seed)
    pools = [pool(c, 150, rng) for c in companies]
    cat = lambda k: tuple(torch.cat([p[k][i] for p in pools]) for i in range(2))
    P, N = cat("pos"), cat("neg")
    m = Clf(); opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for step in range(steps):
        pi, ni = torch.randint(len(P[0]), (64,)), torch.randint(len(N[0]), (64,))
        x = torch.cat([P[0][pi], N[0][ni]]), torch.cat([P[1][pi], N[1][ni]])
        loss = F.binary_cross_entropy_with_logits(m(*x), torch.cat([torch.ones(64), torch.zeros(64)]))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0: print(f"  [fine-tune base] step {step} loss {loss.item():.3f}", flush=True)
    return m.eval(), N


def _jitter(s, rng):
    return "".join(rng.choice(string.digits) if c.isdigit() else rng.choice(string.ascii_uppercase) if c.isupper()
                   else rng.choice(string.ascii_lowercase) if c.islower() else c for c in s)


def finetune(base, decoys, sup, rng, steps=20, label="CUSTOM"):
    m = copy.deepcopy(base).train()
    fakes = [(t[:a] + _jitter(t[a:b], rng) + t[b:], a, b) for t, a, b in sup for _ in range(8)]
    P = feats(sup + fakes)
    ni = torch.randperm(len(decoys[0]))[:96]
    x = torch.cat([P[0], decoys[0][ni]]), torch.cat([P[1], decoys[1][ni]])
    y = torch.cat([torch.ones(len(P[0])), torch.zeros(len(ni))])
    w = torch.where(y == 1, len(ni) / len(P[0]), 1.)                   # K is tiny: weight positives so they are not ignored
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(m(*x), y, weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()

    def detect(text):
        cs = candidates(text)
        if not cs: return []
        with torch.no_grad():
            lg = m(*feats([(text, a, b) for a, b in cs]))
        return [{"start": a, "end": b, "label": label} for (a, b), v in zip(cs, lg) if v > 0]
    return detect


def tagger(path="checkpoints/tagger.pt", label="CUSTOM"):
    from privylens.engine.pipeline import Engine
    ms = Engine.from_checkpoint(path).model_spans
    return lambda text: [{**s, "label": label} for s in ms(text) if s["label"] == "ID_NUMBER"]
