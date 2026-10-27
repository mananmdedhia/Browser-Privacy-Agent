"""Few-shot detector: a prototypical (Siamese-style) network with two views of every ID-like candidate.

  shape view    the candidate's own characters (ORX-72841 -> upper,upper,upper,'-',digits...)
  context view  the characters around it, with the candidate itself masked ("Staff code <ID> was added")
Both go through a small char-CNN and are fused into one unit vector. The user's examples are averaged into a PROTOTYPE;
a candidate is flagged when it is closer to the prototype than a threshold. Optional "not this" examples give a second prototype.
No weights change when a new category is added: that is the whole point (compare fine-tuning in baselines.py).

Train:  python -m privylens.fewshot.run   (trains, evaluates, saves checkpoints/fewshot.pt)
"""
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from privylens.fewshot.data import candidates, rows

SL, LL, RL = 16, 32, 16                          # chars kept: candidate, left context, right context
MASK = 1                                         # id of the "candidate goes here" marker; 0 = padding, chars = ord+2


def _cls(i):                                     # character class per id: pad, digit, upper, lower, other, mask
    if i == 0: return 0
    if i == MASK: return 5
    c = chr(i - 2)
    return 1 if c.isdigit() else 2 if c.isupper() else 3 if c.islower() else 4


CLS = torch.tensor([_cls(i) for i in range(130)])


def _ids(s): return [min(ord(c), 127) + 2 for c in s]


def feats(items):
    """items: [(text, start, end)] -> shape ids (n,SL), context ids (n,LL+1+RL). Tensors the encoder reads."""
    S, C = [], []
    for t, a, b in items:
        s = _ids(t[a:b])[:SL]
        S.append(s + [0] * (SL - len(s)))
        l, r = _ids(t[max(0, a - LL):a]), _ids(t[b:b + RL])
        C.append([0] * (LL - len(l)) + l + [MASK] + r + [0] * (RL - len(r)))
    return torch.tensor(S), torch.tensor(C)


BLANK_C = feats([("x", 0, 1)])[1][0]             # context of an example given WITHOUT a sentence around it


class Branch(nn.Module):
    """chars -> vector: embed (char + char class) -> 2 conv layers -> [max, mean] pool (max = shape, mean = length)."""
    def __init__(self, d=64):
        super().__init__()
        self.e, self.c = nn.Embedding(130, 32, padding_idx=0), nn.Embedding(6, 16, padding_idx=0)
        self.conv = nn.Sequential(nn.Conv1d(48, d, 3, padding=1), nn.ReLU(), nn.Conv1d(d, d, 3, padding=1), nn.ReLU())

    def forward(self, ids):
        h = self.conv(torch.cat([self.e(ids), self.c(CLS[ids])], -1).transpose(1, 2))
        m = (ids > 0).unsqueeze(1)
        return torch.cat([h.masked_fill(~m, -1e4).max(2).values, (h * m).sum(2) / m.sum(2).clamp(min=1)], -1)


class Proto(nn.Module):
    def __init__(self, views="both", d=64):
        super().__init__()
        self.views, self.d = views, d
        self.shape, self.ctx, self.fuse = Branch(d), Branch(d), nn.Linear(4 * d, 64)
        self.scale, self.b = nn.Parameter(torch.tensor(10.)), nn.Parameter(torch.tensor(0.5))

    def raw(self, S, C):
        z = lambda: torch.zeros(len(S), 2 * self.d)
        return self.fuse(torch.cat([self.shape(S) if self.views != "context" else z(),
                                    self.ctx(C) if self.views != "shape" else z()], -1))

    def embed(self, S, C): return F.normalize(self.raw(S, C), dim=-1)

    def logits(self, q, pos, neg=None):
        """q (n,64) unit vectors; pos/neg: support embeddings. Positive logit = closer to the prototype than to 'not this' (or than b)."""
        m = q @ F.normalize(neg.mean(0), dim=0) if neg is not None and len(neg) else self.b
        return self.scale * (q @ F.normalize(pos.mean(0), dim=0) - m)


def pool(c, n_docs, rng, rows_fn=rows, cand=candidates):
    """All candidates in n_docs fresh documents of company c, as features: {'pos': (S,C), 'neg': (S,C)}."""
    P, N = [], []
    for r in rows_fn(c, n_docs, rng):
        gold = {(s["start"], s["end"]) for s in r["spans"]}
        for a, b in cand(r["text"]):
            (P if (a, b) in gold else N).append((r["text"], a, b))
    return {"pos": feats(P), "neg": feats(N)}


def _take(f, idx): return f[0][idx], f[1][idx]


def _blank(f): return f[0], BLANK_C.expand(len(f[0]), -1)


def train_proto(companies, views="both", steps=1500, seed=0, log=True, kit=(rows, candidates)):
    """Episodes: pick a company, show K of its IDs (K=1/5/10, sometimes without a sentence, sometimes with 'not this' examples),
    then ask the model to tell its OTHER IDs from decoys and from other companies' IDs."""
    rng = random.Random(seed); torch.manual_seed(seed)
    pools = [pool(c, 150, rng, *kit) for c in companies]
    m = Proto(views); opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for step in range(steps):
        loss = 0
        for _ in range(4):
            i = rng.randrange(len(pools)); j = rng.choice([k for k in range(len(pools)) if k != i])
            P, N, O = pools[i]["pos"], pools[i]["neg"], pools[j]["pos"]
            K, Kn = rng.choice([1, 5, 10]), rng.choice([0, 0, 1, 5])
            pi = torch.randperm(len(P[0]))[:K + 16]; ni = torch.randperm(len(N[0]))[:Kn + 16]; oi = torch.randperm(len(O[0]))[:8]
            sup = _take(P, pi[:K]); sup = _blank(sup) if rng.random() < .25 else sup
            nsup = _take(N, ni[:Kn])
            qS = torch.cat([P[0][pi[K:]], N[0][ni[Kn:]], O[0][oi]]); qC = torch.cat([P[1][pi[K:]], N[1][ni[Kn:]], O[1][oi]])
            y = torch.cat([torch.ones(len(pi) - K), torch.zeros(len(ni) - Kn + len(oi))])
            e = m.embed(torch.cat([sup[0], nsup[0], qS]), torch.cat([sup[1], nsup[1], qC]))
            lg = m.logits(e[K + Kn:], e[:K], e[K:K + Kn] if Kn else None)
            loss = loss + F.binary_cross_entropy_with_logits(lg, y)
        opt.zero_grad(); (loss / 4).backward(); opt.step()
        if log and step % 300 == 0: print(f"  [{views}] step {step} loss {loss.item() / 4:.3f}", flush=True)
    return m.eval()


def parse_example(s):
    """'Staff code: [[ORX-72841]]' -> (text, start, end) with the sentence as context; a bare 'ORX-72841' has no context."""
    m = re.search(r"\[\[(.+?)\]\]", s)
    if not m: return s, 0, len(s)
    return s[:m.start()] + m.group(1) + s[m.end():], m.start(), m.start() + len(m.group(1))


def detector(model, pos, neg=(), calibrate=False, bare=False, label="CUSTOM", cand=candidates):
    """Build text -> spans from the user's examples ((text,start,end) lists). Nothing is trained here.
    calibrate: no labelled data exists for a new category, so set the cut-off from the examples alone: accept anything at least as
    close to the prototype as the user's least typical example (leave-one-out). Only used with >=2 positives and no 'not this' list."""
    with torch.no_grad():
        S, C = feats(pos); S, C = _blank((S, C)) if bare else (S, C)
        pe = model.embed(S, C)
        ne = model.embed(*feats(neg)) if len(neg) else None
        b = model.b
        if calibrate and ne is None and len(pe) > 1:
            loo = [pe[i] @ F.normalize(torch.cat([pe[:i], pe[i + 1:]]).mean(0), dim=0) for i in range(len(pe))]
            b = torch.minimum(model.b, torch.stack(loo).min())
        pp = F.normalize(pe.mean(0), dim=0)
        pn = F.normalize(ne.mean(0), dim=0) if ne is not None else None

    def detect(text):
        cs = cand(text)
        if not cs: return []
        with torch.no_grad():
            q = model.embed(*feats([(text, a, e) for a, e in cs]))
            conf = torch.sigmoid(model.scale * (q @ pp - (q @ pn if pn is not None else b)))
        return [{"start": a, "end": e, "label": label, "conf": float(p)} for (a, e), p in zip(cs, conf) if p > .5]
    return detect


def load(path="checkpoints/fewshot.pt"):
    m = Proto(); m.load_state_dict(torch.load(path)); return m.eval()


def phrase_detector(phrases, not_phrases=(), label="CUSTOM"):
    """Names with no digits ("Project Nightfall") are never ID-like candidates, so the model cannot see them. Match them literally, and
    if >=2 examples start with the same word ("Project X", "Project Y"), also match that word + 1-3 Capitalised words.
    ponytail: exact words only, no fuzzy match; add a second model pass if codenames need real generalisation."""
    firsts = [p.split()[0] for p in phrases if len(p.split()) > 1]
    lead = {w for w in firsts if firsts.count(w) > 1}
    literal = re.compile(r"(?<!\w)(?:%s)(?!\w)" % "|".join(re.escape(p) for p in phrases), re.I)
    general = re.compile(r"(?<!\w)(?:%s)(?: [A-Z][\w-]*){1,3}" % "|".join(map(re.escape, lead))) if lead else None
    tails = {p.split()[-1] for p in phrases if p.split()[0] in lead and len(p.split()) == 2}      # "Project Nightfall" -> bare "Nightfall" too
    bare = re.compile(r"(?<!\w)(?:%s)(?!\w)" % "|".join(map(re.escape, tails))) if tails else None
    no = re.compile("|".join(re.escape(n) for n in not_phrases), re.I) if not_phrases else None

    def detect(text):
        spans = {m.span() for m in literal.finditer(text)} | ({m.span() for m in general.finditer(text)} if general else set())             | ({m.span() for m in bare.finditer(text)} if bare else set())
        skip = [m.span() for m in no.finditer(text)] if no else []                            # anything touching a "not this" phrase stays visible
        return [{"start": a, "end": b, "label": label, "conf": 1.0} for a, b in sorted(spans) if not any(a < d and c < b for c, d in skip)]
    return detect
