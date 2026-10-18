"""Train the encoder tagger, keep the best checkpoint, then score it on the test sets.

Normal run (Colab GPU):  python -m privylens.training.train_tagger --epochs 10
Overfit sanity check:    python -m privylens.training.train_tagger --limit 10 --epochs 150 --bs 10 --dropout 0 --out checkpoints/overfit.pt
                         (the model must memorise 10 examples: loss -> ~0, else something is broken)

Each epoch: go over the training data once (learn), then score on val (span F1) and
save the checkpoint if it is the best so far.
Loss = cross-entropy per token; padding is ignored (label -100).
Learning rate: warm up for 5% of steps, then cosine decay.
"""
import argparse
import json
import math
import os
import random
import torch
import torch.nn.functional as F
from privylens.evaluation.metrics import evaluate, report
from privylens.models.encoder_tagger import EncoderTagger, load_tagger, predict
from privylens.schema import TAGS
from privylens.tokenizer.align import to_bio
from privylens.tokenizer.bpe import PAD, BPE

IGNORE = -100


def read(split, limit=0):
    rows = [json.loads(l) for l in open(f"data/splits/{split}.jsonl", encoding="utf-8")]
    return rows[:limit] if limit else rows


def make_examples(rows, tok, window=128):
    """text -> (token ids, tag ids), cut into windows of at most 128 tokens."""
    ex = []
    for r in rows:
        ids, offs = tok.encode(r["text"])
        tags = to_bio(offs, r["spans"])
        ex += [(ids[s:s + window], tags[s:s + window]) for s in range(0, len(ids), window)]
    return ex


def batches(ex, bs, shuffle):
    idx = list(range(len(ex)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = [ex[j] for j in idx[i:i + bs]]
        T = max(len(x[0]) for x in b)                  # pad every sequence to the longest in the batch
        yield (torch.tensor([x[0] + [PAD] * (T - len(x[0])) for x in b]),
               torch.tensor([x[1] + [IGNORE] * (T - len(x[1])) for x in b]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="tokenizers/bpe_8k.json")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pos", default="sinusoidal", choices=["sinusoidal", "learned"])
    p.add_argument("--limit", type=int, default=0, help="use only the first N rows (sanity checks)")
    p.add_argument("--out", default="checkpoints/tagger.pt")
    a = p.parse_args()

    random.seed(0); torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPE.load(a.tokenizer)
    train_rows, val_rows = read("train", a.limit), read("val", a.limit)
    train = make_examples(train_rows, tok)

    cfg = dict(vocab_size=len(tok.vocab), num_tags=len(TAGS), d=a.d, heads=a.heads, d_ff=2 * a.d,
               layers=a.layers, dropout=a.dropout, learned_pos=a.pos == "learned")
    model = EncoderTagger(**cfg).to(dev)
    print(f"device={dev} params={sum(x.numel() for x in model.parameters()):,} train_windows={len(train)}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = a.epochs * math.ceil(len(train) / a.bs)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm
                                              else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    best, history = -1.0, []
    for epoch in range(a.epochs):
        model.train()
        losses = []
        for ids, tags in batches(train, a.bs, shuffle=True):
            ids, tags = ids.to(dev), tags.to(dev)
            loss = F.cross_entropy(model(ids).transpose(1, 2), tags, ignore_index=IGNORE)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # stops rare huge updates
            opt.step(); sched.step()
            losses.append(loss.item())
        f1 = evaluate(lambda t: predict(model, tok, t, device=dev), val_rows[:1000])["micro"]["F1"]
        saved = f1 > best
        if saved:
            best = f1
            torch.save({"model": model.state_dict(), "config": cfg}, a.out)
        history.append({"epoch": epoch + 1, "train_loss": sum(losses) / len(losses), "val_f1": f1})
        json.dump(history, open(a.out + ".history.json", "w"))   # for the loss-curve plot in the presentation
        print(f"epoch {epoch + 1:3d}  train_loss={sum(losses) / len(losses):.4f}  val_F1={f1:.4f}{'  *saved' if saved else ''}")

    if not a.limit:                                   # final exam with the best checkpoint
        best_model = load_tagger(a.out, dev)
        final = {}
        for split in ("test", "test_unseen_templates"):
            final[split] = evaluate(lambda t: predict(best_model, tok, t, device=dev), read(split))
            report(f"tagger / {split}", final[split])
        json.dump({"args": vars(a), "params": sum(x.numel() for x in best_model.parameters()), **final},
                  open(a.out + ".test.json", "w"))     # saved so ablation runs can be compared in one table
