"""Train the encoder-decoder (teacher forcing, label smoothing 0.1), keep the best checkpoint, then score it.
Only examples that fit in 128 tokens are used (long chunks are skipped; coverage is printed).

Colab:  python -m privylens.training.train_seq2seq --epochs 15
Sanity: python -m privylens.training.train_seq2seq --limit 20 --epochs 80 --dropout 0 --out checkpoints/s2s_overfit.pt
"""
import argparse
import json
import math
import os
import random
import torch
import torch.nn.functional as F
from privylens.data.seq2seq_pairs import make_pair
from privylens.evaluation.preservation import evaluate_seq2seq, report
from privylens.models.seq2seq import Seq2Seq, load_seq2seq
from privylens.tokenizer.bpe import BOS, EOS, PAD, BPE
from privylens.training.train_tagger import read

MAXLEN = 128


def fits(rows, tok):
    """Keep rows whose source and target both fit in MAXLEN tokens."""
    return [r for r in rows
            if len(tok.encode(r["text"])[0]) <= MAXLEN
            and len(tok.encode(make_pair(r["text"], r["spans"])[1])[0]) < MAXLEN]


def make_examples(rows, tok):
    ex = []
    for r in rows:
        src = tok.encode(r["text"])[0]
        tgt = tok.encode(make_pair(r["text"], r["spans"])[1])[0]
        ex.append((src, [BOS] + tgt, tgt + [EOS]))         # decoder input / what it must predict
    return ex


def batches(ex, bs, shuffle):
    idx = list(range(len(ex)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = [ex[j] for j in idx[i:i + bs]]
        pad = lambda k: torch.tensor([x[k] + [PAD] * (max(len(y[k]) for y in b) - len(x[k])) for x in b])
        yield pad(0), pad(1), pad(2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="tokenizers/bpe_8k.json")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="checkpoints/seq2seq.pt")
    a = p.parse_args()

    random.seed(0); torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPE.load(a.tokenizer)
    all_train = read("train", a.limit)
    train_rows, val_rows = fits(all_train, tok), fits(read("val", a.limit), tok)
    print(f"train rows kept: {len(train_rows)}/{len(all_train)} (<= {MAXLEN} tokens)")
    train = make_examples(train_rows, tok)

    cfg = dict(vocab_size=len(tok.vocab), d=a.d, heads=a.heads, d_ff=2 * a.d,
               enc_layers=a.layers, dec_layers=a.layers, dropout=a.dropout)
    model = Seq2Seq(**cfg).to(dev)
    print(f"device={dev} params={sum(x.numel() for x in model.parameters()):,}")

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
        for src, tgt_in, tgt_out in batches(train, a.bs, True):
            src, tgt_in, tgt_out = src.to(dev), tgt_in.to(dev), tgt_out.to(dev)
            loss = F.cross_entropy(model(src, tgt_in).transpose(1, 2), tgt_out, ignore_index=PAD, label_smoothing=0.1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            losses.append(loss.item())
        r = evaluate_seq2seq(model, tok, val_rows[:300], dev)      # generation is slow: score only 300 val rows
        saved = r["placeholder_F1"] > best
        if saved:
            best = r["placeholder_F1"]
            torch.save({"model": model.state_dict(), "config": cfg}, a.out)
        history.append({"epoch": epoch + 1, "train_loss": sum(losses) / len(losses),
                        "val_placeholder_F1": r["placeholder_F1"], "val_exact": r["exact_match"]})
        json.dump(history, open(a.out + ".history.json", "w"))
        print(f"epoch {epoch + 1:3d}  train_loss={history[-1]['train_loss']:.4f}  val_placeholder_F1={r['placeholder_F1']:.3f}"
              f"  val_exact={r['exact_match']:.3f}{'  *saved' if saved else ''}")

    if not a.limit:
        best_model = load_seq2seq(a.out, dev)
        for split in ("test", "test_unseen_templates"):
            rows = read(split)
            kept = fits(rows, tok)
            print(f"\n{split}: scoring {len(kept)}/{len(rows)} rows that fit in {MAXLEN} tokens")
            report(f"seq2seq / {split}", evaluate_seq2seq(best_model, tok, kept, dev))
