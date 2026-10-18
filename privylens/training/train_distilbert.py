"""Reference model: DistilBERT (pretrained by others on huge text) fine-tuned on OUR data and labels.
It is NOT part of the product. It answers: how close does our from-scratch model get to a pretrained one,
and at what size / speed?  (The only file allowed to import `transformers`.)

Colab:  python -m privylens.training.train_distilbert --epochs 4
Sanity: python -m privylens.training.train_distilbert --limit 50 --epochs 2
Same splits, same scorer (evaluation/metrics.py) as every other model, so numbers are comparable.
"""
import argparse
import json
import math
import os
import random
import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer
from privylens.evaluation.metrics import evaluate, report
from privylens.schema import ID2TAG, TAG2ID, TAGS
from privylens.tokenizer.align import to_bio, to_spans
from privylens.training.train_tagger import IGNORE, read

NAME = "distilbert-base-cased"
WINDOW = 126                       # + [CLS] and [SEP] = 128 tokens


def encode(hf, text):
    """text -> (token ids, char offsets) WITHOUT special tokens, so our own alignment code can be reused."""
    e = hf(text, add_special_tokens=False, return_offsets_mapping=True)
    return e["input_ids"], [tuple(o) for o in e["offset_mapping"]]


def make_examples(rows, hf):
    ex = []
    for r in rows:
        ids, offs = encode(hf, r["text"])
        tags = to_bio(offs, r["spans"])
        for s in range(0, len(ids), WINDOW):
            ex.append(([hf.cls_token_id] + ids[s:s + WINDOW] + [hf.sep_token_id],
                       [IGNORE] + tags[s:s + WINDOW] + [IGNORE]))
    return ex


def batches(ex, bs, pad_id, shuffle):
    idx = list(range(len(ex)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = [ex[j] for j in idx[i:i + bs]]
        T = max(len(x[0]) for x in b)
        yield (torch.tensor([x[0] + [pad_id] * (T - len(x[0])) for x in b]),
               torch.tensor([x[1] + [IGNORE] * (T - len(x[1])) for x in b]))


def logits(model, ids, pad_id):
    return model(input_ids=ids, attention_mask=(ids != pad_id).long()).logits


@torch.no_grad()
def predict(model, hf, text, device="cpu"):
    """text -> char spans (overlapping windows, probabilities averaged), same idea as our own tagger."""
    model.eval()
    ids, offs = encode(hf, text)
    n = len(ids)
    if n == 0:
        return []
    probs, seen = torch.zeros(n, len(TAGS)), torch.zeros(n, 1)
    for s in sorted({*range(0, max(n - WINDOW, 0) + 1, WINDOW // 2), max(n - WINDOW, 0)}):
        chunk = torch.tensor([[hf.cls_token_id] + ids[s:s + WINDOW] + [hf.sep_token_id]], device=device)
        p = logits(model, chunk, hf.pad_token_id).softmax(-1)[0, 1:-1].cpu()      # drop [CLS]/[SEP]
        probs[s:s + len(p)] += p
        seen[s:s + len(p)] += 1
    return to_spans(offs, (probs / seen).argmax(-1).tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)      # much smaller than for scratch models: we only nudge pretrained weights
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="checkpoints/distilbert.pt")
    a = p.parse_args()

    random.seed(0); torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hf = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForTokenClassification.from_pretrained(NAME, num_labels=len(TAGS), id2label=ID2TAG, label2id=TAG2ID).to(dev)
    n_params = sum(x.numel() for x in model.parameters())
    print(f"device={dev} params={n_params:,} (~{n_params * 4 / 1e6:.0f} MB)")

    val_rows = read("val", a.limit)
    train = make_examples(read("train", a.limit), hf)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = a.epochs * math.ceil(len(train) / a.bs)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    best, history = -1.0, []
    for epoch in range(a.epochs):
        model.train()
        losses = []
        for ids, tags in batches(train, a.bs, hf.pad_token_id, True):
            ids, tags = ids.to(dev), tags.to(dev)
            loss = F.cross_entropy(logits(model, ids, hf.pad_token_id).transpose(1, 2), tags, ignore_index=IGNORE)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            losses.append(loss.item())
        f1 = evaluate(lambda t: predict(model, hf, t, dev), val_rows[:1000])["micro"]["F1"]
        saved = f1 > best
        if saved:
            best = f1
            torch.save(model.state_dict(), a.out)
        history.append({"epoch": epoch + 1, "train_loss": sum(losses) / len(losses), "val_f1": f1})
        json.dump(history, open(a.out + ".history.json", "w"))
        print(f"epoch {epoch + 1:3d}  train_loss={history[-1]['train_loss']:.4f}  val_F1={f1:.4f}{'  *saved' if saved else ''}")

    if not a.limit:
        model.load_state_dict(torch.load(a.out, map_location=dev))
        for split in ("test", "test_unseen_templates"):
            report(f"distilbert / {split}", evaluate(lambda t: predict(model, hf, t, dev), read(split)))
