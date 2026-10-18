"""Split data/processed/all.jsonl into train / val / test / test_unseen_templates.

Two anti-cheating rules:
 1. test_unseen_templates: a few synthetic sentence templates are removed from
    EVERYTHING else, so this set tests sentence shapes the model never saw.
 2. Splitting is by doc_id, so chunks of one document never land in two splits.

Run:  python -m privylens.data.splitter
"""
import json
import random
from collections import Counter, defaultdict
from privylens.schema import ENTITIES

N_UNSEEN = 5               # how many synthetic templates to hold out
VAL, TEST = 0.1, 0.1       # rest (80%) is train


def split(rows, seed=0):
    rng = random.Random(seed)
    templates = sorted({r["template_id"] for r in rows if r["template_id"]})
    unseen = set(rng.sample(templates, N_UNSEEN))

    out = defaultdict(list)
    groups = defaultdict(list)                 # doc_id -> its examples
    for r in rows:
        if r["template_id"] in unseen:
            out["test_unseen_templates"].append(r)
        else:
            groups[r["doc_id"]].append(r)

    ids = sorted(groups)
    rng.shuffle(ids)
    n_val, n_test = int(len(ids) * VAL), int(len(ids) * TEST)
    for i, d in enumerate(ids):                # first chunk of ids -> val, next -> test, rest -> train
        name = "val" if i < n_val else "test" if i < n_val + n_test else "train"
        out[name] += groups[d]
    return out, sorted(unseen)


if __name__ == "__main__":
    rows = [json.loads(l) for l in open("data/processed/all.jsonl", encoding="utf-8")]
    parts, unseen = split(rows)
    # sanity checks: no doc in two splits, and every entity type still appears in train
    seen = {}
    for name, rs in parts.items():
        for r in rs:
            assert seen.setdefault(r["doc_id"], name) == name, "doc leaked across splits"
    train_labels = {s["label"] for r in parts["train"] for s in r["spans"]}
    assert train_labels == set(ENTITIES), f"missing from train: {set(ENTITIES) - train_labels}"
    for name, rs in parts.items():
        with open(f"data/splits/{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
        print(f"{name:22s}{len(rs):7d}", dict(Counter(s["label"] for r in rs for s in r["spans"]).most_common()))
    print("held-out templates:", unseen)
