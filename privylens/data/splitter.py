"""Split data/processed/all.jsonl into train / val / test / test_unseen_templates.

Two anti-cheating rules:
 1. test_unseen_templates: a few synthetic sentence templates are removed from
    EVERYTHING else, so this set tests sentence shapes the model never saw.
 2. Splitting is by doc_id, so chunks of one document never land in two splits.

Case robustness: real forms and ID cards are often ALL CAPS, but only ~0.5% of our name examples were.
So AFTER splitting, train gets extra UPPER / lower-case copies (same character offsets, so spans stay valid),
and a `test_caps` split (upper + lower copies of 1000 test rows) measures the effect. val / test / unseen are untouched.

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


def case_variants(rows, p_upper=0.15, p_lower=0.10, seed=1):
    """Randomly add an UPPERCASE and a lowercase copy of some rows. Uses its own random generator,
    so the main split above is unchanged. Rows whose length would change are skipped (offsets must stay valid)."""
    rng = random.Random(seed)
    out = []
    for r in rows:
        for tag, fn, p in (("up", str.upper, p_upper), ("lo", str.lower, p_lower)):
            t = fn(r["text"])
            if len(t) == len(r["text"]) and rng.random() < p:
                out.append({**r, "id": f"{r['id']}_{tag}", "text": t})
    return out


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
    parts["train"] += case_variants(parts["train"])
    caps = [{**r, "id": f"{r['id']}_{tag}", "text": fn(r["text"])} for r in parts["test"][:1000]
            for tag, fn in (("up", str.upper), ("lo", str.lower)) if len(fn(r["text"])) == len(r["text"])]
    parts["test_caps"] = caps
    for name, rs in parts.items():
        with open(f"data/splits/{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
        print(f"{name:22s}{len(rs):7d}", dict(Counter(s["label"] for r in rs for s in r["spans"]).most_common()))
    print("held-out templates:", unseen)
