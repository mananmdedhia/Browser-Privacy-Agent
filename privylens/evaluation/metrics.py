"""Scoring used for every model (regex now, neural models later).

evaluate(detect_fn, rows) runs a detector over examples and reports:
  P / R / F1   exact-match spans (same start, end AND label), per entity + micro
  leaked_span  gold spans NOT fully covered by predictions (privacy risk; lower=better)
  leaked_char  gold characters left uncovered (catches half-hidden names)
  over_redact  harmless characters that got masked (usability cost)
Leakage ignores labels: hiding an email as [PHONE] still hides it.

Run: python -m privylens.evaluation.metrics   (scores the regex baseline)
"""
import json
from collections import Counter


def _covered(chars, spans):
    return sum(any(s["start"] <= i < s["end"] for s in spans) for i in chars)


def evaluate(detect_fn, rows):
    tp, fp, fn = Counter(), Counter(), Counter()
    n_gold = leaked_spans = gold_chars = leaked_chars = safe_chars = over_chars = 0
    for r in rows:
        gold, pred = r["spans"], detect_fn(r["text"])
        g = {(s["start"], s["end"], s["label"]) for s in gold}
        p = {(s["start"], s["end"], s["label"]) for s in pred}
        for t in g & p: tp[t[2]] += 1
        for t in p - g: fp[t[2]] += 1
        for t in g - p: fn[t[2]] += 1
        for s in gold:
            n = s["end"] - s["start"]
            c = _covered(range(s["start"], s["end"]), pred)
            gold_chars += n; leaked_chars += n - c; leaked_spans += c < n
        n_gold += len(gold)
        gold_set = {i for s in gold for i in range(s["start"], s["end"])}
        safe = len(r["text"]) - len(gold_set)
        safe_chars += safe
        over_chars += _covered([i for i in range(len(r["text"])) if i not in gold_set], pred)

    def prf(t, f_p, f_n):
        P = t / (t + f_p) if t + f_p else 0.0
        R = t / (t + f_n) if t + f_n else 0.0
        return {"P": P, "R": R, "F1": 2 * P * R / (P + R) if P + R else 0.0}

    per_entity = {e: prf(tp[e], fp[e], fn[e]) for e in sorted(set(tp) | set(fp) | set(fn))}
    return {"micro": prf(sum(tp.values()), sum(fp.values()), sum(fn.values())),
            "per_entity": per_entity,
            "leaked_span": leaked_spans / n_gold, "leaked_char": leaked_chars / gold_chars,
            "over_redact": over_chars / safe_chars}


def report(name, res):
    m = res["micro"]
    print(f"\n== {name} ==  micro P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} | "
          f"leaked_span={res['leaked_span']:.3f} leaked_char={res['leaked_char']:.3f} over_redact={res['over_redact']:.4f}")
    for e, v in res["per_entity"].items():
        print(f"   {e:13s} P={v['P']:.2f} R={v['R']:.2f} F1={v['F1']:.2f}")


if __name__ == "__main__":
    from privylens.baselines.regex_detector import detect
    for split in ("test", "test_unseen_templates"):
        rows = [json.loads(l) for l in open(f"data/splits/{split}.jsonl", encoding="utf-8")]
        report(f"regex / {split}", evaluate(detect, rows))
