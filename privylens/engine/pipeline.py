"""The privacy engine: text in -> sanitised text + the secret mapping out. Runs entirely on the user's machine.

  detect   regex finds structured PII (sure), our tagger finds names/places/orgs (with confidence)
  merge    overlapping detections are fused into one span (favours recall: hiding too much beats leaking)
  redact   every private value becomes a numbered placeholder; the same value always gets the same one
  validate re-scan the OUTPUT with regex: any hit = a leak -> blocked. Low-confidence model spans -> flagged
  vault    remembers placeholder -> original value (per session) so replies can be restored

Not done (ponytail): unicode/whitespace normalisation with an offset map. Add if odd input breaks offsets.
Try:  python -m privylens.engine.pipeline   (scores the hybrid on the test sets)
"""
import re
import uuid
from privylens.baselines.regex_detector import detect as regex_detect
from privylens.tokenizer.pretokenize import pretokenize

PLACEHOLDER = re.compile(r"\[[A-Z_]+_\d+\]")


def snap(text, spans):
    """Widen each model span to whole words, so half a word is never left visible ("MIH" found inside "MIHIR").
    Cheap safety net for the model's fragmented predictions; it can only hide MORE text, never less."""
    words = [(a, b) for _, a, b in pretokenize(text)]
    for s in spans:
        hit = [w for w in words if w[0] < s["end"] and w[1] > s["start"]]
        if hit:
            s["start"], s["end"] = min(s["start"], hit[0][0]), max(s["end"], hit[-1][1])
    return spans


STRONG = {"EMAIL", "URL", "IP_ADDRESS", "PAN"}      # formats regex gets (almost) always right


def merge(spans):
    """Pick non-overlapping spans. Priority when two overlap:
       0. user-defined categories (few-shot)   1. precise regex (STRONG types)   2. the neural tagger   3. loose regex (phone, Aadhaar, date, ID)
    then the longer span. Loose regex still counts where the model found nothing (recall).
    ponytail: the loser is dropped whole, so a partial overlap can leave a few characters visible;
    widen the winner to the union if the leak metric shows it matters."""
    def rank(s):
        if s["source"] == "custom": return 3                    # a category the user defined themselves always wins
        if s["source"] == "regex" and s["label"] in STRONG: return 2
        if s["source"] == "model": return 1
        # loose regex normally ranks last, but if it swallows smaller model spans whole ("Oct 15, 2026" vs the model's
        # "15" + "2026"), the model only caught pieces of a format the regex recognises: the regex wins
        # (dates only: phone / Aadhaar / ID patterns over-flag, so letting them win cost precision)
        swallows = s["label"] == "DATE" and any(m["source"] == "model" and s["start"] <= m["start"] and m["end"] <= s["end"]
                       and m["end"] - m["start"] < s["end"] - s["start"] for m in spans)
        return 2 if swallows else 0
    keep = []
    for s in sorted(spans, key=lambda s: (-rank(s), -(s["end"] - s["start"]))):
        if all(s["end"] <= k["start"] or s["start"] >= k["end"] for k in keep):
            keep.append(s)
    return sorted(keep, key=lambda s: s["start"])


def redact(text, spans, vault):
    """Replace spans (right to left so offsets stay valid). vault: {value: placeholder}, counters per label."""
    counts = {}
    for ph in vault.values():
        label, n = ph[1:-1].rsplit("_", 1)
        counts[label] = max(counts.get(label, 0), int(n))
    for s in sorted(spans, key=lambda s: -s["start"]):
        value = text[s["start"]:s["end"]]
        if value not in vault:
            counts[s["label"]] = counts.get(s["label"], 0) + 1
            vault[value] = f"[{s['label']}_{counts[s['label']]}]"
        s["placeholder"] = vault[value]
        text = text[:s["start"]] + vault[value] + text[s["end"]:]
    return text


class Engine:
    def __init__(self, model_spans=None, threshold=0.8):
        """model_spans: function text -> spans from the neural tagger (None = regex only)."""
        self.model_spans, self.threshold = model_spans, threshold
        self.custom, self.fewshot, self.fewshot_words = {}, None, None                 # user-defined categories: label -> detector, built from a few examples
        self.sessions = {}                                   # session_id -> {placeholder: original}  (memory only, never sent anywhere)

    @classmethod
    def from_checkpoint(cls, tagger_path="checkpoints/tagger.pt", tokenizer_path=None, **kw):
        from privylens.models.encoder_tagger import load_tagger, predict
        from privylens.tokenizer.bpe import BPE
        model = load_tagger(tagger_path)
        tok = BPE.load(tokenizer_path or model.tokenizer_path)      # the tokenizer this model was trained with
        return cls(lambda t: predict(model, tok, t), **kw)

    def add_category(self, name, examples, not_examples=()):
        """Teach a new kind of secret from a few examples. Best: a sentence with the secret in double brackets,
        "Staff code: [[ORX-72841]]" or "[[Nightfall]] is the internal codename for the launch". No training happens; see privylens/fewshot.
        IDs (contain a digit) go to the ID model, capitalised names to the name model (needs a bracketed sentence), and every
        example is also matched literally. Needs checkpoints/fewshot.pt and fewshot_words.pt (python -m privylens.fewshot.run / .words)."""
        from privylens.fewshot.model import detector, load, parse_example, phrase_detector
        from privylens.fewshot.words import tokens, word_detector
        label = re.sub(r"[^A-Z]+", "_", name.upper()).strip("_") or "CUSTOM"        # placeholders look like [LABEL_1]
        pos, neg = [parse_example(e) for e in examples], [parse_example(e) for e in not_examples]
        has_digit = lambda x: any(c.isdigit() for c in x[0][x[1]:x[2]])
        txt = lambda x: x[0][x[1]:x[2]]
        ids = [x for x in pos if has_digit(x)]
        words = [x for x in pos if not has_digit(x)]
        dets = []
        if ids:
            self.fewshot = self.fewshot or load()
            dets.append(detector(self.fewshot, ids, [x for x in neg if has_digit(x)], label=label))
        sentenced = [x for x in words if x[0] != txt(x)]                              # only examples with a sentence around them teach the model
        if sentenced:
            self.fewshot_words = self.fewshot_words or load("checkpoints/fewshot_words.pt")
            dets.append(word_detector(self.fewshot_words, tokens(sentenced), tokens([x for x in neg if not has_digit(x)]), label=label))
        if words:
            dets.append(phrase_detector([txt(x) for x in words], [txt(x) for x in neg if not has_digit(x)], label))
        self.custom[label] = lambda t: [s for d in dets for s in d(t)]
        return label

    def detect(self, text, model=True, custom=True):
        """model / custom = False switch off the neural tagger / the user's categories (used to compare against plain NER)."""
        found = [{**s, "conf": 1.0, "source": "regex"} for s in regex_detect(text)]
        if self.model_spans and model:
            found += [{**s, "source": "model"} for s in snap(text, self.model_spans(text))]
        for d in (self.custom.values() if custom else ()):
            found += [{**s, "source": "custom"} for s in d(text)]
        return merge(found)

    def sanitise(self, text, session_id=None):
        spans = self.detect(text)
        session_id = session_id or uuid.uuid4().hex[:8]
        vault = {v: k for k, v in self.sessions.get(session_id, {}).items()}   # value -> placeholder, kept across calls
        out = redact(text, spans, vault)
        self.sessions[session_id] = {ph: v for v, ph in vault.items()}
        leaks = regex_detect(out)                            # validator: structured PII still visible in the OUTPUT
        return {"sanitised_text": out, "spans": spans, "session_id": session_id,
                "flagged": [s for s in spans if s["conf"] < self.threshold],
                "blocked": bool(leaks), "leaks": leaks}

    def restore(self, text, session_id):
        mapping = self.sessions.get(session_id, {})
        return PLACEHOLDER.sub(lambda m: mapping.get(m.group(), m.group()), text)


if __name__ == "__main__":
    import json
    from privylens.evaluation.metrics import evaluate, report
    eng = Engine.from_checkpoint()
    for split in ("test", "test_unseen_templates", "test_caps"):
        rows = [json.loads(l) for l in open(f"data/splits/{split}.jsonl", encoding="utf-8")]
        report(f"hybrid (regex + tagger) / {split}", evaluate(eng.detect, rows))
