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

PLACEHOLDER = re.compile(r"\[[A-Z_]+_\d+\]")


STRONG = {"EMAIL", "URL", "IP_ADDRESS", "PAN"}      # formats regex gets (almost) always right


def merge(spans):
    """Pick non-overlapping spans. Priority when two overlap:
       1. precise regex (STRONG types)   2. the neural tagger   3. loose regex (phone, Aadhaar, date, ID)
    then the longer span. Loose regex still counts where the model found nothing (recall).
    ponytail: the loser is dropped whole, so a partial overlap can leave a few characters visible;
    widen the winner to the union if the leak metric shows it matters."""
    rank = lambda s: 2 if s["source"] == "regex" and s["label"] in STRONG else 1 if s["source"] == "model" else 0
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
        self.sessions = {}                                   # session_id -> {placeholder: original}  (memory only, never sent anywhere)

    @classmethod
    def from_checkpoint(cls, tagger_path="checkpoints/tagger.pt", tokenizer_path="tokenizers/bpe_8k.json", **kw):
        from privylens.models.encoder_tagger import load_tagger, predict
        from privylens.tokenizer.bpe import BPE
        model, tok = load_tagger(tagger_path), BPE.load(tokenizer_path)
        return cls(lambda t: predict(model, tok, t), **kw)

    def detect(self, text):
        found = [{**s, "conf": 1.0, "source": "regex"} for s in regex_detect(text)]
        if self.model_spans:
            found += [{**s, "source": "model"} for s in self.model_spans(text)]
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
    for split in ("test", "test_unseen_templates"):
        rows = [json.loads(l) for l in open(f"data/splits/{split}.jsonl", encoding="utf-8")]
        report(f"hybrid (regex + tagger) / {split}", evaluate(eng.detect, rows))
