"""Build (original, sanitised) training pairs for the encoder-decoder model from our spans.

  source: Send the report to Rahul Sharma at rahul@example.com.
  target: Send the report to [PERSON] at [EMAIL].

Numbered placeholders ([PERSON_1]) are added later by the engine, not learned by the model.
"""


def make_pair(text, spans):
    out, pos = [], 0
    for s in sorted(spans, key=lambda s: s["start"]):
        out += [text[pos:s["start"]], f"[{s['label']}]"]      # keep the text before the span, then the placeholder
        pos = s["end"]
    return text, "".join(out) + text[pos:]
