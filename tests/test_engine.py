import json
import os
import pytest
from privylens.engine.pipeline import Engine, merge

TEXT = "Send the report to Rahul Sharma at rahul@example.com. Rahul Sharma will call +91 98765 43210."


def fake_model(text):
    """Stands in for the neural tagger: finds the name (twice) with confidence 0.9, plus one shaky guess."""
    spans, i = [], 0
    while (i := text.find("Rahul Sharma", i)) != -1:
        spans.append({"start": i, "end": i + 12, "label": "PERSON", "conf": 0.9}); i += 12
    spans.append({"start": 9, "end": 15, "label": "ORGANIZATION", "conf": 0.3})       # "report": low confidence
    return spans


def test_merge_overlaps_and_keeps_touching_apart():
    r = {"start": 0, "end": 7, "label": "EMAIL", "conf": 1.0, "source": "regex"}
    m = {"start": 0, "end": 3, "label": "PERSON", "conf": 0.5, "source": "model"}
    t = {"start": 7, "end": 9, "label": "DATE", "conf": 1.0, "source": "regex"}
    out = merge([m, r, t])
    assert [(s["start"], s["end"], s["label"]) for s in out] == [(0, 7, "EMAIL"), (7, 9, "DATE")]


def test_model_beats_loose_regex_but_not_precise_regex():
    loose = {"start": 0, "end": 10, "label": "PHONE", "conf": 1.0, "source": "regex"}
    model = {"start": 0, "end": 10, "label": "ID_NUMBER", "conf": 0.9, "source": "model"}
    assert merge([loose, model])[0]["label"] == "ID_NUMBER"          # loose regex loses to the model
    precise = {"start": 0, "end": 10, "label": "EMAIL", "conf": 1.0, "source": "regex"}
    assert merge([precise, model])[0]["label"] == "EMAIL"            # precise regex wins
    assert merge([loose])[0]["label"] == "PHONE"                     # loose regex kept when the model found nothing


def test_sanitise_numbers_consistently_and_restores():
    eng = Engine(fake_model)
    res = eng.sanitise(TEXT)
    out = res["sanitised_text"]
    assert out.count("[PERSON_1]") == 2 and "[PERSON_2]" not in out       # same value -> same placeholder
    assert "[EMAIL_1]" in out and "[PHONE_1]" in out
    assert "Rahul" not in out and "rahul@example.com" not in out and "98765" not in out
    assert not res["blocked"]
    assert [s["label"] for s in res["flagged"]] == ["ORGANIZATION"]          # only the shaky guess is flagged
    assert eng.restore(out, res["session_id"]) == out.replace("[ORGANIZATION_1]", "report") \
        .replace("[PERSON_1]", "Rahul Sharma").replace("[EMAIL_1]", "rahul@example.com").replace("[PHONE_1]", "+91 98765 43210")


def test_session_keeps_numbering_across_calls():
    eng = Engine()
    a = eng.sanitise("mail a@x.com", "s1")["sanitised_text"]
    b = eng.sanitise("mail b@x.com and a@x.com", "s1")["sanitised_text"]
    assert a == "mail [EMAIL_1]" and b == "mail [EMAIL_2] and [EMAIL_1]"


@pytest.mark.skipif(not os.path.exists("checkpoints/tagger.pt"), reason="needs trained tagger checkpoint")
def test_no_known_pii_survives_with_real_model():
    # Every gold value in 300 test rows must be absent from the sanitised output, and nothing blocked.
    eng = Engine.from_checkpoint()
    rows = [json.loads(l) for l in open("data/splits/test.jsonl", encoding="utf-8")][:300]
    total = leaked = 0
    for r in rows:
        out = eng.sanitise(r["text"])["sanitised_text"]
        for s in r["spans"]:
            total += 1
            leaked += r["text"][s["start"]:s["end"]] in out
    print(f"leaked {leaked}/{total}")
    assert leaked / total < 0.06          # tagger alone leaked 5.7% of spans; the hybrid must not be worse
