from privylens.schema import ID2TAG
from privylens.tokenizer.align import to_bio, to_spans
from privylens.tokenizer.bpe import BPE


def test_bio_and_round_trip():
    text = "Send to Rahul Sharma at rahul@example.com."
    spans = [{"start": 8, "end": 20, "label": "PERSON"}, {"start": 24, "end": 41, "label": "EMAIL"}]
    assert text[8:20] == "Rahul Sharma" and text[24:41] == "rahul@example.com"
    tok = BPE.train([text] * 5, 60)
    ids, offs = tok.encode(text)
    tags = to_bio(offs, spans)
    names = [ID2TAG[t] for t in tags]
    assert names[0] == "O" and "I-PERSON" in names and names.count("B-PERSON") == 1
    assert to_spans(offs, tags) == spans              # spans survive tokens -> tags -> spans
