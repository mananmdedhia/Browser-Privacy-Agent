from privylens.tokenizer.bpe import BPE

TEXTS = ["Send the report to Rahul Sharma at rahul@example.com.", "Call Rahul on 98765 43210 about [PERSON] and the report."] * 20


def test_round_trip_and_offsets():
    tok = BPE.train(TEXTS, 120)
    for t in TEXTS[:2] + ["Call Sharma about the 98765 report."]:
        ids, offs = tok.encode(t)
        assert tok.decode(ids) == t                      # text survives encode->decode
        assert all(t[a:b] for a, b in offs)              # every token maps to real text
        assert [t[a:b] for a, b in offs] == [tok.inv[i].replace("▁", "") for i in ids]


def test_merges_happened():
    tok = BPE.train(TEXTS, 200)
    ids, _ = tok.encode("rahul@example.com.")
    assert len(ids) < len("rahul@example.com.")        # merges compressed it
