import random
from privylens.engine.pipeline import Engine, merge
from privylens.fewshot import baselines as B
from privylens.fewshot.data import CUES_TEST, candidates, make_company, rows
from privylens.fewshot.model import parse_example, train_proto


def test_candidates_cover_every_gold_id_and_skip_plain_words():
    c = make_company(random.Random(1), CUES_TEST)
    for r in rows(c, 30, random.Random(2)):
        assert all((s["start"], s["end"]) in candidates(r["text"]) for s in r["spans"])
    assert candidates("Please call John about the report") == []


def test_parse_example_marks_value_inside_sentence():
    assert parse_example("Staff code: [[ORX-72841]]") == ("Staff code: ORX-72841", 12, 21)
    assert parse_example("ORX-72841") == ("ORX-72841", 0, 9)


def test_induced_regex_learns_shape_not_value():
    d = B.induce_regex([parse_example("ORX-72841")])
    t = "id ORX-11111 and EMP_N7Q291 and ORX-1111"
    assert [t[s["start"]:s["end"]] for s in d(t)] == ["ORX-11111"]


def test_tiny_model_and_engine_category(tmp_path, monkeypatch):
    import torch
    m = train_proto([make_company(random.Random(i), CUES_TEST) for i in range(4)], steps=3, log=False)
    torch.save(m.state_dict(), tmp_path / "f.pt")
    monkeypatch.setattr("privylens.fewshot.model.load.__defaults__", (str(tmp_path / "f.pt"),))
    e = Engine()
    label = e.add_category("Employee ID!", ["Staff code: [[ORX-72841]]", "ORX-10442"])
    assert label == "EMPLOYEE_ID" and e.detect("Staff code ORX-55555 ok") is not None   # untrained net: only checks the plumbing
    a = {"start": 0, "end": 5, "label": "EMPLOYEE_ID", "source": "custom", "conf": 1.0}
    b = {"start": 0, "end": 5, "label": "EMAIL", "source": "regex", "conf": 1.0}
    assert merge([b, a])[0]["label"] == "EMPLOYEE_ID"                                    # user category beats precise regex
