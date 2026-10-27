from fastapi.testclient import TestClient
from api.server import app

client = TestClient(app)


def test_sanitise_then_restore_round_trip():
    text = "Mail rahul@example.com about PAN ABCDE1234F."
    r = client.post("/sanitise", json={"text": text}).json()
    assert "rahul@example.com" not in r["sanitised_text"] and "ABCDE1234F" not in r["sanitised_text"]
    assert not r["blocked"]
    back = client.post("/restore", json={"text": r["sanitised_text"], "session_id": r["session_id"]}).json()
    assert back["restored_text"] == text


def test_compare_has_every_engine_and_distilbert_is_optional():
    r = client.post("/compare", json={"text": "Mail rahul@example.com"}).json()
    assert set(r) == {"regex", "ner", "privylens", "distilbert"}
    assert "rahul@example.com" not in r["regex"]["sanitised_text"]
