from privylens.baselines.regex_detector import detect

def found(text):
    return {text[s["start"]:s["end"]]: s["label"] for s in detect(text)}

def test_structured():
    f = found("Mail a@b.com, PAN ABCDE1234F, Aadhaar 2345 6789 0123, call +91 98765 43210 on 25/12/2020 re EMP4821 via http://x.in/a.")
    assert f == {"a@b.com": "EMAIL", "ABCDE1234F": "PAN", "2345 6789 0123": "AADHAAR",
                 "+91 98765 43210": "PHONE", "25/12/2020": "DATE", "EMP4821": "ID_NUMBER", "http://x.in/a": "URL"}

def test_no_false_positive():
    assert detect("Meet me at 5 pm, room 12.") == []


def test_written_out_dates():
    f = found("Issued Oct 15, 2026 and 15th October 2026 and 3 Sept 2025 but not May 12.")
    assert f == {"Oct 15, 2026": "DATE", "15th October 2026": "DATE", "3 Sept 2025": "DATE"}
