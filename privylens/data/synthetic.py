"""Fake-data generator.

Takes sentence templates like "Send it to {PERSON} at {EMAIL}." and fills the
{SLOTS} with fake values (Faker, Indian locale). While filling, it records
exactly where each value landed (character start/end) -> these are the "spans"
the model will later learn from. Every value is fake; no real person's data.

Run:  python -m privylens.data.synthetic 5000
Out:  data/synthetic/synthetic.jsonl  (one JSON example per line)
"""
import json
import random
import re
import sys
from faker import Faker

fake = Faker("en_IN")

# ---- 1. Value makers: one small function per entity type -------------------
def _digits(n):
    return "".join(random.choices("0123456789", k=n))

def _letters(n):
    return "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=n))

def _phone():
    # Indian mobiles start with 6-9; show in a few common writing styles
    num = random.choice("6789") + _digits(9)
    return random.choice([num, f"+91 {num}", f"+91-{num[:5]}-{num[5:]}", f"{num[:5]} {num[5:]}"])

def _aadhaar():
    # 12 digits, first digit 2-9, usually written in groups of 4
    n = random.choice("23456789") + _digits(11)
    return random.choice([n, f"{n[:4]} {n[4:8]} {n[8:]}"])

def _date():
    d = fake.date_object()
    return d.strftime(random.choice(["%d/%m/%Y", "%d-%m-%Y", "%d %B %Y", "%Y-%m-%d"]))

MAKERS = {
    "PERSON":       fake.name,
    "ORGANIZATION": fake.company,
    "LOCATION":     fake.city,
    "EMAIL":        fake.email,
    "PHONE":        _phone,
    "PAN":          lambda: _letters(5) + _digits(4) + _letters(1),  # format AAAAA9999A
    "AADHAAR":      _aadhaar,
    "URL":          lambda: f"https://www.{fake.domain_name()}/{_letters(5).lower()}",
    "IP_ADDRESS":   fake.ipv4,
    "DATE":         _date,
    "ID_NUMBER":    lambda: random.choice(["EMP", "CASE", "ACC", "INV"]) + _digits(random.choice([4, 6])),
}

# ---- 2. Templates: (template_id is its index, used later for unseen-template split)
TEMPLATES = [
    "Send the report to {PERSON} at {EMAIL}.",
    "Please call {PERSON} on {PHONE} before noon.",
    "My name is {PERSON} and I live in {LOCATION}.",
    "{PERSON} works at {ORGANIZATION} in {LOCATION}.",
    "His employee ID is {ID_NUMBER} and his phone is {PHONE}.",
    "PAN of the applicant: {PAN}.",
    "Aadhaar number {AADHAAR} belongs to {PERSON}.",
    "Visit {URL} for more details.",
    "The request came from IP {IP_ADDRESS} on {DATE}.",
    "Invoice {ID_NUMBER} was issued to {ORGANIZATION} on {DATE}.",
    "Dear {PERSON}, your account {ID_NUMBER} has been updated.",
    "Hi, this is {PERSON} from {ORGANIZATION}. Reach me at {EMAIL} or {PHONE}.",
    "{PERSON} was born on {DATE} in {LOCATION}.",
    "Case {ID_NUMBER} is assigned to officer {PERSON}.",
    "Please verify PAN {PAN} and Aadhaar {AADHAAR} for {PERSON}.",
    "We are meeting {PERSON} of {ORGANIZATION} in {LOCATION} on {DATE}.",
    "Forward this mail to {EMAIL} and cc {EMAIL}.",
    "The server at {IP_ADDRESS} was accessed by {PERSON}.",
    "Mr. {PERSON} lives near {LOCATION}; contact {PHONE}.",
    "Shipment {ID_NUMBER} will reach {LOCATION} by {DATE}.",
    "Complaint filed by {PERSON} against {ORGANIZATION}.",
    "Login from {IP_ADDRESS} for user {EMAIL}.",
    "{ORGANIZATION} has an office in {LOCATION}.",
    "Thanks {PERSON}! I will update the record for {ID_NUMBER}.",
    "Witness statement of {PERSON}, resident of {LOCATION}, dated {DATE}.",
    "Mobile: {PHONE}, Email: {EMAIL}, PAN: {PAN}.",
    "Transfer to account {ID_NUMBER} held by {PERSON}.",
    "Check the policy at {URL} before sharing {AADHAAR}.",
    "Ask {PERSON} or {PERSON} to approve it.",
    "The meeting with {ORGANIZATION} is on {DATE} in {LOCATION}.",
    "Can you summarise this note from {PERSON} ({EMAIL})?",
    "Patient {PERSON}, DOB {DATE}, ID {ID_NUMBER}, admitted in {LOCATION}.",
    "Kindly contact {PERSON} at {PHONE} regarding case {ID_NUMBER}.",
    "{PERSON} joined {ORGANIZATION} on {DATE}.",
    "Our team in {LOCATION} will call {PHONE} to confirm.",
    "Profile link: {URL} - owner {PERSON}.",
    "The suspect, {PERSON}, was last seen in {LOCATION} on {DATE}.",
    "Use PAN {PAN} to file the return for {ORGANIZATION}.",
    "Hello {PERSON}, we sent an OTP to {PHONE}.",
    "Address proof for {PERSON}: Aadhaar {AADHAAR}, city {LOCATION}.",
    "Ticket {ID_NUMBER} raised by {EMAIL} from {IP_ADDRESS}.",
]

# ---- 3. Fill one template, tracking where each value lands -----------------
def fill(template_id):
    # split "a {X} b" -> ["a ", "X", " b"]; odd positions are slot names
    parts = re.split(r"\{(\w+)\}", TEMPLATES[template_id])
    text, spans = "", []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            text += part                      # plain text, copy as-is
        else:
            value = MAKERS[part]()            # make a fake value
            spans.append({"start": len(text), "end": len(text) + len(value), "label": part})
            text += value
    return text, spans

def generate(n):
    for i in range(n):
        tid = random.randrange(len(TEMPLATES))
        text, spans = fill(tid)
        yield {"id": f"syn_{i:06d}", "text": text, "spans": spans,
               "source": "synthetic", "template_id": f"T{tid:03d}"}

# ---- 4. Self-check + write file ---------------------------------------------
if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    random.seed(0); Faker.seed(0)             # same output every run
    rows = list(generate(n))
    for r in rows:                            # spans must point at real, non-empty text
        for s in r["spans"]:
            assert r["text"][s["start"]:s["end"]].strip(), r
    with open("data/synthetic/synthetic.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"wrote {n} examples, {len(TEMPLATES)} templates")
    for r in rows[:3]:
        print(r["text"], [(r["text"][s["start"]:s["end"]], s["label"]) for s in r["spans"]])
